from os import path
import sys

import cv2
import numpy as np
import scipy.ndimage as ndimage

import torch

MODEL_FILE = 'models/ggcnn_epoch_23_cornell'

here = path.dirname(path.abspath(__file__))
sys.path.append(here)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(path.join(path.dirname(__file__), MODEL_FILE), map_location=device)

def process_depth_image(depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):
    imh, imw = depth.shape
    
    # Crop image from v*u to v*v
    depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                        (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
 
    # Inpaint the depth image
    # OpenCV inpainting does weird things at the border.
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    depth_crop[depth_nan_mask==1] = 0

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    depth_scale = np.abs(depth_crop).max()
    depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

    # Inpaint
    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale

    # Resize cropped depth image 
    depth_crop = cv2.resize(depth_crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    if return_mask:
        # Resize Nan mask 
        depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
        depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        return depth_crop, depth_nan_mask
    else:
        return depth_crop


def predict(depth, process_depth=True, crop_size=300, out_size=300, depth_nan_mask=None, crop_y_offset=0, filters=(2.0, 1.0, 1.0)):
    if process_depth:
        depth, depth_nan_mask = process_depth_image(depth, crop_size, out_size=out_size, return_mask=True, crop_y_offset=crop_y_offset)

    # Inference
    depth = np.clip((depth - depth.mean()), -1, 1)

    depthT = torch.from_numpy(depth.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)
    with torch.no_grad():
        pred_out = model(depthT)

    points_out = pred_out[0].cpu().numpy().squeeze()
    points_out[depth_nan_mask] = 0

    # Calculate the angle map.
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0

    width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

    # Filter the outputs.
    if filters[0]:
        points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
    if filters[1]:
        ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
    if filters[2]:
        width_out = ndimage.filters.gaussian_filter(width_out, filters[2])

    points_out = np.clip(points_out, 0.0, 1.0-1e-3)

    return points_out, ang_out, width_out, depth.squeeze()
