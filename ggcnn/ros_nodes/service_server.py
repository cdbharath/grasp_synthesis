#!/usr/bin/env python

import rospy
import numpy as np
import math
import cv2
# To use Tensorflow implementation
# from ggcnn.ggcnn import predict, process_depth_image
# To use Pytorch implementation

from ggcnn.ggcnn_torch import predict, process_depth_image
from ggcnn.srv import Grasp2DPrediction, Grasp2DPredictionResponse, Grasp2DPredictionRequest

import cv_bridge
bridge = cv_bridge.CvBridge()

class GraspService:
    def __init__(self):
        rospy.Service('~predict', Grasp2DPrediction, self.service_cb)

    def service_cb(self, data):
        depth = bridge.imgmsg_to_cv2(data.depth_image,"32FC1")
        # print(depth)
        # print(cv_image_array)
        # print(cv_image_norm)
        # depth_square = process_depth_image(depth)
        # print(depth)
        # print(depth_square)
        # cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 255, cv2.NORM_MINMAX)
            
        depth_crop, depth_nan_mask = process_depth_image(depth, depth.shape[0], 300, return_mask=True, crop_y_offset=0)
        # cv_image_array = np.array(depth_crop, dtype = np.uint8)
        # cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow("depth", cv_image_norm)
        # cv2.waitKey(20000)
        # cv2.destroyAllWindows()
        points, angle, width_img, _ = predict(depth_crop, process_depth=False, depth_nan_mask=depth_nan_mask, filters=(2.0, 2.0, 2.0))

        x, y = np.unravel_index(np.argmax(points), points.shape)
        ang = angle[x][y]

        response = Grasp2DPredictionResponse()
        g = response.best_grasp

        # Scale detection for correct 3D transformation
        g.px = int(x*depth.shape[0]/300)
        g.py = int(y*depth.shape[0]/300 + (depth.shape[0] - 300)/2)
        g.angle = ang
        g.width = int(width_img[x][y]*depth.shape[0]/300)
        g.quality = points[x][y]

        print(g.px, g.py, depth.shape)
        print(x, y, depth.shape)

        return response

if __name__ == '__main__':
    rospy.init_node('grasp_service')
    GraspService()
    rospy.spin()


