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
        depth = bridge.imgmsg_to_cv2(data.depth_image)
        
        depth_crop, depth_nan_mask = process_depth_image(depth, depth.shape[0], 300, return_mask=True, crop_y_offset=0)
        points, angle, width_img, _ = predict(depth_crop, process_depth=False, depth_nan_mask=depth_nan_mask, filters=(2.0, 2.0, 2.0))

        x, y = np.unravel_index(np.argmax(points), points.shape)
        ang = angle[x][y]

        response = Grasp2DPredictionResponse()
        g = response.best_grasp
        g.px = int(x*depth.shape[0]/300)
        g.py = int(y*depth.shape[1]/300)
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


