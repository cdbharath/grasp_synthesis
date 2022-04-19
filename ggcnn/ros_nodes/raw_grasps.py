#!/usr/bin/env python3

#!/usr/bin/env python3

import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt

from ggcnn.ggcnn_torch import predict, process_depth_image
from ggcnn.srv import Grasp2DPrediction, Grasp2DPredictionResponse, Grasp2DPredictionRequest

import cv_bridge
bridge = cv_bridge.CvBridge()

class GraspService:
    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode

        if self.sim_mode:
            rospy.Subscriber("", Image, self.rgb_cb)
            rospy.Subscriber("", Image, self.depth_cb)
        else:
            rospy.Subscriber("", Image, self.rgb_cb)
            rospy.Subscriber("", Image, self.depth_cb)

        self.curr_depth_img = None
        self.curr_rgb_img = None

    def depth_cb(self, msg):
        self.curr_depth_img = bridge.imgmsg_to_cv2(msg)
        self.received = True

    def rgb_cb(self, msg):
        self.curr_rgb_img = bridge.imgmsg_to_cv2(msg)

    def service_cb(self, data):
        depth = self.curr_depth_img
        rgb = self.curr_rgb_img

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

        print(g.px, g.py, g.angle, g.width)

        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(rgb)
        axarr[0,1].imshow(depth)
        axarr[1,0].imshow(points)
        axarr[1,1].imshow(angle)
        plt.show()

        return response

if __name__ == '__main__':
    rospy.init_node('grasp_service')
    GraspService()
    rospy.spin()
