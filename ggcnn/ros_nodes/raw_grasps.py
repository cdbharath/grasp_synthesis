#!/usr/bin/env python3

from encodings import normalize_encoding
import rospy
import numpy as np
import math
import cv2
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt

from ggcnn.ggcnn_torch import predict, process_depth_image
from ggcnn.srv import GraspPrediction, GraspPredictionResponse  

import cv_bridge
bridge = cv_bridge.CvBridge()

class GraspService:
    def __init__(self, sim_mode=False, crop=True):
        self.sim_mode = sim_mode
        self.crop = crop
        # Full image: [0, 0, 720, 1280]
        # Full image: [50, 0, 480, 640]

        self.crop_size = [0, 0, 720, 1280]

        if self.sim_mode:
            rospy.Subscriber("", Image, self.rgb_cb)
            rospy.Subscriber("", Image, self.depth_cb)
        else:
            rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_cb)
            rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb)

        rospy.Service('debug', GraspPrediction, self.service_cb)

        self.rgb_cropped_pub = rospy.Publisher("cropped_rgb", Image, queue_size=10)
        self.depth_cropped_pub = rospy.Publisher("cropped_depth", Image, queue_size=10) 

        self.curr_depth_img = None
        self.curr_rgb_img = None

    def depth_cb(self, msg):
        img = bridge.imgmsg_to_cv2(msg)
        if self.crop:
            self.curr_depth_img = img[self.crop_size[0]:self.crop_size[2], self.crop_size[1]:self.crop_size[3]]
            # normalized = cv2.normalize(self.curr_depth_img, None, alpha=0, beta=10, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            depth_crop = self.curr_depth_img.copy()
            depth_scale = np.abs(depth_crop).max()
            depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.
            normalized = (depth_crop*255).astype('uint8')
            self.depth_cropped_pub.publish(bridge.cv2_to_imgmsg(normalized))
        else:
            self.curr_depth_img = img 
        self.received = True

    def rgb_cb(self, msg):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.crop:
            self.curr_rgb_img = img[self.crop_size[0]:self.crop_size[2], self.crop_size[1]:self.crop_size[3], :]
            self.rgb_cropped_pub.publish(bridge.cv2_to_imgmsg(self.curr_rgb_img, encoding='bgr8'))
        else:
            self.curr_rgb_img = img

    def service_cb(self, data):
        depth = self.curr_depth_img
        rgb = self.curr_rgb_img

        #####################################################################
        # Insert your algorithm specific code here

        depth_crop, depth_nan_mask = process_depth_image(depth, depth.shape[0], 300, return_mask=True, crop_y_offset=0)
        points, angle, width_img, _ = predict(depth_crop, process_depth=False, depth_nan_mask=depth_nan_mask, filters=(2.0, 2.0, 2.0))

        x, y = np.unravel_index(np.argmax(points), points.shape)
        ang = angle[x][y]

        response = GraspPredictionResponse()
        g = response.best_grasp
        # Scale detection for correct 3D transformation
        g.pose.position.x = int(x*depth.shape[0]/300)
        g.pose.position.y = int(y*depth.shape[0]/300 + (depth.shape[0] - 300)/2)
        g.pose.orientation.z = ang
        g.width = int(width_img[x][y]*depth.shape[0]/300)
        # g.quality = points[x][y]

        print(g.pose.position.x, g.pose.position.y, g.pose.orientation.z, g.width)
        bb = self.draw_angled_rect(rgb, g.pose.position.y, g.pose.position.x, g.pose.orientation.z)

        cv2.imshow('bb', bb)
        norm_image = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow('depth', norm_image)
        cv2.imshow('points', points)
        cv2.imshow('angle', angle)
        cv2.waitKey(0)

        ########################################################################

        return response

    def draw_angled_rect(self, image, x, y, angle, width = 220, height = 100):
        print(x, y, angle, image.shape)
        _angle = -angle
        b = math.cos(_angle) * 0.5
        a = math.sin(_angle) * 0.5

        # gray_image = image.copy()
        # display_image = cv2.applyColorMap((gray_image * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        display_image = image.copy()

        pt0 = (int(x - a * height - b * width), int(y + b * height - a * width))
        pt1 = (int(x + a * height - b * width), int(y - b * height - a * width))
        pt2 = (int(2 * x - pt0[0]), int(2 * y - pt0[1]))
        pt3 = (int(2 * x - pt1[0]), int(2 * y - pt1[1]))

        cv2.line(display_image, pt0, pt1, (255, 0, 0), 5)
        cv2.line(display_image, pt1, pt2, (0, 0, 0), 5)
        cv2.line(display_image, pt2, pt3, (255, 0, 0), 5)
        cv2.line(display_image, pt3, pt0, (0, 0, 0), 5)
        cv2.circle(display_image, ((pt0[0] + pt2[0])//2, (pt0[1] + pt2[1])//2), 3, (0, 0, 0), -1)
        return display_image

if __name__ == '__main__':
    rospy.init_node('grasp_service')
    GraspService()
    rospy.spin()
