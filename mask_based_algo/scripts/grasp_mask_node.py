#!/usr/bin/env python

import rospy
import cv2

from mask_based_algo_module.grasp_mask import GraspMask

if __name__ == '__main__':
    rospy.init_node('grasp_mask_node')
    
    depth_image = cv2.imread('image1.png', cv2.IMREAD_UNCHANGED)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)    
    
    grasp_mask = GraspMask()
    # grasp_mask.get_grasp(depth_image)
    
    rospy.spin()