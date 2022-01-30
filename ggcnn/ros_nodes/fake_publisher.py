#!/usr/bin/env python2

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospkg

bridge = CvBridge()
rospack = rospkg.RosPack()

def image_publisher_cb(timer):
    pkg_path = rospack.get_path('ggcnn')
    cv_image = cv2.imread(pkg_path + "/data/ggcnn_test1.png")
    
    image_message = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
    image_pub.publish(image_message)
    
if __name__ == "__main__":
    rospy.init_node("image_publisher")
    image_pub = rospy.Publisher("/panda_camera/depth/image_raw", Image, queue_size=10)
    rospy.Timer(rospy.Duration(0.1), image_publisher_cb)
    # rospy.logwarn("Timer Initiated")
    rospy.spin()