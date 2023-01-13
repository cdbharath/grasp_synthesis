#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

class Nodo(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Publishers
        self.pub = rospy.Publisher('imagetimer', Image,queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",Image,self.Depthcallback)

    def callback(self, msg):
        rospy.loginfo('Image received...')
        self.image = self.br.imgmsg_to_cv2(msg)
        print(self.image)
        cv2.imshow("depth", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Depthcallback(self,msg_depth): # TODO still too noisy!
        try:
            # The depth image is a single-channel float32 image
            # the values is the distance in mm in z axis
            cv_image = self.br.imgmsg_to_cv2(msg_depth, "32FC1")
            print(cv_image)
            
            

            # Convert the depth image to a Numpy array since most cv2 functions
            # require Numpy arrays.
            # Max_z = 1000
            cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
            # cv_image_array = np.clip(cv_image_array, 0, Max_z)
            # print(cv_image_array)
            # Normalize the depth image to fall between 0 (black) and 1 (white)
            # http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
            cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 255, cv2.NORM_MINMAX)
            # Resize to the desired size
            self.depthimg = cv_image_norm.astype('uint8')
            # blur = cv2.medianBlur(self.depthimg,5)
            cv2.imshow("Image from my node", self.depthimg)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
        except CvBridgeError as e:
            print(e)

    # def start(self):
    #     rospy.loginfo("Timing images")
    #     #rospy.spin()
    #     while not rospy.is_shutdown():
    #         rospy.loginfo('publishing image')
    #         #br = CvBridge()
    #         if self.image is not None:
    #             self.pub.publish(br.cv2_to_imgmsg(self.image))
    #         self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node("visualize", anonymous=True)
    my_node = Nodo()
    rospy.spin()
    # my_node.start()