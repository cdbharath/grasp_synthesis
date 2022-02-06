#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np

from copy import deepcopy
from PIL import Image as PILImage
from cv_bridge import CvBridge, CvBridgeError

IMAGE_TOPIC = "/kinect/rgb/image_raw"
DEPTH_TOPIC = "/kinect/depth/image_raw"


def get_image(show=False):
    #print("CALLING GET_KINECT_IMAGE")
    rospy.init_node("kinect_subscriber")
    rgb = rospy.wait_for_message(IMAGE_TOPIC, Image)
    depth = rospy.wait_for_message(DEPTH_TOPIC, Image)

    rgb = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)
    depth = np.frombuffer(depth.data, dtype=np.uint8).reshape(depth.height, depth.width, -1)
    
    # Convert sensor_msgs.Image readings into readable format
    # bridge = CvBridge()
    # rgb = bridge.imgmsg_to_cv2(rgb, rgb.encoding)
    # depth = bridge.imgmsg_to_cv2(depth, depth.encoding)

    image = deepcopy(rgb)
    # image[:, :, 2] = depth
    if (show):
        im = PILImage.fromarray(image, 'RGB')
        im.show()

    return image


if __name__ == '__main__':
    image = get_image(show=True)

