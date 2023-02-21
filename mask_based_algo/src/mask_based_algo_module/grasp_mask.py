import rospy
import cv2
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image

class GraspMask:
    '''
    Calulates the grasp for a given depth image using mask based algorithm.
    '''
    def __init__(self):
        self.top_k = 1
        self.bridge = cv_bridge.CvBridge()
        
        # Create masks of different sizes
        positive_mask = [60]
        negative_mask = [20]
        assert len(positive_mask) == len(negative_mask)
        
        self.masks = []        
        for i in range(len(positive_mask)):
            top_mask = np.ones((negative_mask[i], positive_mask[i])) * -1
            middle_mask = np.ones((positive_mask[i], positive_mask[i])) * 1
            bottom_mask = np.ones((negative_mask[i], positive_mask[i])) * -1
            mask = np.concatenate((top_mask, middle_mask, bottom_mask), axis=0) / (positive_mask[i] * (positive_mask[i] + negative_mask[i] * 2))
            self.masks.append(mask)

        rospy.Subscriber('/panda_camera/depth/image_raw', Image, self.depth_image_callback)

    def depth_image_callback(self, depth_image_msg):
        depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding='passthrough') 
        self.get_grasp(depth_image)

    def normalize_depth(self, depth_image):
        '''
        Normalizes the depth image to be between 0 and 255.
        
        :param depth_image: The depth image to normalize.
        :return normalized_depth_image: The normalized depth image.
        '''
        normalized_depth_image = (depth_image - np.min(depth_image)) * 255 / (np.max(depth_image) - np.min(depth_image))
        normalized_depth_image = np.uint8(normalized_depth_image)
        
        return normalized_depth_image

    def get_grasp(self, depth_image):
        original_depth_image = depth_image.copy()
    
        original_depth_image_norm = self.normalize_depth(depth_image)
        
        depth_image = original_depth_image_norm.copy()
        depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
        _, depth_image = cv2.threshold(depth_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)        
        contours, _ = cv2.findContours(depth_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        contours_image = np.ones(depth_image.shape, np.uint8)*255
        contours_image = cv2.drawContours(contours_image, [largest_contour], -1, (0,255,0), 3)
        
        major_directions, contour_mean, major_components_image = self.get_major_directions(largest_contour, depth_image)
        
        affine_trans = cv2.getRotationMatrix2D(contour_mean, np.arctan2(major_directions[0, 1], major_directions[0, 0]) * 180 / np.pi, 1.0)
        depth_rotated = cv2.warpAffine((255-original_depth_image_norm), affine_trans, dsize=(depth_image.shape[1], depth_image.shape[0]))
                
        filtered_rotated = depth_rotated.copy()
        
        filtered_rotated = cv2.filter2D(filtered_rotated, -1, self.masks[0])
        max_idx = np.argmax(filtered_rotated)
        max_loc = np.unravel_index(max_idx, filtered_rotated.shape)
                
        filtered_rotated = cv2.circle(filtered_rotated, (max_loc[1], max_loc[0]), 10, 255, -1)
        
        cv2.imshow('major_components', major_components_image)
        cv2.imshow('rotated_major_components', filtered_rotated)
        cv2.imshow('depth_rotated', depth_rotated)

        cv2.waitKey(0)
        
    def get_major_directions(self, largest_contour, depth_image):
        '''
        Finds major components of the input contour.
        
        :param largest_contour: The contour to find the major components of.
        :param depth_image: The depth image the contour is from.
        
        :return major_directions: The major components of the contour.
        :return mean_flattened_contour: The mean of the contour.
        :return major_components_image: The image with the major components drawn on it.
        '''
        flattened_contour = np.float32(largest_contour.reshape(-1, 2))
        mean_flattened_contour = np.mean(flattened_contour, axis=0)
        centered_contour = flattened_contour - mean_flattened_contour
        covariance_matrix = np.cov(centered_contour, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[: : -1]
        sorted_eigenvecs = eigenvectors[:, sorted_indices]
        
        major_directions = sorted_eigenvecs[:, :self.top_k]
        
        major_components_image = cv2.cvtColor(depth_image.copy(), cv2.COLOR_GRAY2BGR)
        for direction in major_directions.T:
            start = tuple(np.int32(mean_flattened_contour))
            end = tuple(np.int32(mean_flattened_contour + 100 * direction))

            cv2.arrowedLine(major_components_image, start, end, (0, 0, 255), 2)

        return major_directions.T, mean_flattened_contour, major_components_image