#!/usr/bin/env python3
#################################################################################

import rospy
from sensor_msgs.msg import Image, CameraInfo
import cv_bridge
import cv2
import numpy as np
import tf
import copy



class DepthMaskingNode:
    def __init__(self):
        rospy.init_node('depth_masking_node', anonymous=True)
        self.bridge = cv_bridge.CvBridge()
        self.tf_listener = tf.TransformListener()

        # Placeholder variables for camera intrinsics
        self.depth_camera_intrinsics = None
        self.rgb_camera_intrinsics = None

        self.rgb_distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.depth_distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Placeholder variable for mask image
        self.mask_image = None
        self.mask_image_header = None

        self.rgb_image = None

        # Fetch ROS params
        self.__depth_in_topic = rospy.get_param("~depth_in_topic", "/camera/depth/image_rect_raw")
        self.__image_in_topic = rospy.get_param("~image_in_topic", "/camera/color/image_raw")
        self.__mask_in_topic = rospy.get_param("~mask_in_topic", "/camera/color/mask")
        self.__depth_camera_info_topic = rospy.get_param("~depth_camera_info_topic", "/camera/depth/camera_info")
        self.__rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/color/camera_info")

        self.__masked_depth_pub_topic = rospy.get_param("~masked_depth_pub_topic", "/camera/depth/image_raw_masked")
        self.__masked_rgb_pub_topic = rospy.get_param("~masked_rgb_pub_topic", "/camera/color/image_raw_masked")

        self.__rgb_image_frame_id = rospy.get_param("~rgb_image_frame_id", "camera_color_optical_frame")
        self.__depth_image_frame_id = rospy.get_param("~depth_image_frame_id", "camera_depth_optical_frame")

        self.__aligned_depth_rgb = rospy.get_param("~aligned_depth_rgb", "False")

        self.__rectangle_mask = rospy.get_param("~rectangle_mask", "False")
        self.__x_lower = rospy.get_param("~x_lower", 200)
        self.__x_upper = rospy.get_param("~x_upper", 300)
        self.__y_lower = rospy.get_param("~y_lower", 200)
        self.__y_upper = rospy.get_param("~y_upper", 300)


        # Subscribe to topics
        rospy.Subscriber(self.__depth_in_topic, Image, self.callback_depth)
        rospy.Subscriber(self.__image_in_topic, Image, self.callback_rgb)
        rospy.Subscriber(self.__mask_in_topic, Image, self.callback_mask)
        rospy.Subscriber(self.__depth_camera_info_topic, CameraInfo, self.callback_depth_intrinsics)
        rospy.Subscriber(self.__rgb_camera_info_topic, CameraInfo, self.callback_rgb_intrinsics)


        # ROS Publisher
        self.depth_pub = rospy.Publisher(self.__masked_depth_pub_topic, Image, queue_size=10)
        self.rgb_pub = rospy.Publisher(self.__masked_rgb_pub_topic, Image, queue_size=10)


    def callback_depth(self, depth_msg):
        # Convert ROS image to OpenCV image
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        rgb_img = copy.copy(self.rgb_image)

        # Apply mask on rgb and depth images
        if self.rgb_image is not None and self.mask_image is not None:
            if self.__aligned_depth_rgb:
                # print("Param was true")
                masked_depth_image, masked_rgb_image = self.apply_mask_aligned_depth_rgb(depth_image, rgb_img, self.mask_image)
            else:
                masked_depth_image = self.apply_mask_2(depth_image, self.mask_image, self.depth_camera_intrinsics, self.rgb_camera_intrinsics)
        else:  # if either the mask or rgb image was not available: 
            rospy.logwarn("Not all messages have been recieved yet. Waiting...")
            return
        
        # Publish the masked depth image
        masked_depth_msg = self.bridge.cv2_to_imgmsg(masked_depth_image, encoding='passthrough')
        masked_rgb_msg = self.bridge.cv2_to_imgmsg(masked_rgb_image, encoding="bgr8")
        # Pass headers from depth image message recieved to ensure synchronization:
        masked_depth_msg.header = depth_msg.header
        masked_rgb_msg.header = depth_msg.header

        # Publish masked images
        self.depth_pub.publish(masked_depth_msg)
        self.rgb_pub.publish(masked_rgb_msg)



    def callback_rgb(self, rgb_msg):
        # Convert ROS image to OpenCV image
        self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        self.rgb_image_header = rgb_msg.header



    def callback_mask(self, mask_msg):
        # Convert ROS image to OpenCV image
        self.mask_image = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        self.mask_image_header = mask_msg.header



    def callback_depth_intrinsics(self, depth_info_msg): 
        # Save depth camera intrinsics for later use
        # We don't really care about synchro since those params don't change
        # the same goes to the rgb intrinsics messages
        self.depth_camera_intrinsics = np.array(depth_info_msg.K).reshape((3, 3))



    def callback_rgb_intrinsics(self, rgb_info_msg):
        # Save RGB camera intrinsics for later use
        self.rgb_camera_intrinsics = np.array(rgb_info_msg.K).reshape((3, 3))



    def apply_mask_aligned_depth_rgb(self, depth_image, rgb_image, mask_image):
        # Mask the depth image first, easy, both depth and mask images are grayscales.
        mask_image[np.where(mask_image == 255)] = 1 # replace 255 by 1. Unit rescale of the mask
        # print("mask image: {}".format(mask_image))
        masked_depth_image = np.multiply(depth_image, mask_image)

        # Now mask the rgb image
        # rgb_mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        masked_rgb_image = cv2.bitwise_and(rgb_image, rgb_image, mask = mask_image)

        if self.__rectangle_mask:
            mask = np.zeros_like(mask_image)
            mask[self.__y_lower:self.__y_upper, self.__x_lower:self.__x_upper] = 1
            masked_depth_image = masked_depth_image * (1 - mask)

        return masked_depth_image, masked_rgb_image



#     def apply_mask(self, depth_image, mask_image, depth_intrinsics, color_instrinsics = None):
#         try:
#             # Get the transform from the mask frame to the depth camera frame
#             (trans, rot) = self.tf_listener.lookupTransform(self.__depth_image_frame_id, self.__rgb_image_frame_id, rospy.Time(0))
# 
#             # Create a 3x3 rotation matrix from the transform
#             rotation_matrix = np.array(self.tf_listener.fromTranslationRotation(trans, rot)[:3, :3])
# 
#             # Resize the mask to match the resolution of the depth image
#             resized_mask = cv2.resize(mask_image, (depth_image.shape[1], depth_image.shape[0]))
# 
#             # Convert mask to 3 channels for compatibility with depth image
#             resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
# 
#             # Transform the mask to the depth camera frame
#             # Debug:
#             # print("Transformation matrix: {}".format(rotation_matrix))
#             transformed_mask = cv2.warpPerspective(resized_mask, rotation_matrix, (depth_image.shape[1], depth_image.shape[0]))
# 
#             # Apply the reprojected mask to the depth image
#             # masked_depth_image = depth_image * reprojected_mask[:, :, 2]
# 
#             # return masked_depth_image7
#             return 
# 
#         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
#             rospy.logerr("Error during transformation: %s", str(e))
#             return depth_image
        



    def apply_mask_2(self, depth_image, mask_image, depth_intrinsics, rgb_intrinsics):
        try:
            # Get the transform from the mask frame to the depth camera frame
            (trans, rot) = self.tf_listener.lookupTransform(self.__depth_image_frame_id, self.__rgb_image_frame_id, rospy.Time(0))
            
            # Create a 3x3 rotation matrix from the transform
            rotation_matrix = np.array(self.tf_listener.fromTranslationRotation(trans, rot)[:3, :3])
            trans = np.array(trans)


            print("Translation: {}".format(trans.shape))
            print("o translation: {}".format((np.array([0.0, 0.0, 0.0])).shape))
            print("Rotation: {}".format(rotation_matrix))
            print("No rotation: {}".format((np.eye(3))))

            print("RGB Intrinsics: {}".format(rgb_intrinsics))
            print("Depth Intrinsics: {}".format(depth_intrinsics))


            # Stereo rectify the cameras
            # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(rgb_intrinsics, self.rgb_distortion, depth_intrinsics, self.depth_distortion, (640, 480), np.eye(3), np.zeros(3), alpha=-1)
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(rgb_intrinsics, self.rgb_distortion, depth_intrinsics, self.depth_distortion, (640, 480), rotation_matrix, trans, alpha=1)
            
            print("ROI 1: {}".format(roi1))
            print("ROI 2: {}".format(roi2))

            # Undistort and rectify the mask image
            # undistorted_mask = cv2.remap(mask_image, R1, None, cv2.INTER_LINEAR)
            undistorted_mask = cv2.warpPerspective(mask_image, R1, (640, 480))

            
            # reprojected_mask = cv2.cvtColor(reprojected_mask, cv2.COLOR_GRAY2BGR)

            undistorted_mask[np.where(undistorted_mask == 255)] = 1

            # Apply the reprojected mask to the depth image
            masked_depth_image = np.multiply(depth_image, undistorted_mask)

            return masked_depth_image

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Error during transformation: %s", str(e))
            return depth_image
        





    def run(self):
        # Start ROS Node
        rospy.spin()





if __name__ == "__main__":
    node = DepthMaskingNode()
    node.run()
