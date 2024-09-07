#! /usr/bin/python3.8

import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

# Assuming you have already imported necessary modules and defined convert_depth_image function

def main():
    rospy.init_node('target_depth_distance')



    # Create a publisher for the cropped depth image
    cropped_depth_image_pub = rospy.Publisher('/target/cropped_depth_image', Image, queue_size=10)
    contact_done_pub = rospy.Publisher('target/depth', Float32, queue_size=10)
    bridge = CvBridge()

    def depth_image_callback(ros_image):
        try:
            depth_image = np.frombuffer(ros_image.data, dtype=np.uint16).reshape((ros_image.height, ros_image.width, -1))
        except ValueError:
            rospy.logerr("Invalid depth image format")
            return
        

        rospy.loginfo_once('[depth-dist] start sending target/depth ..')
        
        depth_array = np.array(depth_image, dtype=np.float32)
        crop_depth_array = depth_array[100:280, 140:500]
        mean_depth = np.mean(crop_depth_array)

        cropped_depth_image_msg = bridge.cv2_to_imgmsg(crop_depth_array, encoding="passthrough")
        cropped_depth_image_msg.header.frame_id = 'd400_color_optical_frame'
        cropped_depth_image_pub.publish(cropped_depth_image_msg)
        
        contact_done_msg = Float32()
        contact_done_msg.data = 0.001*mean_depth 
        contact_done_pub.publish(contact_done_msg)

        # rospy.loginfo("Mean crop depth: %f", mean_depth)
        rospy.loginfo_once("Image height: %f", ros_image.height)
        rospy.loginfo_once("Image width: %f", ros_image.width)


    rospy.Subscriber('/d400/aligned_depth_to_color/image_raw', Image, callback=depth_image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
