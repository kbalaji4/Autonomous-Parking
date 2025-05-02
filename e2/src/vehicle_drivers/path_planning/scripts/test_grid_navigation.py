#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from tf.transformations import quaternion_from_euler
import time

def create_pose_stamped(x, y, yaw_deg):
    """Create a PoseStamped message with the given position and yaw"""
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = rospy.Time.now()
    
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0
    
    # Convert yaw to quaternion
    q = quaternion_from_euler(0, 0, np.radians(yaw_deg))
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    
    return pose

def create_gps_msg(lat, lon):
    """Create a NavSatFix message with the given GPS coordinates"""
    msg = NavSatFix()
    msg.header.frame_id = "gps"
    msg.header.stamp = rospy.Time.now()
    
    msg.latitude = lat
    msg.longitude = lon
    msg.altitude = 0.0
    
    return msg

def main():
    rospy.init_node('test_grid_navigation')
    
    # Create publishers
    gps_pub = rospy.Publisher('/gps/fix', NavSatFix, queue_size=1)
    grid_goal_pub = rospy.Publisher('/grid_goal', PoseStamped, queue_size=1)
    
    # Wait for subscribers
    rospy.sleep(1.0)
    
    # Center coordinates
    center_lat = 40.0928174
    center_lon = -88.2356714
    
    # Publish initial GPS position (center of grid)
    gps_msg = create_gps_msg(center_lat, center_lon)
    gps_pub.publish(gps_msg)
    rospy.loginfo("Published initial GPS position at center of grid")
    
    # Wait for GPS message to be processed
    rospy.sleep(2.0)
    
    # Create goal pose at (10,10) with 0 degree yaw
    goal_pose = create_pose_stamped(10.0, 10.0, 0.0)
    grid_goal_pub.publish(goal_pose)
    rospy.loginfo("Published goal position at (10,10)")
    
    # Keep the node running to allow visualization
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Test completed")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass 