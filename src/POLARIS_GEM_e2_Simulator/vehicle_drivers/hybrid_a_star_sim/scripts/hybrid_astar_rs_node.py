#!/usr/bin/env python3

import rospy
import math
import numpy as np
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
import pyproj
import argparse
from hybrid_astar_rs import hybrid_astar, plot_path

current_utm = None
current_yaw = None

def gps_callback(msg):
    global current_utm
    lon, lat = msg.longitude, msg.latitude
    utm_proj = pyproj.Proj(proj='utm', zone=16, ellps='WGS84')
    x, y = utm_proj(lon, lat)
    current_utm = (x, y)

def imu_callback(msg):
    global current_yaw
    q = msg.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    current_yaw = yaw

def publish_path(path_points, offset_x, offset_y):
    pub = rospy.Publisher('/waypoints', Path, queue_size=1, latch=True)
    rospy.sleep(1.0)

    path_msg = Path()
    path_msg.header.frame_id = "map"
    path_msg.header.stamp = rospy.Time.now()

    for x, y, yaw in path_points:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = x - offset_x
        pose.pose.position.y = y - offset_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        path_msg.poses.append(pose)

    pub.publish(path_msg)
    rospy.loginfo(f"✅ Published {len(path_msg.poses)} waypoints to /waypoints (Gazebo frame)")

def wait_for_pose():
    global current_utm, current_yaw
    while not rospy.is_shutdown() and (current_utm is None or current_yaw is None):
        rospy.sleep(0.1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal_x", type=float, required=True, help="Goal X in Gazebo map frame")
    parser.add_argument("--goal_y", type=float, required=True, help="Goal Y in Gazebo map frame")
    parser.add_argument("--goal_yaw", type=float, required=True, help="Goal yaw in degrees")
    args = parser.parse_args()

    rospy.init_node("hybrid_astar_rs_node")

    rospy.Subscriber("/gps/fix", NavSatFix, gps_callback)
    rospy.Subscriber("/imu", Imu, imu_callback)

    rospy.loginfo("⌛ Waiting for GPS and IMU...")
    wait_for_pose()
    rospy.loginfo("✅ Received live GPS and IMU.")

    # Get UTM start pose from live GPS
    start_x, start_y = current_utm
    start_yaw = current_yaw
    start_pose = (start_x, start_y, start_yaw)

    # GEM starts at Gazebo: x = -1.5, y = -21
    gazebo_start_x = -1.5
    gazebo_start_y = -21.0

    # Offset between GPS UTM and Gazebo's map frame
    offset_x = start_x - gazebo_start_x
    offset_y = start_y - gazebo_start_y

    # Convert local Gazebo goal to UTM
    goal_x = args.goal_x + offset_x
    goal_y = args.goal_y + offset_y
    goal_yaw = math.radians(args.goal_yaw)
    goal_pose = (goal_x, goal_y, goal_yaw)

    rospy.loginfo("🚀 Planning path from live GPS to local goal...")
    path = hybrid_astar(start_pose, goal_pose)

    if path:
        publish_path(path, offset_x, offset_y)

        # Optional debug plot in local frame
        local_path = [(x - offset_x, y - offset_y, yaw) for x, y, yaw in path]
        plot_path(local_path, (start_x - offset_x, start_y - offset_y, start_yaw),
                  (goal_x - offset_x, goal_y - offset_y, goal_yaw))
    else:
        rospy.logerr("❌ Path planning failed.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
