#!/usr/bin/env python3

import rospy
import math
import numpy as np
from sensor_msgs.msg import NavSatFix
# Fix is for e2
from septentrio_gnss_driver.msg import INSNavGeod
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion
import pyproj
import argparse

import sys
import os

# Ensure the `scripts` directory is at the very beginning of the Python path
scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

print("printing paths in main(): ")
print(sys.path)

from astar_utils import hybrid_astar, plot_path
from constants import STARTX, STARTY, STARTYAW, GPS_STARTLON, GPS_STARTLAT 


current_utm = None
current_yaw = None

def gps_callback(msg):
    global current_utm
    lon, lat = msg.longitude, msg.latitude
    utm_proj = pyproj.Proj(proj='utm', zone=16, ellps='WGS84')
    x, y = utm_proj(lon, lat)
    current_utm = (x, y)
    # current_utm = {lon, lat}

# def imu_callback(msg):
#     global current_yaw
#     q = msg.orientation
#     _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
#     current_yaw = (yaw + np.pi) % (2*np.pi) # offset by 180, it's backwards for some reason
#     # print(current_yaw)

def ins_callback(msg):
    global current_yaw
    current_yaw = heading_to_yaw(round(msg.heading, 6))
    # pls be radians and modded

def heading_to_yaw(heading_curr):
    if (heading_curr >= 270 and heading_curr < 360):
        yaw_curr = np.radians(450 - heading_curr)
    else:
        yaw_curr = np.radians(90 - heading_curr)
    return yaw_curr

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

# def publish_path_markers(path_points, offset_x, offset_y):
#     marker_pub = rospy.Publisher('/path_markers', MarkerArray, queue_size=1, latch=True)
#     rospy.sleep(1.0)

#     marker_array = MarkerArray()
#     for i, (x, y, yaw) in enumerate(path_points):
#         marker = Marker()
#         marker.header.frame_id = "map"
#         marker.header.stamp = rospy.Time.now()
#         marker.ns = "path_markers"
#         marker.id = i
#         marker.type = Marker.ARROW
#         marker.action = Marker.ADD
#         marker.pose.position.x = x - offset_x
#         marker.pose.position.y = y - offset_y
#         marker.pose.position.z = 0.0
#         marker.pose.orientation.x = 0.0
#         marker.pose.orientation.y = 0.0
#         marker.pose.orientation.z = math.sin(yaw / 2.0)
#         marker.pose.orientation.w = math.cos(yaw / 2.0)
#         marker.scale.x = 0.5  # Arrow length
#         marker.scale.y = 0.1  # Arrow width
#         marker.scale.z = 0.1  # Arrow height
#         marker.color.a = 1.0  # Alpha
#         marker.color.r = 1.0  # Red
#         marker.color.g = 0.0  # Green
#         marker.color.b = 0.0  # Blue

#         marker_array.markers.append(marker)

#     marker_pub.publish(marker_array)
#     rospy.loginfo(f"✅ Published {len(marker_array.markers)} markers to /path_markers")

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

    rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, gps_callback)
    # rospy.Subscriber("/septentrio_gnss/imu", Imu, imu_callback)
    rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, ins_callback)

    rospy.loginfo("⌛ Waiting for GPS and IMU...")
    wait_for_pose()
    rospy.loginfo("✅ Received live GPS and IMU.")

    # Get UTM start pose from live GPS
    start_x, start_y = current_utm
    start_yaw = current_yaw # we take this from sensor not ground truth, so 180
    # start_yaw = math.radians(180) # fix it to face 180 degrees for now
    start_pose = (start_x, start_y, start_yaw)

    # keep it at 0 so the plots work
    offset_x = 0
    offset_y = 0

    # Convert local Gazebo goal to UTM
    # 270 is irl west 
    # goal_x = args.goal_x + offset_x 
    # goal_y = args.goal_y + offset_y

    # project both, since raw lon/lat is very granular
    lon, lat = args.goal_x, args.goal_y
    goal_proj = pyproj.Proj(proj='utm', zone=16, ellps='WGS84')
    x, y = goal_proj(lon, lat)
    goal_x, goal_y = x, y

    goal_yaw = math.radians(args.goal_yaw)
    goal_pose = (goal_x, goal_y, goal_yaw)

    print(f"start: {start_x, start_y}")
    # print(f"gazebo: {gazebo_start_x, gazebo_start_y}")
    # print(f"offsets: {offset_x, offset_y}")
    print(f'goal: {goal_x, goal_y}')

    rospy.loginfo("🚀 Planning path from live GPS to local goal...")
    path = hybrid_astar(start_pose, goal_pose)

    # rosrun hybrid_a_star_sim hybrid_astar_rs_node.py --goal_x -88.2360875 --goal_y 40.0928 --goal_yaw 180


    if path:
        publish_path(path, offset_x, offset_y)

        # publish_path_markers(path, offset_x, offset_y) # off for now

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
