#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from sklearn.cluster import DBSCAN
import os
import sys

scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

print("printing paths in main(): ")
print(sys.path)

def pointcloud_callback(msg):
    # PointCloud2 to numpy array (x,y,z)
    points = []
    for p in pc2.read_points(msg, skip_nans=True):
        points.append([p[0], p[1], p[2]])
    if not points:
        return
    points = np.array(points)
    
    # dbscan for unsupervised clustering
    clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
    
    labels = clustering.labels_
    unique_labels = set(labels)
    
    marker_array = MarkerArray()
    marker_id = 0
    for k in unique_labels:
        if k == -1:
            # noise
            continue
        class_member_mask = (labels == k)
        cluster = points[class_member_mask]
        # Compute centroid
        centroid = np.mean(cluster, axis=0)
        
        # Create a marker for this obstacle
        marker = Marker()
        marker.header = msg.header
        marker.ns = "obstacles"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = centroid[0]
        marker.pose.position.y = centroid[1]
        marker.pose.position.z = centroid[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker_array.markers.append(marker)
        marker_id += 1

    # Publish the obstacles markers
    marker_pub.publish(marker_array)

if __name__ == '__main__':
    rospy.init_node('lidar_obstacle_detector')
    lidar_sub = rospy.Subscriber('/ouster/points', PointCloud2, pointcloud_callback, queue_size=10)
    marker_pub = rospy.Publisher('/lidar_obstacles', MarkerArray, queue_size=10)
    rospy.loginfo("Lidar obstacle detector node started")
    rospy.spin()