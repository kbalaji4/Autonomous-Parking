#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import sys

scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

print("printing paths in main(): ")
print(sys.path)

def filter_points(points_array, max_range=15.0, min_height=-1.7, max_height=-0.5):
    """Filter points based on range and height"""
    # Calculate distances from origin
    distances = np.sqrt(points_array[:,0]**2 + points_array[:,1]**2)
    
    # Create mask for points within range and height limits
    mask = (distances < max_range) & \
           (points_array[:,2] > min_height) & \
           (points_array[:,2] < max_height)
    
    return points_array[mask]

def downsample_points(points_array, voxel_size=0.1):
    """Voxel grid downsampling"""
    # Calculate voxel indices for each point
    voxel_indices = np.floor(points_array / voxel_size).astype(int)
    
    # Create dictionary to store points in each voxel
    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        idx_tuple = tuple(idx)
        if idx_tuple in voxel_dict:
            voxel_dict[idx_tuple].append(i)
        else:
            voxel_dict[idx_tuple] = [i]
    
    # Take centroid of each voxel
    downsampled_points = []
    for indices in voxel_dict.values():
        downsampled_points.append(np.mean(points_array[indices], axis=0))
    
    return np.array(downsampled_points)

def pointcloud_callback(msg):
    try:
        # PointCloud2 to numpy array (x,y,z)
        points = []
        for p in pc2.read_points(msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])
        if not points:
            return
        points = np.array(points)
        rospy.loginfo(f"Original points: {points.shape}")

        filtered_points = filter_points(points)
        rospy.loginfo(f"Filtered points: {filtered_points.shape}")

        downsampled_points = downsample_points(filtered_points)
        rospy.loginfo(f"Downsampled points: {downsampled_points.shape}")

        if len(downsampled_points) == 0:
            rospy.loginfo("no points after downsampling")
            return
        
        scaled_points = downsampled_points
        scaler = StandardScaler()
        scaled_points = scaler.fit_transform(downsampled_points)
        
        # dbscan for unsupervised clustering
        clustering = DBSCAN(eps=0.05, min_samples=6, n_jobs=-1).fit(scaled_points)
        
        labels = clustering.labels_
        unique_labels = set(labels)
        
        marker_array = MarkerArray()
        marker_id = 0
        rospy.loginfo(f"Unique labels: {len(unique_labels)}")
        for k in unique_labels:
            if k == -1:
                # noise
                continue
            class_member_mask = (labels == k)
            cluster = downsampled_points[class_member_mask]

            # if len(cluster) < 5:
            #     # skip small clusters
            #     continue 

            # get centroid
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

            # cluster dims for our markers
            cluster_std = np.std(cluster, axis=0)
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            # marker.scale.x = max(0.3, 2*cluster_std[0])
            # marker.scale.y = max(0.3, 2*cluster_std[1])
            marker.scale.z = 0.5

            marker.color.a = 0.7
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)
            marker_id += 1

        # Publish the obstacles markers
        marker_pub.publish(marker_array)
    except Exception as e:
        rospy.logerr(f"Error processing point cloud in pointcloud_callback: {e}")

def clear_markers():
    """publish empty marker array"""
    marker = Marker()
    # marker.header.frame_id = "map"  # or whatever frame you're using
    marker.header.stamp = rospy.Time.now()
    marker.ns = "obstacles"
    marker.action = Marker.DELETEALL
    marker.id = 0
    
    marker_array = MarkerArray()
    marker_array.markers.append(marker)
    marker_pub.publish(marker_array)
    rospy.sleep(0.1) # wait for publish 


if __name__ == '__main__':
    rospy.init_node('lidar_obstacle_detector')
    lidar_sub = rospy.Subscriber('/ouster/points', PointCloud2, pointcloud_callback, queue_size=1)
    marker_pub = rospy.Publisher('/lidar_obstacles', MarkerArray, queue_size=1)

    clear_markers() # could also just toggle the checkbox in rviz 

    rospy.loginfo("Lidar obstacle detector node started")
    rospy.spin()