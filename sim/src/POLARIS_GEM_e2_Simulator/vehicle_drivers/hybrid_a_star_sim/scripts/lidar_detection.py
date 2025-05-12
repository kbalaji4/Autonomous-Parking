#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import tf
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sensor_msgs
import os
import sys


scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

print("printing paths in main(): ")
print(sys.path)

class ConeMapper:
    def __init__(self):
        rospy.init_node('cone_mapper')
        # world frame for output poses
        # self.world_frame = rospy.get_param('~world_frame','base_link')

        # Subscribers & Publishers
        self.sub = rospy.Subscriber('/ouster/points', PointCloud2, self.cb, queue_size=1)

        self.filtered_cloud_pub = rospy.Publisher('/filtered_points_sim', PointCloud2, queue_size=1)
        self.filtered_intense_cloud_pub = rospy.Publisher('/filtered_intense_points_sim', PointCloud2, queue_size=1)
        self.pub_markers = rospy.Publisher('/cone_world_positions_sim', PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('/lidar_obstacles_sim', MarkerArray, queue_size=1)

        self.cone_xy_pub = rospy.Publisher('/cone_xy', Float32MultiArray, queue_size=1)

        # # TF2 listener to transform cloud into world_frame
        # self.tfbuf = tf2_ros.Buffer()
        # tf2_ros.TransformListener(self.tfbuf)

        # # TF listener for manual cloud→world (fallback)
        # self.tf = tf.TransformListener()

    def filter_points(self, points_array, max_range=15.0, min_height=-1.5, max_height=-1.0): # gets cone stripes
        """Filter points based on range and height"""
        # Calculate distances from origin
        distances = np.sqrt(points_array[:,0]**2 + points_array[:,1]**2)
        
        # Create mask for points within range and height limits
        mask = (distances < max_range) & \
            (points_array[:,2] > min_height) & \
            (points_array[:,2] < max_height)
        
        return points_array[mask]

    def cb(self, msg: PointCloud2):

        # 2) Convert to numpy Nx3
        pts = np.array([[p[0],p[1],p[2],p[3]] for p in pc2.read_points(msg, skip_nans=True)])
        if pts.shape[0] < 50:
            return
        
        just_filtered_pts = self.filter_points(pts)
        filtered_cloud = pc2.create_cloud_xyz32(
            header=msg.header,
            points=just_filtered_pts[:, :3]  # Only use x,y,z coordinates
        )
        self.filtered_cloud_pub.publish(filtered_cloud)
        
        # print("points.shape: ", pts.shape)
        # print("4th col max min mean: ", np.max(pts[:,3]), np.min(pts[:,3]), np.mean(pts[:,3]))
        high_intensity_pts = pts[pts[:,3] > 5000.0] # only strong reflections
        

        # filter points
        high_intensity_pts = self.filter_points(high_intensity_pts)
        # print("high_intensity points shape: ", high_intensity_pts.shape)

        filtered_intense_cloud = pc2.create_cloud_xyz32(
            header=msg.header,
            points=high_intensity_pts[:, :3]  # Only use x,y,z coordinates
        )
        
        # Publish filtered cloud
        self.filtered_intense_cloud_pub.publish(filtered_intense_cloud)

        # 3) Make Open3D pointcloud

        pts = high_intensity_pts[:,:3] # xyz no intensity

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # # 4) Voxel downsample
        # pcd = pcd.voxel_down_sample(voxel_size=0.05)

        # # 5) Remove ground plane via RANSAC
        # plane_model, inliers = pcd.segment_plane(
        #     distance_threshold=0.02,
        #     ransac_n=3,
        #     num_iterations=1000
        # )
        # pcd_without_ground = pcd.select_by_index(inliers, invert=True)

        # 6) Height & range crop in camera frame: if needed,
        #    you could re-filter by z or xy here

        # 7) DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=0.3, min_points=3, print_progress=False))
        # print("labels: ", labels)
        unique_labels = set(labels) - {-1}
        print("unique labels: ", len(labels), len(unique_labels))

        marker_array = MarkerArray()
        marker_id = 0
        # 8) Publish centroids
        for k in unique_labels:
            class_member_mask = (labels == k)
            cluster = np.asarray(pcd.points)[class_member_mask]

            # if len(cluster) < 3:
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
            rospy.loginfo(f"Centroid: {centroid}, Marker ID: {marker_id}")
            # publish centroid 
            msg - Float32MultiArray()
            msg.data = [centroid[0], centroid[1]]
            self.cone_xy_pub.publish(msg)


        # Publish the obstacles markers
        self.marker_pub.publish(marker_array)
        
        # for lid in unique_labels:
        #     idx = np.where(labels==lid)[0]
        #     cluster = np.asarray(pcd.points)[idx]
        #     # if len(cluster)<2: 
        #     #     continue
        #     centroid = cluster.mean(axis=0)
        #     ps = PoseStamped()
        #     ps.header.stamp = msg.header.stamp
        #     ps.header.frame_id = self.world_frame
        #     ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = centroid
        #     ps.pose.orientation.w = 1.0
        #     self.pub_markers.publish(ps)

    def run(self):
        rospy.spin()    

if __name__ == '__main__':
    cone_mapper = ConeMapper()
    cone_mapper.run()
# def cloud_cb(msg: PointCloud2):
#     try:
#         # 1. lookup transform from your world frame (e.g. "map")  
#         #    from the sensor frame (msg.header.frame_id, e.g. "os_sensor")
#         rospy.loginfo(f"source frame: msg.header.frame_id: {msg.header.frame_id}") # this is os.sensor fs
#         trans = tf_buffer.lookup_transform(
#             world_frame,                # target frame
#             msg.header.frame_id,        # source frame
#             msg.header.stamp,           # at the time the cloud was captured
#             rospy.Duration(0.5)         # wait up to 0.5s for the transform
#         )
#         # 2. transform the entire PointCloud2 into world_frame
#         cloud_world = tf2_sensor_msgs.do_transform_cloud(msg, trans)
#         # 3. re-publish
#         pub_world.publish(cloud_world)

#     except (tf2_ros.LookupException,
#             tf2_ros.ExtrapolationException,
#             tf2_ros.ConnectivityException) as e:
#         rospy.logwarn(f"TF lookup failed: {e}")

# def filter_points(points_array, max_range=15.0, min_height=-1.7, max_height=-0.5):
#     """Filter points based on range and height"""
#     # Calculate distances from origin
#     distances = np.sqrt(points_array[:,0]**2 + points_array[:,1]**2)
    
#     # Create mask for points within range and height limits
#     mask = (distances < max_range) & \
#            (points_array[:,2] > min_height) & \
#            (points_array[:,2] < max_height)
    
#     return points_array[mask]

# def downsample_points(points_array, voxel_size=0.1):
#     """Voxel grid downsampling"""
#     # Calculate voxel indices for each point
#     voxel_indices = np.floor(points_array / voxel_size).astype(int)
    
#     # Create dictionary to store points in each voxel
#     voxel_dict = {}
#     for i, idx in enumerate(voxel_indices):
#         idx_tuple = tuple(idx)
#         if idx_tuple in voxel_dict:
#             voxel_dict[idx_tuple].append(i)
#         else:
#             voxel_dict[idx_tuple] = [i]
    
#     # Take centroid of each voxel
#     downsampled_points = []
#     for indices in voxel_dict.values():
#         downsampled_points.append(np.mean(points_array[indices], axis=0))
    
#     return np.array(downsampled_points)

# def pointcloud_callback(msg):
#     try:
#         # PointCloud2 to numpy array (x,y,z)
#         points = []
#         for p in pc2.read_points(msg, skip_nans=True):
#             points.append([p[0], p[1], p[2]])
#         if not points:
#             return
#         points = np.array(points)
#         # rospy.loginfo(f"Original points: {points.shape}")

#         filtered_points = filter_points(points)
#         # rospy.loginfo(f"Filtered points: {filtered_points.shape}")

#         downsampled_points = downsample_points(filtered_points)
#         # rospy.loginfo(f"Downsampled points: {downsampled_points.shape}")

#         if len(downsampled_points) == 0:
#             rospy.loginfo("no points after downsampling")
#             return
        
#         scaled_points = downsampled_points
#         scaler = StandardScaler()
#         scaled_points = scaler.fit_transform(downsampled_points)
        
#         # dbscan for unsupervised clustering
#         clustering = DBSCAN(eps=0.4, min_samples=4, n_jobs=-1).fit(scaled_points)
        
#         labels = clustering.labels_
#         unique_labels = set(labels)
        
#         marker_array = MarkerArray()
#         marker_id = 0
#         # rospy.loginfo(f"Unique labels: {len(unique_labels)}")
#         for k in unique_labels:
#             if k == -1:
#                 # noise
#                 continue
#             class_member_mask = (labels == k)
#             cluster = downsampled_points[class_member_mask]

#             if len(cluster) < 3:
#                 # skip small clusters
#                 continue 

#             # get centroid
#             centroid = np.mean(cluster, axis=0)
            
#             # Create a marker for this obstacle
#             marker = Marker()
#             marker.header = msg.header
#             marker.ns = "obstacles"
#             marker.id = marker_id
#             marker.type = Marker.SPHERE
#             marker.action = Marker.ADD
#             marker.pose.position.x = centroid[0]
#             marker.pose.position.y = centroid[1]
#             marker.pose.position.z = centroid[2]
#             marker.pose.orientation.w = 1.0

#             # cluster dims for our markers
#             cluster_std = np.std(cluster, axis=0)
#             marker.scale.x = 0.5
#             marker.scale.y = 0.5
#             # marker.scale.x = max(0.3, 2*cluster_std[0])
#             # marker.scale.y = max(0.3, 2*cluster_std[1])
#             marker.scale.z = 0.5

#             marker.color.a = 0.7
#             marker.color.r = 1.0
#             marker.color.g = 0.0
#             marker.color.b = 0.0

#             marker_array.markers.append(marker)
#             marker_id += 1

#         # Publish the obstacles markers
#         marker_pub.publish(marker_array)
#     except Exception as e:
#         rospy.logerr(f"Error processing point cloud in pointcloud_callback: {e}")

# def clear_markers():
#     """publish empty marker array"""
#     marker = Marker()
#     # marker.header.frame_id = "map"  # or whatever frame you're using
#     marker.header.stamp = rospy.Time.now()
#     marker.ns = "obstacles"
#     marker.action = Marker.DELETEALL
#     marker.id = 0
    
#     marker_array = MarkerArray()
#     marker_array.markers.append(marker)
#     marker_pub.publish(marker_array)
#     rospy.sleep(0.1) # wait for publish 


# if __name__ == '__main__':
#     rospy.init_node('lidar_obstacle_detector')
#     lidar_sub = rospy.Subscriber('/ouster/points', PointCloud2, cloud_cb, queue_size=1)


#     marker_pub = rospy.Publisher('/lidar_obstacles', MarkerArray, queue_size=1)
#     pub_world = rospy.Publisher("/ouster/points_world", PointCloud2, queue_size=1) # for visualizing 

#     # which frame your GNSS/INS is publishing as the “map” (or world) frame:
#     world_frame = rospy.get_param("~world_frame", "base_link")
#     print(f"world frame: {world_frame}")

#     # setup TF2
#     tf_buffer   = tf2_ros.Buffer()
#     tf_listener = tf2_ros.TransformListener(tf_buffer)

#     # clear_markers() # could also just toggle the checkbox in rviz 

#     rospy.loginfo("Lidar obstacle detector node started")
#     rospy.spin()