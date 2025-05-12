#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from threading import Lock
from sensor_msgs.msg import NavSatFix, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from septentrio_gnss_driver.msg import INSNavGeod
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import tf
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sensor_msgs

import sys
import os

scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from alvin import ll2xy, xy2ll
from dpp.env.map import Map
from dpp.env.car import SimpleCar
from dpp.env.environment import Environment


class MapPlotter:
    def __init__(self):
        rospy.init_node('map_plotter_node')        
        
        # Initialize map and environment
        self.map = Map()
        self.map.add_walls()
        self.env = Environment(self.map.obs, lx=self.map.lx, ly=self.map.ly)
        
        # Vehicle state
        self.lat = None
        self.lon = None
        self.heading = None
        self.state_lock = Lock()
        
        # Path data
        self.path_points_x = []
        self.path_points_y = []
        self.path_points_heading = []
        self.path_lock = Lock()
        
        # Origin coordinates
        self.olat = 40.0928563
        self.olon = -88.2359994
        
        # Setup subscribers
        rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, self.gnss_callback)
        rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_callback)
        rospy.Subscriber("/waypoints", Path, self.path_callback)
        
        self.filtered_cloud_pub = rospy.Publisher('/filtered_points_sim', PointCloud2, queue_size=1)
        self.filtered_intense_cloud_pub = rospy.Publisher('/filtered_intense_points_sim', PointCloud2, queue_size=1)
        self.pub_markers = rospy.Publisher('/cone_world_positions_sim', PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('/lidar_obstacles_sim', MarkerArray, queue_size=1)

        # Subscribers & Publishers
        self.sub = rospy.Subscriber('/ouster/points', PointCloud2, self.lidar_callback, queue_size=1)

        
        # Setup plot
        self.setup_plot()
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=100, blit=True,
            cache_frame_data=False, save_count=100
        )
        plt.show()

        
    
    def filter_points(self, points_array, max_range=15.0, min_height=-1.5, max_height=-1.0): # gets cone stripes
        """Filter points based on range and height"""
        # Calculate distances from origin
        distances = np.sqrt(points_array[:,0]**2 + points_array[:,1]**2)
        
        # Create mask for points within range and height limits
        mask = (distances < max_range) & \
            (points_array[:,2] > min_height) & \
            (points_array[:,2] < max_height)
        
        return points_array[mask]

    def lidar_callback(self, msg: PointCloud2):

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
        labels = np.array(pcd.cluster_dbscan(eps=0.3, min_points=3, print_progress=False))
        # print("labels: ", labels)
        unique_labels = set(labels) - {-1}
        # print("unique labels: ", len(labels), len(unique_labels))

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
            marker.scale.z = 0.5

            marker.color.a = 0.7
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)
            marker_id += 1
            rospy.loginfo(f"Centroid: {centroid}, Marker ID: {marker_id}")
            
            # get state
            x, y, yaw = self.get_vehicle_state()

            cone_x, cone_y = centroid[0] + x, centroid[1] + y

            # print("successfully added cone_x, cone_y: ", cone_x, cone_y)

            if self.map.add_cone(cone_x, cone_y):
                
                self.update_obstacles()

        # vizualization: obstacle markers
        self.marker_pub.publish(marker_array)


    def setup_plot(self):
        """Initialize the plot with map and static elements"""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Set plot limits based on map dimensions
        self.ax.set_xlim(0, self.map.lx)
        self.ax.set_ylim(0, self.map.ly)
        self.ax.set_aspect("equal")
        
        # Add grid lines
        self.ax.set_xticks(np.arange(0, self.map.lx + 1, 10))
        self.ax.set_yticks(np.arange(0, self.map.ly + 1, 10))
        self.ax.grid(True)
        
        # # Plot obstacles
        # for ob in self.map.obs:
        #     self.ax.add_patch(Rectangle((ob[0], ob[1]), ob[2], ob[3], fc='gray', ec='k'))
            
        self.obstacle_patches = []
        self.update_obstacles()
        
        # Initialize vehicle position marker
        self.vehicle_marker = self.ax.plot([], [], 'ro', markersize=10)[0]
        
        # Initialize path line
        self.path_line = self.ax.plot([], [], 'b-', linewidth=2)[0]
        
        # Initialize vehicle heading line
        self.heading_line = self.ax.plot([], [], 'r-', linewidth=2)[0]
        
        # Initialize car model
        self.car_model = None
        
        plt.title("GEM e2 Live Position and Path")
    
    def create_car_model(self, x, y, yaw):
        """Create car model using SimpleCar class"""
        # Create a SimpleCar instance with current position
        car = SimpleCar(self.env, [x, y, yaw], [x, y, yaw])
        
        # Update car parameters to match GEM e2 specs
        car.l = 1.75  # Wheelbase: 69 in = 1.75m
        car.carl = 2.62  # Length: 103 in = 2.62m
        car.carw = 1.41  # Width: 55.5 in = 1.41m
        car.max_phi = 0.5  # Maximum steering angle
        
        # Get car state with model
        car_state = car.get_car_state([x, y, yaw])
        
        # Create collection with car model
        car_collection = PatchCollection(car_state.model, match_original=True)
        
        return car_collection
        
    def gnss_callback(self, msg):
        """Callback for GNSS position updates"""
        with self.state_lock:
            self.lat = round(msg.latitude, 6)
            self.lon = round(msg.longitude, 6)
        # print("lat, lon: ", self.lat, self.lon)
    
    def ins_callback(self, msg):
        """Callback for INS heading updates"""
        with self.state_lock:
            self.heading = round(msg.heading, 3)
    
    def path_callback(self, msg):
        """Callback for path updates"""
        with self.path_lock:
            self.path_points_x = []
            self.path_points_y = []
            self.path_points_heading = []
            
            for pose in msg.poses:
                # Get original coordinates
                x = pose.pose.position.x
                y = pose.pose.position.y
                yaw = pose.pose.position.z
                
                # Shift coordinates relative to map origin
                x_shifted = x - self.map.grid_top_left[0]
                y_shifted = self.map.ly - (self.map.grid_top_left[1] - y)
                
                self.path_points_x.append(x_shifted)
                self.path_points_y.append(y_shifted)
                self.path_points_heading.append(yaw)
    
    def wps_to_local_xy(self, lon_wp, lat_wp):
        """Convert GNSS coordinates to local coordinates"""
        x, y = ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        # print("x, y: ", x, y)
        return x, y
    
    def heading_to_yaw(self, heading_curr):
        """Convert heading to yaw angle"""
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr
    
    def get_vehicle_state(self):
        """Get current vehicle state in local coordinates"""
        with self.state_lock:
            if self.lon is None or self.lat is None or self.heading is None:
                return None, None, None
            
            # Convert to local coordinates
            local_x, local_y = self.wps_to_local_xy(self.lon, self.lat)
            
            # Convert heading to yaw
            yaw = self.heading_to_yaw(self.heading)
            
            # Shift coordinates relative to map origin
            x_shifted = local_x - self.map.grid_top_left[0]
            y_shifted = self.map.ly - (self.map.grid_top_left[1] - local_y)
            
            return x_shifted, y_shifted, yaw
        
    def update_obstacles(self):
        """Update obstacle visualization"""
        # Remove old obstacle patches
        # for patch in self.obstacle_patches:
        #     patch.remove()
        # self.obstacle_patches.clear()
        
        # Add new obstacle patches
        
        for ob in self.map.obs:
            self.ax.add_patch(Rectangle((ob[0], ob[1]), ob[2], ob[3], fc='gray', ec='k'))
        # for ob in self.map.obs:
        #     rect = Rectangle((ob[0], ob[1]), ob[2], ob[3], fc='gray', ec='k')
        #     self.ax.add_patch(rect)
        #     self.obstacle_patches.append(rect)
    
    def update_plot(self, frame):
        """Update the plot with current vehicle position and path"""
        # Get current vehicle state
        x, y, yaw = self.get_vehicle_state()
        
        if x is not None:
            # Update vehicle position
            self.vehicle_marker.set_data([x], [y])
            
            # Update heading line
            heading_length = 2.0  # meters
            self.heading_line.set_data(
                [x, x + heading_length * np.cos(yaw)],
                [y, y + heading_length * np.sin(yaw)]
            )
            
            # Update car model
            if self.car_model is not None:
                self.car_model.remove()
            self.car_model = self.create_car_model(x, y, yaw)
            self.ax.add_collection(self.car_model)
        
        # Update path
        with self.path_lock:
            if self.path_points_x:
                self.path_line.set_data(self.path_points_x, self.path_points_y)
        
        return self.vehicle_marker, self.heading_line, self.path_line, self.car_model

def main():
    try:
        plotter = MapPlotter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main() 
    