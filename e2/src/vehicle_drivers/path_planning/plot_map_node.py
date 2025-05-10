#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from threading import Lock

from sensor_msgs.msg import NavSatFix, PointCloud2
from septentrio_gnss_driver.msg import INSNavGeod
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import alvinxy.alvinxy as axy
import sensor_msgs.point_cloud2 as pc2
from vision_msgs.msg import Detection2DArray

import sys
import os

scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

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
        
        # Cone detection data
        self.cone_positions = []
        self.cone_lock = Lock()
        
        # Origin coordinates
        self.olat = 40.0928563
        self.olon = -88.2359994
        
        # Setup subscribers
        rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, self.gnss_callback)
        rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_callback)
        rospy.Subscriber("/waypoints", Path, self.path_callback)
        rospy.Subscriber("/os1_cloud_node/points", PointCloud2, self.lidar_callback)
        rospy.Subscriber("/detection/objects", Detection2DArray, self.detection_callback)
        
        # Setup plot
        self.setup_plot()
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=100, blit=True,
            cache_frame_data=False, save_count=100
        )
        plt.show()
        
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
        
        # Initialize obstacle patches
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
        
        plt.title("GEM e2 Live Position, Path, and Cone Detection")
    
    def update_obstacles(self):
        """Update obstacle visualization"""
        try:
            # Remove old obstacle patches
            for patch in self.obstacle_patches:
                patch.remove()
            self.obstacle_patches.clear()
            
            # Add new obstacle patches
            for ob in self.map.obs:
                rect = Rectangle((ob[0], ob[1]), ob[2], ob[3], fc='gray', ec='k')
                self.ax.add_patch(rect)
                self.obstacle_patches.append(rect)
            
            # Force a redraw
            self.fig.canvas.draw_idle()
            rospy.loginfo(f"Updated obstacles. Total obstacles: {len(self.map.obs)}")
        except Exception as e:
            rospy.logerr(f"Error updating obstacles: {str(e)}")
    
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
    
    def ins_callback(self, msg):
        """Callback for INS heading updates"""
        with self.state_lock:
            self.heading = round(msg.heading, 3)
    
    def path_callback(self, msg):
        """Callback for path updates"""
        with self.path_lock:
            # Clear current path data
            self.path_points_x = []
            self.path_points_y = []
            self.path_points_heading = []
            
            # Clear the path line from the plot
            if hasattr(self, 'path_line'):
                self.path_line.set_data([], [])
            
            # Add new path points
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
            
            # Force a redraw of the plot
            self.fig.canvas.draw_idle()
    
    def wps_to_local_xy(self, lon_wp, lat_wp):
        """Convert GNSS coordinates to local coordinates"""
        x, y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
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
    
    def lidar_callback(self, msg):
        """Process LiDAR data to detect orange cones"""
        try:
            # Convert point cloud to numpy array
            points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points.append([point[0], point[1], point[2], point[3]])
            points = np.array(points)
            
            # Filter points based on intensity (orange cones typically have high intensity)
            high_intensity_points = points[points[:, 3] > 0.8]  # Adjust threshold as needed
            
            if len(high_intensity_points) > 0:
                # Convert LiDAR points to map coordinates
                with self.state_lock:
                    if self.lon is not None and self.lat is not None:
                        vehicle_x, vehicle_y = self.wps_to_local_xy(self.lon, self.lat)
                        vehicle_yaw = self.heading_to_yaw(self.heading)
                        
                        # Transform points to map frame and cluster them
                        map_points = []
                        for point in high_intensity_points:
                            # Transform point from LiDAR frame to vehicle frame
                            x_lidar = point[0]
                            y_lidar = point[1]
                            
                            # Transform to map frame
                            x_map = vehicle_x + x_lidar * np.cos(vehicle_yaw) - y_lidar * np.sin(vehicle_yaw)
                            y_map = vehicle_y + x_lidar * np.sin(vehicle_yaw) + y_lidar * np.cos(vehicle_yaw)
                            
                            # Shift coordinates relative to map origin
                            x_shifted = x_map - self.map.grid_top_left[0]
                            y_shifted = self.map.ly - (self.map.grid_top_left[1] - y_map)
                            
                            map_points.append((x_shifted, y_shifted))
                        
                        # Cluster points that are within 0.5m of each other
                        clusters = []
                        for point in map_points:
                            # Check if point belongs to an existing cluster
                            added_to_cluster = False
                            for cluster in clusters:
                                center = np.mean(cluster, axis=0)
                                if ((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2) ** 0.5 < 0.5:
                                    cluster.append(point)
                                    added_to_cluster = True
                                    break
                            
                            # If point doesn't belong to any cluster, create a new one
                            if not added_to_cluster:
                                clusters.append([point])
                        
                        # Add cones to map using cluster centers
                        cones_added = False
                        for cluster in clusters:
                            center = np.mean(cluster, axis=0)
                            if self.map.add_cone(center[0], center[1]):
                                cones_added = True
                                rospy.loginfo(f"Added cone at position: ({center[0]:.2f}, {center[1]:.2f})")
                        
                        # Always update obstacles after processing LiDAR data
                        if cones_added:
                            rospy.loginfo("New cones detected, updating obstacles")
                            self.update_obstacles()
                        else:
                            rospy.loginfo("No new cones detected in this frame")
        except Exception as e:
            rospy.logerr(f"Error in lidar_callback: {str(e)}")

    def detection_callback(self, msg):
        """Process object detection results for cones"""
        # This callback will be used when you have a trained object detection model
        # that can detect cones in camera images
        pass

    def update_plot(self, frame):
        """Update the plot with current vehicle position, path, and cones"""
        try:
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
            
            # Ensure obstacles are up to date
            if len(self.obstacle_patches) != len(self.map.obs):
                self.update_obstacles()
            
            return self.vehicle_marker, self.heading_line, self.path_line, self.car_model, *self.obstacle_patches
        except Exception as e:
            rospy.logerr(f"Error in update_plot: {str(e)}")
            return []

def main():
    try:
        plotter = MapPlotter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main() 