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
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from threading import Lock
from std_msgs.msg import Int64
import time
import csv
import pyproj
import argparse

import sys

import os
import csv
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from scipy.interpolate import splprep, splev
import matplotlib.animation as animation
from alvin import ll2xy, xy2ll
from msg import Goal, Waypoint, WaypointArray

from dpp.env.grid import Grid
from dpp.env.car import SimpleCar
from dpp.env.environment import Environment
from dpp.test_cases.cases import TestCase
from dpp.methods.hybrid_astar import HybridAstar

class Hybrid(object):
    def __init__(self):
        """plotting globals"""
        self.vehicle_positions = []
        self.vehicle_positions_lock = Lock()
        self.csv_writer = None
        self.csv_writer1 = None
        self.csv_file = None
        self.csv_file1 = None
        self.current_goal_idx = 0
        """vehicle position tracking
        goal is not the current goal waypoint but the 
        
        """
        self.state = (None,None,None) # x, y , yaw (radians)
        self.goal = (None,None,None)# x, y , yaw (radians)
        
        self.lon = None
        self.lat = None
        self.olat = 40.0928563
        self.olon = -88.2359994
        
        self.obs = []
        self.num_points = 50
        
    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)
        
    def ins_callback(self, msg):
        """updates the current vehicle heading"""
        self.state[2] = round(msg.heading, 3)
        # pls be radians and modded
        
    def update_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_local_xy(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading_to_yaw(self.heading) 

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)

        self.state = (round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4))

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr
    
    def goal_callback(self, msg):
        """Updates the current goal waypoint and triggers path planning."""
        self.goal = msg.data
        rospy.loginfo("📍 New goal received. Starting hybrid path planning...")
        try:
            self.start_hybrid()
        except Exception as e:
            rospy.logerr(f"❌ Error during hybrid path planning: {e}")
        
    def plotting_goal_callback(self,msg):
        """ 
        get goal_idx from each actual position of the vehicle. 
        not to be confused with the other goal callback which tells us
        which waypoint to go to currently (multiple parking spot waypoints)
        """
        print(f"goal_idx: {msg.data}")
        self.current_goal_idx = msg.data

    def wps_to_local_xy(self, lon_wp, lat_wp, olat, olon):
        """Convert GNSS waypoints into local fixed frame represented in x and y"""
        x, y = ll2xy(lat_wp, lon_wp, olat, olon)
        return x, y

    def local_xy_to_wps(self, x, y, olat, olon):
        """Convert local x,y coordinates back to GPS coordinates"""
        lat, lon = xy2ll(x, y, olat, olon)
        return lon, lat
    
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
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w  = quaternion_from_euler(0.0, 0.0, yaw)
            print(euler_from_quaternion([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]))
            print(yaw)
            # pose.pose.orientation.x = 0.0
            # pose.pose.orientation.y = 0.0
            # pose.pose.orientation.z = math.sin(yaw / 2.0)
            # pose.pose.orientation.w = math.cos(yaw / 2.0)
            path_msg.poses.append(pose)
            
        pub.publish(path_msg)
        rospy.loginfo(f"✅ Published {len(path_msg.poses)} waypoints to /waypoints (Gazebo frame)")

    def smooth_path(self, path, smoothing=0.3):
        # Not Currently Used Needs Fixing
        """Smooth the path using B-spline interpolation"""
        # Extract x, y coordinates
        x = [state.pos[0] for state in path]
        y = [state.pos[1] for state in path]
        
        # Extract yaw angles and convert to radians for interpolation
        yaw = [np.radians(state.pos[2]) for state in path]
        
        # Check if we have enough points for smoothing
        if len(path) < 4:
            print("Warning: Path too short for smoothing, returning original path")
            return path
        
        try:
            # Create B-spline representation with increased smoothing
            tck, u = splprep([x, y], s=smoothing, k=3)  # k=3 for cubic spline
            
            # Generate more points for smoother curve
            # Increase the number of points by a factor of 4
            u_new = np.linspace(0, 1, len(path) * 4)
            x_new, y_new = splev(u_new, tck)
            
            # Calculate yaw angles for new points using central differences
            dx = np.gradient(x_new)
            dy = np.gradient(y_new)
            yaw_new = np.arctan2(dy, dx)
            
            # Ensure smooth yaw transition
            # Calculate the target yaw change
            start_yaw = np.radians(path[0].pos[2])
            end_yaw = np.radians(path[-1].pos[2])
            
            # Create a smooth yaw transition
            yaw_progress = np.linspace(0, 1, len(x_new))
            yaw_transition = start_yaw + (end_yaw - start_yaw) * yaw_progress
            
            # Blend between path-based yaw and target yaw
            blend_weight = np.exp(-5 * (yaw_progress - 0.5)**2)  # Gaussian blend
            yaw_new = (1 - blend_weight) * yaw_transition + blend_weight * yaw_new
            
            # Create new path with smoothed points
            smoothed_path = []
            for i in range(len(x_new)):
                pos = [x_new[i], y_new[i], np.degrees(yaw_new[i])]
                state = path[0].__class__(pos, path[0].model)  # Create new state with same class
                smoothed_path.append(state)
            
            # Ensure start and end yaw angles are exactly as specified
            smoothed_path[0].pos[2] = path[0].pos[2]
            smoothed_path[-1].pos[2] = path[-1].pos[2]
            
            # Print statistics about the smoothing
            print(f"Original path points: {len(path)}")
            print(f"Smoothed path points: {len(smoothed_path)}")
            
            return smoothed_path
        except Exception as e:
            print(f"Warning: Path smoothing failed: {str(e)}")
            print("Returning original path")
            return path

    def save_path_to_csv(self, path, filename, olat, olon):
        """Save path waypoints to a CSV file with local coordinates and yaw in degrees"""
        # Create waypoints directory if it doesn't exist
        os.makedirs('waypoints', exist_ok=True)
        
        # Full path to the CSV file
        filepath = os.path.join('waypoints', filename)
        
        # Write path data to CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write waypoints
            for state in path:
                writer.writerow([round(state.pos[0], 3), round(state.pos[1], 3), round(state.pos[2], 3)])

    def wait_for_pose(self):
        """Wait for the vehicle's GPS and IMU data to be available"""
        while not rospy.is_shutdown() and (self.state[0] is None or self.state[2] is None):
            rospy.sleep(0.1)
            
    def save_vehicle_position(self):
        if self.state is not None:
            with self.vehicle_positions_lock:
                position = [time.time(), self.state[0], self.state[1], self.state[2],  self.current_goal_idx]
                self.vehicle_positions.append(position)
                if self.csv_writer:
                    self.csv_writer.writerow(['actual'] + position)
                if self.csv_writer1:
                    self.csv_writer1.writerow(['actual'] + position)

    def setup_vehicle_tracking(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"vehicle_trajectory_{timestamp}.csv"
        filename1 = "vehicle_trajectory_latest.csv"
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['type', 'timestamp', 'x', 'y', 'yaw',  'target_waypoint_idx'])

        self.csv_file1 = open(filename1, 'w', newline='')
        self.csv_writer1 = csv.writer(self.csv_file1)
        self.csv_writer1.writerow(['type', 'timestamp', 'x', 'y', 'yaw',  'target_waypoint_idx'])
        
        # Start position tracking timer
        rospy.Timer(rospy.Duration(0.1), lambda _: self.save_vehicle_position())

    def cleanup_vehicle_tracking(self,):
        if self.csv_file:
            self.csv_file.close()
        if self.csv_file1:
            self.csv_file1.close()

    def plot_path(self, env, path, closed_, olat, olon):
        """Plot the path with animation"""
        # Extract path data
        xl, yl = [], []
        carl = []
        for state in path:
            xl.append(state.pos[0])
            yl.append(state.pos[1])
            carl.append(state.model[0])
        
        # Extract branches for visualization
        branches = []
        bcolors = []
        for node in closed_:
            for b in node.branches:
                branches.append(b[1:])
                bcolors.append('y' if b[0] == 1 else 'b')
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, env.lx)
        ax.set_ylim(0, env.ly)
        ax.set_aspect("equal")
        
        # Add grid lines
        ax.set_xticks(np.arange(0, env.lx + 1, 10))
        ax.set_yticks(np.arange(0, env.ly + 1, 10))
        ax.grid(True)
        
        # Plot obstacles
        for ob in env.obs:
            ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))
        
        # Plot branches
        for b, c in zip(branches, bcolors):
            x = [p[0] for p in b]
            y = [p[1] for p in b]
            ax.plot(x, y, color=c, linewidth=0.5, alpha=0.3)
        
        # Plot path
        _path, = ax.plot([], [], color='lime', linewidth=2)
        _carl = PatchCollection([])
        ax.add_collection(_carl)
        _car = PatchCollection([])
        ax.add_collection(_car)
        
        def animate(i):
            _path.set_data(xl[min(i, len(path)-1):], yl[min(i, len(path)-1):])
            
            sub_carl = carl[:min(i+1, len(path))]
            _carl.set_paths(sub_carl[::4])
            _carl.set_color('m')
            _carl.set_alpha(0.1)
            
            edgecolor = ['k']*5 + ['r']
            facecolor = ['y'] + ['k']*4 + ['r']
            _car.set_paths(path[min(i, len(path)-1)].model)
            _car.set_edgecolor(edgecolor)
            _car.set_facecolor(facecolor)
            _car.set_zorder(3)
            
            return _path, _carl, _car
        
        # Create animation
        frames = len(path) + 1
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
        plt.title("GEM e2 Path Planning Visualization")
        plt.show()


    """
    start_hybrid() runs once per goal
    goal is final goal waypoint, not the current goal in controller look ahead
    """
    def start_hybrid(self):
        # plot which waypoint the current position is following 
        self.setup_vehicle_tracking() # plotting vehicle trajectory
        
        rospy.loginfo("⌛ Waiting for GPS and IMU...")
        self.wait_for_pose()
        rospy.loginfo("✅ Received live GPS and IMU.")
        # Set origin GPS coordinates
        
        self.update_gem_state()
        # Convert yaw angles from degrees to radians
        start_yaw_rad = self.state[2]
        goal_yaw_rad = self.goal[2]  # Already in radians
        
        start_x, start_y = self.state[0], self.state[1]
        goal_x, goal_y = self.goal[0], self.goal[1]
        
        # Calculate environment size and center
        dx = abs(goal_x - start_x)
        dy = abs(goal_y - start_y)
        env_size = max(dx, dy) * 2.0  # Make it twice as large as needed
        env_size = max(env_size, 100.0)  # Ensure minimum size of 100m
        
        # Calculate center point
        center_x = (start_x + goal_x) / 2.0
        center_y = (start_y + goal_y) / 2.0
        
        # Shift coordinates relative to center
        start_x_shifted = start_x - center_x + env_size/2
        start_y_shifted = start_y - center_y + env_size/2
        goal_x_shifted = goal_x - center_x + env_size/2
        goal_y_shifted = goal_y - center_y + env_size/2
        
        # Initialize environment and car with shifted coordinates and yaw angles
        env = Environment(self.obs, lx=env_size, ly=env_size)  # Set environment size based on coordinates
        start_pos = [start_x_shifted, start_y_shifted, start_yaw_rad]  # Initial yaw in radians
        goal_pos = [goal_x_shifted, goal_y_shifted, goal_yaw_rad]     # Final yaw in radians
        car = SimpleCar(env, start_pos, goal_pos)
        
        # Update car parameters to match GEM e2 specs
        car.l = 1.75  # Wheelbase: 69 in = 1.75m
        car.carl = 2.62  # Length: 103 in = 2.62m
        car.carw = 1.41  # Width: 55.5 in = 1.41m
        car.max_phi = 0.5  # Maximum steering angle
        
        # Adjust grid size based on environment size
        cell_size = max(0.25, env_size / 200)  # Ensure reasonable number of cells
        grid = Grid(env, cell_size=cell_size)
        
        # Initialize hybrid A* planner with modified parameters for smoother paths
        hastar = HybridAstar(car, grid, reverse=True)
        
        # Modify weights to prioritize orientation
        hastar.w1 = 0.8   # weight for astar heuristic
        hastar.w2 = 0.2   # weight for simple heuristic
        hastar.w3 = 0.8   # increased weight for steering angle change
        hastar.w4 = 0.6   # increased weight for turning
        hastar.w5 = 2.0   # weight for reversing
        
        try:
            self.setup_vehicle_tracking()
            rospy.loginfo("🚀 Planning path from live GPS to local goal...")
            # Plan path
            print("Planning path...")
            print(f"Environment size: {env_size:.2f}m x {env_size:.2f}m")
            print(f"Cell size: {cell_size:.2f}m")
            print(f"Environment center: x={center_x:.2f}, y={center_y:.2f}")
            print(f"Start position (local): x={start_x:.2f}, y={start_y:.2f}, yaw={start_yaw_deg:.3f}°")
            print(f"Goal position (local): x={goal_x:.2f}, y={goal_y:.2f}, yaw={goal_yaw_deg:.3f}°")
            print(f"Start position (shifted): x={start_x_shifted:.2f}, y={start_y_shifted:.2f}, yaw={start_yaw_deg:.3f}°")
            print(f"Goal position (shifted): x={goal_x_shifted:.2f}, y={goal_y_shifted:.2f}, yaw={goal_yaw_deg:.3f}°")
            t = time()
            path, closed_ = hastar.search_path(heu=1, extra=True)
            print('Total time: {}s'.format(round(time()-t, 3)))

            if path:
                for point in path:
                    point.pos[0] = point.pos[0] + center_x - env_size/2
                    point.pos[1] = point.pos[1] + center_y - env_size/2
                    point.pos[2] = np.degrees(point.pos[2])  # Convert yaw to degrees
                    # Normalize yaw to [0, 360)
                    point.pos[2] = point.pos[2] % 360.0
                    point.pos[2] = (90-point.pos[2]) % 360.0
                
                # Downsample path for waypoints (use smaller step for shorter paths)
                step = max(1, len(path) // self.num_points)  
                path = path[::step] + [path[-1]]
                publish_path(path)

                # Save both original and smoothed paths to CSV with local coordinates and yaw in degrees
                self.save_path_to_csv(path, 'hybrid_astar_path_original.csv', self.olat, self.olon)
                #save_path_to_csv(smoothed_path, 'hybrid_astar_path_smoothed.csv', olat, olon)
                print(f"Paths saved to waypoints/")
                # Print some statistics
                print(f"Number of waypoints (original): {len(path)}")
                # Plot paths
                print("Plotting original path...")
                self.plot_path(env, path, closed_, self.olat, self.olon)
                print("Plotting vehicle trajectory...")
            else:
                rospy.logerr("❌ Path planning failed.")
            rospy.spin()
        finally:
            self.cleanup_vehicle_tracking()
        
      
        
if __name__ == "__main__":
    hybrid = Hybrid()
    try:
        rospy.init_node("hybrid_astar_node")
        rospy.Subscriber("/current_goal_idx", Int64, hybrid.plotting_goal_callback)
        rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, hybrid.gnss_callback)
        rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, hybrid.ins_callback)
        rospy.Subscriber("/goal_topic", GoalMsgType, hybrid.goal_callback)  # Replace GoalMsgType with the actual message type
        rospy.spin()
    except rospy.ROSInterruptException:
        hybrid.cleanup_vehicle_tracking()
    
        
        


