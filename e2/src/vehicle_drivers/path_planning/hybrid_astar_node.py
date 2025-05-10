#!/usr/bin/env python3

import rospy
import alvinxy.alvinxy as axy

import os
import sys
import csv
import numpy as np
import math
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import time
import csv
from threading import Lock

from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import Int64

scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from dpp.env.grid import Grid
from dpp.env.car import SimpleCar
from dpp.env.map import Map
from dpp.env.environment import Environment
from dpp.test_cases.cases import TestCase
from dpp.methods.hybrid_astar import HybridAstar
from alvin import ll2xy, xy2ll

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
        self.goal = (None,None,None) # x, y , yaw (radians)
        
        self.lon = None
        self.lat = None
        self.heading = None
        
        self.olat = 40.0928563
        self.olon = -88.2359994
        
        self.obs = []
        self.offset = 0.46
        self.num_points = 50
        
    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)
        
    def ins_callback(self, msg):
        """updates the current vehicle heading"""
        self.heading = round(msg.heading, 3)
        # pls be radians and modded
        
    def update_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_local_xy(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        print(self.heading)
        curr_yaw = self.car_heading_to_planner_yaw(self.heading) # this outputs radians too
        #curr_yaw = self.heading_to_yaw(start_yaw) 
        print("curr yaw: ", np.degrees(curr_yaw), curr_yaw)

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)

        self.state = (round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4))
        
    def car_heading_to_planner_yaw(self, yaw):
        """
        input: yaw is degrees
        yaw is car (0 north, CW)
        convert to planner (0 east, CCW)
        output: RADIANS
        """
        planner_yaw = 0.0
        if yaw <= 90.0:
            planner_yaw = 90 - yaw
        else:
            planner_yaw = 450 - yaw
        
        return np.radians(planner_yaw % 360.0) # none should be negative anyway

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr
        
    def plotting_goal_callback(self,msg):
        """ 
        get goal_idx from each actual position of the vehicle. 
        not to be confused with the other goal callback which tells us
        which waypoint to go to currently (multiple parking spot waypoints)
        """
        # print(f"goal_idx: {msg.data}")
        self.current_goal_idx = msg.data

    def wps_to_local_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return lon_wp_x, lat_wp_y

    def local_xy_to_wps(self, x, y):
        """Convert local x,y coordinates back to GPS coordinates"""
        lat, lon = xy2ll(x, y, self.olat, self.olon)
        return lon, lat
    
    def planner_to_local_coords(self, x, y, map):
        """Convert planner coordinates back to local coordinates"""
        local_x = x + map.grid_top_left[0]
        local_y = map.grid_top_left[1] - (map.ly - y)  # Reverse the y-axis flip
        return local_x, local_y
    
    def publish_path(self, path_points):
        pub = rospy.Publisher('/waypoints', Path, queue_size=1, latch=True)
        rospy.sleep(1.0)

        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for point in path_points:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = point.pos[0]
            pose.pose.position.y = point.pos[1]
            pose.pose.position.z = point.pos[2] # yaw degrees
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w  = quaternion_from_euler(0.0, 0.0, np.radians(point.pos[2]))
            path_msg.poses.append(pose)

        pub.publish(path_msg)
        rospy.loginfo(f"✅ Published {len(path_msg.poses)} waypoints to /waypoints")

    

    def wait_for_pose(self):
        """Wait for the vehicle's GPS and IMU data to be available"""
        while not rospy.is_shutdown() and (self.lon is None or self.lat is None or self.heading is None):
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
    
    def save_path_to_csv(self, path, filename, olat, olon):
        """Save path waypoints to a CSV file with local coordinates and yaw in degrees"""
        os.makedirs('waypoints', exist_ok=True)
        filepath = os.path.join('waypoints', filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for state in path:
                writer.writerow([round(state.pos[0], 3), round(state.pos[1], 3), round(state.pos[2], 3)])


    """
    start_hybrid() runs once per goal
    goal is final goal waypoint, not the current goal in controller look ahead
    """
    def start_hybrid(self):
        # plot which waypoint the current position is following 
        #self.setup_vehicle_tracking() # plotting vehicle trajectory
        
        rospy.init_node("hybrid_astar_node")
        rospy.Subscriber("/current_goal_idx", Int64, hybrid.plotting_goal_callback)
        rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, hybrid.gnss_callback)
        rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, hybrid.ins_callback)
        
        rospy.loginfo("⌛ Waiting for GPS and IMU...")
        self.wait_for_pose()
        rospy.loginfo("✅ Received live GPS and IMU.")
        # Set origin GPS coordinates
        
        self.update_gem_state()
        #self.update_gem_state_test() # No GPS Needed for testing
        map = Map()
        map.add_walls() #if you want parking spots
       
        start_x, start_y, start_yaw = self.state
        goal_lon, goal_lat, goal_yaw = self.goal
        
        goal_yaw = self.car_heading_to_planner_yaw(goal_yaw) # radians
        print(goal_yaw)
        goal_x, goal_y = self.wps_to_local_xy(goal_lon, goal_lat)


        
        start_x_shifted = start_x - map.grid_top_left[0]
        start_y_shifted = map.ly - (map.grid_top_left[1] - start_y)  # Flip y-axis
        goal_x_shifted = goal_x - map.grid_top_left[0]
        goal_y_shifted = map.ly - (map.grid_top_left[1] - goal_y)  # Flip y-axis\\

        print(start_x_shifted, start_y_shifted)
        print(goal_x_shifted, goal_y_shifted)
        
        # Initialize environment and car with shifted coordinates and yaw angles
        env = Environment(map.obs, lx=map.lx, ly=map.ly)  # Set environment size based on coordinates
        start_pos = [start_x_shifted, start_y_shifted, start_yaw]  # Initial yaw in radians
        goal_pos = [goal_x_shifted, goal_y_shifted, goal_yaw]     # Final yaw in radians
        car = SimpleCar(env, start_pos, goal_pos)
        
        # Update car parameters to match GEM e2 specs
        car.l = 1.75  # Wheelbase: 69 in = 1.75m
        car.carl = 2.62  # Length: 103 in = 2.62m
        car.carw = 1.41  # Width: 55.5 in = 1.41m
        car.max_phi = 0.3  # Maximum steering angle
        
        # Adjust grid size based on environment size
        #cell_size = max(0.25, env_size / 200)  # Ensure reasonable number of cells
        grid = Grid(env, cell_size= map.cell_size)
        
        # Initialize hybrid A* planner with modified parameters for smoother paths
        hastar = HybridAstar(car, grid, reverse=True)
        
        # Modify weights to prioritize orientation
        hastar.w1 = 0.8   # weight for astar heuristic
        hastar.w2 = 0.2   # weight for simple heuristic
        hastar.w3 = 0.2  # increased weight for steering angle change
        hastar.w4 = 0.2   # increased weight for turning
        hastar.w5 = 2.0   # weight for reversing
        
        try:
            #self.setup_vehicle_tracking()
            rospy.loginfo("🚀 Planning path from live GPS to local goal...")
            # Plan path
            print("Planning path...")
            #print(f"Environment size: {env_size:.2f}m x {env_size:.2f}m")
            print(f"Cell size: {map.cell_size:.2f}m")
            #print(f"Environment center: x={center_x:.2f}, y={center_y:.2f}")
            print(f"Start position (local): x={start_x:.2f}, y={start_y:.2f}, yaw={start_yaw:.3f}°")
            print(f"Goal position (local): x={goal_x:.2f}, y={goal_y:.2f}, yaw={goal_yaw:.3f}°")
            print(f"Start position (shifted): x={start_x_shifted:.2f}, y={start_y_shifted:.2f}, yaw={start_yaw:.3f}°")
            print(f"Goal position (shifted): x={goal_x_shifted:.2f}, y={goal_y_shifted:.2f}, yaw={goal_yaw:.3f}°")
            path, closed_ = hastar.search_path(heu=1, extra=True)
            # t = time()
            # print('Total time: {}s'.format(round(time()-t, 3)))
            if path:
                for state in path:
                    local_x, local_y = self.planner_to_local_coords(state.pos[0], state.pos[1], map)
                    state.pos[0] = local_x
                    state.pos[1] = local_y
                    # state.pos[0] = state.pos[0] + center_x - env_size/2
                    # state.pos[1] = state.pos[1] + center_y - env_size/2
                    state.pos[2] = np.degrees(state.pos[2])  # Convert yaw to degrees
                    # Normalize yaw to [0, 360)
                    state.pos[2] = state.pos[2] % 360.0
                    state.pos[2] = (90-state.pos[2]) % 360.0
                    
                
                # Downsample path for waypoints (use smaller step for shorter paths)
                step = max(1, len(path) // self.num_points)  
                path = path[::step] + [path[-1]]
                self.publish_path(path)

                # Save both original and smoothed paths to CSV with local coordinates and yaw in degrees
                self.save_path_to_csv(path, 'hybrid_astar_path_original.csv', self.olat, self.olon)
                print(f"Paths saved to waypoints/")
                print(f"Number of waypoints (original): {len(path)}")
                print("Plotting original path...")
                self.plot_path(env, path, closed_, self.olat, self.olon)
            else:
                rospy.logerr("❌ Path planning failed.")
            rospy.spin()
        finally:
            pass
            #self.cleanup_vehicle_tracking()
        
      
        
if __name__ == "__main__":
    hybrid = Hybrid()
    try:
        # lon, lat, yaw (degrees)
        parking_spots = [
            (-88.2353660,40.0928328, 90), # spot facing east
            (-88.235317,40.092751,141.43), # angle parking spot (not supported with walls)
            (-88.235711,40.092788,180)  # Yellow main parking spot facing south  
            ] 
        hybrid.goal = parking_spots[1]
        hybrid.start_hybrid()
    except rospy.ROSInterruptException:
        hybrid.cleanup_vehicle_tracking()
    
        
        