#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import NavSatFix
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from scipy.interpolate import splprep, splev
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from dpp.env.environment import Environment
from dpp.env.car import SimpleCar
from dpp.methods.hybrid_astar import HybridAstar
from dpp.utils.utils import transform
from dpp.utils.coordinate_transform import CoordinateTransform

class HybridAstarNode:
    def __init__(self):
        rospy.init_node('hybrid_astar_node')
        
        # Parameters
        self.grid_size = rospy.get_param('~grid_size', 80.0)  # Size of grid in meters
        self.cell_size = rospy.get_param('~cell_size', 0.5)  # Size of each cell in meters
        self.center_lat = rospy.get_param('~center_lat', 40.0928174)  # Center latitude
        self.center_lon = rospy.get_param('~center_lon', -88.2356714)  # Center longitude
        self.smoothing = rospy.get_param('~smoothing', 0.3)  # Path smoothing parameter
        self.frame_id = rospy.get_param('~frame_id', 'map')  # Frame ID for transformations
        
        # Initialize coordinate transformer
        self.coord_transform = CoordinateTransform(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            grid_size=self.grid_size
        )
        
        # Initialize TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Initialize environment and car
        self.env = Environment()
        self.car = SimpleCar(self.env)
        
        # Initialize hybrid A* planner with modified parameters for smoother paths
        self.planner = HybridAstar(self.car, self.env.grid, reverse=True)
        
        # Modify weights for smoother paths
        self.planner.w1 = 0.95  # weight for astar heuristic
        self.planner.w2 = 0.05  # weight for simple heuristic
        self.planner.w3 = 0.50  # increased weight for steering angle change
        self.planner.w4 = 0.30  # increased weight for turning
        self.planner.w5 = 2.00  # weight for reversing
        
        # Publishers and Subscribers
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1)
        self.smoothed_path_pub = rospy.Publisher('/smoothed_path', Path, queue_size=1)
        self.gps_sub = rospy.Subscriber('/gps/fix', NavSatFix, self.gps_callback)
        self.goal_sub = rospy.Subscriber('/goal_pose', PoseStamped, self.goal_callback)
        self.grid_goal_sub = rospy.Subscriber('/grid_goal', PoseStamped, self.grid_goal_callback)
        
        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.setup_plot()
        
        # State variables
        self.current_pos = None
        self.goal_pos = None
        self.path = None
        self.smoothed_path = None
        
        rospy.loginfo("Hybrid A* Node initialized successfully")
        
    def setup_plot(self):
        """Setup the matplotlib plot for visualization"""
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Plot obstacles
        for ob in self.env.obs:
            self.ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))
            
        self._path, = self.ax.plot([], [], color='lime', linewidth=1)
        self._smoothed_path, = self.ax.plot([], [], color='blue', linewidth=1, alpha=0.5)
        self._carl = PatchCollection([])
        self.ax.add_collection(self._carl)
        self._car = PatchCollection([])
        self.ax.add_collection(self._car)
        
    def gps_callback(self, msg):
        """Callback for GPS position updates"""
        try:
            # Convert GPS to local grid coordinates
            x, y = self.coord_transform.gps_to_grid_coordinates(msg.latitude, msg.longitude)
            
            if not self.coord_transform.validate_grid_point(x, y):
                rospy.logwarn("GPS position outside grid boundaries")
                return
                
            # Assuming heading from IMU or other source
            heading = 0.0  # Replace with actual heading
            self.current_pos = [x, y, heading]
            
            if self.goal_pos is not None:
                self.plan_path()
        except Exception as e:
            rospy.logerr(f"Error in GPS callback: {str(e)}")
            
    def goal_callback(self, msg):
        """Callback for GPS goal position updates"""
        try:
            # Transform goal pose to map frame if needed
            if msg.header.frame_id != self.frame_id:
                transform = self.tf_buffer.lookup_transform(
                    self.frame_id,
                    msg.header.frame_id,
                    rospy.Time(0)
                )
                msg = tf2_geometry_msgs.do_transform_pose(msg, transform)
            
            # Convert GPS coordinates to grid coordinates
            x, y = self.coord_transform.gps_to_grid_coordinates(
                msg.pose.position.y,  # latitude
                msg.pose.position.x   # longitude
            )
            
            if not self.coord_transform.validate_grid_point(x, y):
                rospy.logwarn("Goal position outside grid boundaries")
                return
            
            # Convert quaternion to yaw
            q = msg.pose.orientation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            self.goal_pos = [x, y, yaw]
            
            if self.current_pos is not None:
                self.plan_path()
        except Exception as e:
            rospy.logerr(f"Error in goal callback: {str(e)}")
            
    def grid_goal_callback(self, msg):
        """Callback for grid goal position updates"""
        try:
            x = msg.pose.position.x
            y = msg.pose.position.y
            
            if not self.coord_transform.validate_grid_point(x, y):
                rospy.logwarn("Goal position outside grid boundaries")
                return
            
            # Convert quaternion to yaw
            q = msg.pose.orientation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            self.goal_pos = [x, y, yaw]
            
            if self.current_pos is not None:
                self.plan_path()
        except Exception as e:
            rospy.logerr(f"Error in grid goal callback: {str(e)}")
            
    def smooth_path(self, path):
        """Smooth the path using B-spline interpolation"""
        try:
            # Extract x, y coordinates
            x = [state.pos[0] for state in path]
            y = [state.pos[1] for state in path]
            
            # Create B-spline representation
            tck, u = splprep([x, y], s=self.smoothing)
            
            # Generate more points for smoother curve
            u_new = np.linspace(0, 1, len(path) * 2)
            x_new, y_new = splev(u_new, tck)
            
            # Calculate yaw angles for new points
            dx = np.gradient(x_new)
            dy = np.gradient(y_new)
            yaw_new = np.arctan2(dy, dx)
            
            # Create new path with smoothed points
            smoothed_path = []
            for i in range(len(x_new)):
                pos = [x_new[i], y_new[i], yaw_new[i]]
                state = path[0].__class__(pos, path[0].model)  # Create new state with same class
                smoothed_path.append(state)
            
            return smoothed_path
        except Exception as e:
            rospy.logerr(f"Error in path smoothing: {str(e)}")
            return path
            
    def plan_path(self):
        """Plan path using hybrid A*"""
        if self.current_pos is None or self.goal_pos is None:
            return
            
        try:
            # Update car start and goal positions
            self.car.start_pos = self.current_pos
            self.car.end_pos = self.goal_pos
            
            # Plan path
            path, closed = self.planner.search_path(heu=1, extra=True)
            
            if path is not None:
                self.path = path
                # Smooth the path
                self.smoothed_path = self.smooth_path(path)
                self.publish_paths()
                self.visualize_path()
            else:
                rospy.logwarn("No valid path found")
        except Exception as e:
            rospy.logerr(f"Error in path planning: {str(e)}")
            
    def publish_paths(self):
        """Publish both original and smoothed paths as ROS Path messages"""
        try:
            # Publish original path
            path_msg = Path()
            path_msg.header.frame_id = self.frame_id
            path_msg.header.stamp = rospy.Time.now()
            
            for state in self.path:
                pose = PoseStamped()
                # Convert grid coordinates to GPS coordinates
                lat, lon = self.coord_transform.grid_to_gps_coordinates(
                    state.pos[0], state.pos[1]
                )
                pose.pose.position.x = lon
                pose.pose.position.y = lat
                # Convert yaw to quaternion
                q = quaternion_from_euler(0, 0, state.pos[2])
                pose.pose.orientation = Quaternion(*q)
                path_msg.poses.append(pose)
                
            self.path_pub.publish(path_msg)
            
            # Publish smoothed path
            smoothed_path_msg = Path()
            smoothed_path_msg.header.frame_id = self.frame_id
            smoothed_path_msg.header.stamp = rospy.Time.now()
            
            for state in self.smoothed_path:
                pose = PoseStamped()
                # Convert grid coordinates to GPS coordinates
                lat, lon = self.coord_transform.grid_to_gps_coordinates(
                    state.pos[0], state.pos[1]
                )
                pose.pose.position.x = lon
                pose.pose.position.y = lat
                # Convert yaw to quaternion
                q = quaternion_from_euler(0, 0, state.pos[2])
                pose.pose.orientation = Quaternion(*q)
                smoothed_path_msg.poses.append(pose)
                
            self.smoothed_path_pub.publish(smoothed_path_msg)
        except Exception as e:
            rospy.logerr(f"Error in publishing paths: {str(e)}")
        
    def visualize_path(self):
        """Visualize the planned path"""
        if self.path is None or self.smoothed_path is None:
            return
            
        try:
            xl, yl = [], []
            carl = []
            for state in self.path:
                xl.append(state.pos[0])
                yl.append(state.pos[1])
                carl.append(state.model[0])
                
            # Plot original path
            self._path.set_data(xl, yl)
            
            # Plot smoothed path
            x_smooth = [state.pos[0] for state in self.smoothed_path]
            y_smooth = [state.pos[1] for state in self.smoothed_path]
            self._smoothed_path.set_data(x_smooth, y_smooth)
            
            sub_carl = carl[::4]
            self._carl.set_paths(sub_carl)
            self._carl.set_color('m')
            self._carl.set_alpha(0.1)
            
            edgecolor = ['k']*5 + ['r']
            facecolor = ['y'] + ['k']*4 + ['r']
            self._car.set_paths(self.path[-1].model)
            self._car.set_edgecolor(edgecolor)
            self._car.set_facecolor(facecolor)
            self._car.set_zorder(3)
            
            plt.draw()
            plt.pause(0.001)
        except Exception as e:
            rospy.logerr(f"Error in path visualization: {str(e)}")
        
    def run(self):
        """Main loop"""
        try:
            plt.ion()  # Enable interactive mode
            rospy.spin()
        except Exception as e:
            rospy.logerr(f"Error in main loop: {str(e)}")
        finally:
            plt.ioff()

if __name__ == '__main__':
    try:
        node = HybridAstarNode()
        node.run()
    except rospy.ROSInterruptException:
        pass 