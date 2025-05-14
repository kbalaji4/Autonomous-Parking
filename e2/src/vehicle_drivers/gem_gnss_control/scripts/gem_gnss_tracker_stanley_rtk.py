#!/usr/bin/env python3

#==============================================================================
# File name          : controller_stanley.py                                                                  
# Description        : GNSS waypoints tracker using Stanley controller with reverse and stopping capabilities                                                              
# Author             : Original by Hang Cui, Modified for GEM e2                                       
# Date created       : 08/08/2022                                                                 
# Date last modified : 03/14/2025                                                          
# Version            : 1.0                                                                    
# Usage              : rosrun gem_gnss_control controller_stanley.py                                                                      
# Python version     : 3.8   
#==============================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from filters import OnlineFilter
from pid_controllers import PID

# ROS Headers
import rospy
import alvinxy.alvinxy as axy 
from ackermann_msgs.msg import AckermannDrive

from tf.transformations import euler_from_quaternion, quaternion_from_euler

from std_msgs.msg import String, Bool, Float32, Float64, Int64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

class Stanley(object):
    
    def __init__(self):
        # Add plotting variables
        self.plot_data = {
            'time': [],
            'ct_error': [],
            'steering_angle': []
        }
        self.start_time = time.time()
        
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Stanley Controller Performance Metrics')
        
        # Initialize lines for plotting
        self.line1, = self.ax1.plot([], [], 'b-', label='Cross-track Error')
        self.line2, = self.ax2.plot([], [], 'r-', label='Steering Angle')
        
        # Setup subplots
        self.ax1.set_ylabel('Cross-track Error (m)')
        self.ax1.set_title('Cross-track Error vs Time')
        self.ax1.grid(True)
        self.ax1.legend()
        
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Steering Angle (deg)')
        self.ax2.set_title('Steering Angle vs Time')
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Initialize animation with blit=True for better performance
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100, blit=True)
        plt.ion()  # Enable interactive mode
        plt.show(block=False)

        self.rate = rospy.Rate(30)

        # Origin coordinates
        self.olat = 40.0928232 
        self.olon = -88.2355788

        # Vehicle parameters
        self.offset = 1.1  # meters
        self.wheelbase = 1.75  # meters
        self.min_speed_threshold = 0.2  # m/s

        # Control parameters
        self.desired_speed = 1.0  # m/s
        self.max_accel = 0.48  # % of acceleration
        self.pid_speed = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter = OnlineFilter(1.2, 30, 4)

        # Goal reaching parameters
        self.goal_reached_threshold = 1.0  # meters
        self.goal_reached = False

        # Path data
        self.path_points_lon_x = []
        self.path_points_lat_y = []
        self.path_points_heading = []
        self.path_lock = threading.Lock()

        # Subscribers
        self.gnss_sub = rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, self.gnss_callback)
        self.ins_sub = rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_callback)
        self.path_sub = rospy.Subscriber("/waypoints", Path, self.path_callback)
        self.speed_sub = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.steer_sub = rospy.Subscriber("/pacmod/parsed_tx/steer_rpt", SystemRptFloat, self.steer_callback)
        self.detection_sub = rospy.Subscriber("/detection_world_positions", PoseStamped, self.detection_callback)
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        # Vehicle state
        self.lat = 0.0
        self.lon = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.steer = 0.0
        self.closest_person_depth = np.inf
        self.gem_enable = False
        self.pacmod_enable = False

        # Publishers
        self.stanley_pub = rospy.Publisher('/gem/stanley_gnss_cmd', AckermannDrive, queue_size=1)
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)

        # Initialize commands
        self.ackermann_msg = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration = 0.0
        self.ackermann_msg.jerk = 0.0
        self.ackermann_msg.speed = 0.0 
        self.ackermann_msg.steering_angle = 0.0

        # PACMod commands
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2  # SHIFT_NEUTRAL

        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = True
        self.brake_cmd.clear = True
        self.brake_cmd.ignore = True

        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear = True
        self.accel_cmd.ignore = True

        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1  # None

        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0
        self.steer_cmd.angular_velocity_limit = 2.0

    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)

    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3)

    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.output), 1)

    def detection_callback(self, msg):
        if msg.pose.position.z == -10.0:
            self.closest_person_depth = np.inf
        else:
            self.closest_person_depth = msg.pose.position.z

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def path_callback(self, msg):
        with self.path_lock:
            self.path_points_lon_x = []
            self.path_points_lat_y = []
            self.path_points_heading = []
            
            for pose in msg.poses:
                x = pose.pose.position.x
                y = pose.pose.position.y
                yaw = pose.pose.position.z
                lon, lat = axy.xy2ll(x, y, self.olat, self.olon)
                self.path_points_lon_x.append(lon)
                self.path_points_lat_y.append(lat)
                self.path_points_heading.append(yaw)

    def front2steer(self, f_angle):
        if f_angle > 35:
            f_angle = 35
        if f_angle < -35:
            f_angle = -35
        if f_angle > 0:
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif f_angle < 0:
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle

    def wps_to_local_xy_stanley(self, lon_wp, lat_wp):
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return -lon_wp_x, -lat_wp_y

    def heading_to_yaw_stanley(self, heading_curr):
        if heading_curr >= 0 and heading_curr < 90:
            yaw_curr = np.radians(-heading_curr-90)
        else:
            yaw_curr = np.radians(-heading_curr+270)
        return yaw_curr

    def get_gem_state(self):
        local_x_curr, local_y_curr = self.wps_to_local_xy_stanley(self.lon, self.lat)
        curr_yaw = self.heading_to_yaw_stanley(self.heading)
        curr_x = local_x_curr + self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr + self.offset * np.sin(curr_yaw)
        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    def find_close_yaw(self, arr, val):
        diff_arr = np.array(np.abs(np.abs(arr) - np.abs(val)))
        idx = np.where(diff_arr < 0.5)
        return idx

    def pi_2_pi(self, angle):
        if angle > np.pi:
            return angle - 2.0 * np.pi
        if angle < -np.pi:
            return angle + 2.0 * np.pi
        return angle

    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def stop_car(self):
        self.brake_cmd.f64_cmd = 0.5
        self.brake_pub.publish(self.brake_cmd)
        rospy.loginfo("Car stopped.")

    def update_plot(self, frame):
        if not self.plot_data['time']:
            return self.line1, self.line2
            
        self.line1.set_data(self.plot_data['time'], self.plot_data['ct_error'])
        self.line2.set_data(self.plot_data['time'], self.plot_data['steering_angle'])
        
        if len(self.plot_data['ct_error']) > 0:
            min_ct = min(self.plot_data['ct_error'])
            max_ct = max(self.plot_data['ct_error'])
            padding = (max_ct - min_ct) * 0.1 if max_ct != min_ct else 0.1
            self.ax1.set_ylim(min_ct - padding, max_ct + padding)
            
        if len(self.plot_data['steering_angle']) > 0:
            min_steer = min(self.plot_data['steering_angle'])
            max_steer = max(self.plot_data['steering_angle'])
            padding = (max_steer - min_steer) * 0.1 if max_steer != min_steer else 5
            self.ax2.set_ylim(min_steer - padding, max_steer + padding)
        
        if len(self.plot_data['time']) > 0:
            self.ax1.set_xlim(self.plot_data['time'][0], self.plot_data['time'][-1])
            self.ax2.set_xlim(self.plot_data['time'][0], self.plot_data['time'][-1])
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        return self.line1, self.line2

    def start_stanley(self):
        while not rospy.is_shutdown():
            # Enable PACMod if not already enabled
            if not self.gem_enable and self.pacmod_enable:
                self.gear_cmd.ui16_cmd = 3  # Forward gear
                self.brake_cmd.enable = True
                self.brake_cmd.clear = False
                self.brake_cmd.ignore = False
                self.brake_cmd.f64_cmd = 0.0
                self.accel_cmd.enable = True
                self.accel_cmd.clear = False
                self.accel_cmd.ignore = False
                self.accel_cmd.f64_cmd = 0.0

                self.gear_pub.publish(self.gear_cmd)
                self.turn_pub.publish(self.turn_cmd)
                self.brake_pub.publish(self.brake_cmd)
                self.accel_pub.publish(self.accel_cmd)

                self.gem_enable = True
                rospy.loginfo("Vehicle enabled and ready")

            # Check for obstacles
            if self.closest_person_depth < 10:
                self.stop_car()
                rospy.sleep(3)
                rospy.loginfo("Human detected - stopping car")
                continue
            else:
                self.brake_cmd.f64_cmd = 0.0
                self.brake_pub.publish(self.brake_cmd)

            with self.path_lock:
                if not self.path_points_x:
                    self.rate.sleep()
                    continue

                self.path_points_x = np.array(self.path_points_lon_x)
                self.path_points_y = np.array(self.path_points_lat_y)
                self.path_points_yaw = np.array(self.path_points_heading)

            # Get current vehicle state
            curr_x, curr_y, curr_yaw = self.get_gem_state()

            # Check if goal is reached
            if not self.goal_reached:
                final_x = self.path_points_x[-1]
                final_y = self.path_points_y[-1]
                dist_to_goal = self.dist((curr_x, curr_y), (final_x, final_y))
                
                if dist_to_goal < self.goal_reached_threshold:
                    self.goal_reached = True
                    rospy.loginfo(f"Goal reached! Distance: {dist_to_goal:.2f}m")
                    self.stop_car()
                    break

            # Find target waypoint
            target_idx = self.find_close_yaw(self.path_points_yaw, curr_yaw)
            target_path_points_x = self.path_points_x[target_idx]
            target_path_points_y = self.path_points_y[target_idx]
            target_path_points_yaw = self.path_points_yaw[target_idx]

            # Find closest point
            dx = [curr_x - x for x in target_path_points_x]
            dy = [curr_y - y for y in target_path_points_y]
            target_point_idx = int(np.argmin(np.hypot(dx, dy)))

            if target_point_idx != len(target_path_points_x) - 1:
                target_point_idx = target_point_idx + 1

            # Calculate cross-track error
            vec_target_2_front = np.array([[dx[target_point_idx]], [dy[target_point_idx]]])
            front_axle_vec_rot_90 = np.array([[np.cos(curr_yaw - np.pi / 2.0)], [np.sin(curr_yaw - np.pi / 2.0)]])
            ct_error = float(np.squeeze(np.dot(vec_target_2_front.T, front_axle_vec_rot_90)))

            # Calculate heading error
            theta_e = self.pi_2_pi(target_path_points_yaw[target_point_idx] - curr_yaw)
            theta_e_deg = round(np.degrees(theta_e), 1)

            # Check if we need to reverse
            waypoint_vector = [target_path_points_x[target_point_idx] - curr_x, 
                             target_path_points_y[target_point_idx] - curr_y]
            vehicle_heading_vector = [np.cos(curr_yaw), np.sin(curr_yaw)]
            angle_to_waypoint = self.find_angle(vehicle_heading_vector, waypoint_vector)

            # If waypoint is behind the car (angle > 90 degrees)
            if abs(angle_to_waypoint) > np.pi / 2:
                if self.gear_cmd.ui16_cmd != 1:  # If not already in reverse
                    rospy.loginfo("Waypoint is behind. Stopping the car to switch to reverse gear.")
                    self.stop_car()
                    rospy.sleep(2)
                    self.brake_cmd.f64_cmd = 0.0  # Disable brake
                    self.brake_pub.publish(self.brake_cmd)
                    rospy.loginfo("Switching to reverse gear.")
                    self.gear_cmd.ui16_cmd = 1  # Reverse gear
                    self.gear_pub.publish(self.gear_cmd)
            else:
                if self.gear_cmd.ui16_cmd != 3:  # If not already in forward
                    rospy.loginfo("Waypoint is ahead. Stopping the car to switch to forward gear.")
                    self.stop_car()
                    rospy.sleep(2)
                    self.brake_cmd.f64_cmd = 0.0  # Disable brake
                    self.brake_pub.publish(self.brake_cmd)
                    rospy.loginfo("Switching to forward gear.")
                    self.gear_cmd.ui16_cmd = 3  # Forward gear
                    self.gear_pub.publish(self.gear_cmd)

            # Update plot data
            current_time = time.time() - self.start_time
            self.plot_data['time'].append(current_time)
            self.plot_data['ct_error'].append(ct_error)
            self.plot_data['steering_angle'].append(theta_e_deg)
            
            if len(self.plot_data['time']) > 100:
                self.plot_data['time'] = self.plot_data['time'][-100:]
                self.plot_data['ct_error'] = self.plot_data['ct_error'][-100:]
                self.plot_data['steering_angle'] = self.plot_data['steering_angle'][-100:]

            # Longitudinal control
            filt_vel = np.squeeze(self.speed_filter.get_data(self.speed))
            
            # Adjust speed based on distance to goal
            dist_to_goal = self.dist((curr_x, curr_y), (self.path_points_x[-1], self.path_points_y[-1]))
            if dist_to_goal < 6:
                adjusted_speed = self.desired_speed * (dist_to_goal / 2.0)
                adjusted_speed = max(adjusted_speed, self.min_speed_threshold)
            else:
                adjusted_speed = self.desired_speed

            a_expected = self.pid_speed.get_control(rospy.get_time(), adjusted_speed - filt_vel)

            if a_expected > 0.64:
                throttle_percent = 0.5
            elif a_expected < 0.0:
                throttle_percent = 0.0
            else:
                throttle_percent = (a_expected + 2.3501) / 7.3454

            throttle_percent = np.clip(throttle_percent, 0.3, self.max_accel)

            # Stanley controller with reverse handling
            is_reverse = self.gear_cmd.ui16_cmd == 1
            if is_reverse:
                # In reverse, we need to flip the sign of the cross-track error
                ct_error = -ct_error
                # Also flip the heading error
                theta_e = -theta_e

            f_delta = round(theta_e + np.arctan2(ct_error * 0.4, filt_vel), 3)
            f_delta = round(np.clip(f_delta, -0.61, 0.61), 3)
            f_delta_deg = np.degrees(f_delta)
            steering_angle = self.front2steer(f_delta_deg)

            # Set turn signals
            if f_delta_deg <= 30 and f_delta_deg >= -30:
                self.turn_cmd.ui16_cmd = 1
            elif f_delta_deg > 30:
                self.turn_cmd.ui16_cmd = 2  # turn left
            else:
                self.turn_cmd.ui16_cmd = 0  # turn right

            # Publish commands
            if filt_vel < self.min_speed_threshold:
                self.accel_cmd.f64_cmd = throttle_percent
                self.steer_cmd.angular_position = 0
            else:
                self.accel_cmd.f64_cmd = throttle_percent
                # In reverse, we need to flip the steering angle
                if is_reverse:
                    self.steer_cmd.angular_position = -np.radians(steering_angle)
                else:
                    self.steer_cmd.angular_position = np.radians(steering_angle)

            self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)
            self.turn_pub.publish(self.turn_cmd)

            self.rate.sleep()

    def __del__(self):
        plt.close('all')
        plt.ioff()

def stanley_run():
    rospy.init_node('gnss_stanley_node', anonymous=True)
    stanley = Stanley()

    try:
        stanley.start_stanley()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    stanley_run()
