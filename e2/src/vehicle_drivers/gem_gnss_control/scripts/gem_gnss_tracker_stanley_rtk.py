#!/usr/bin/env python3

#==============================================================================
# File name          : gem_gnss_tracker_stanley_rtk.py                                                                  
# Description        : gnss waypoints tracker using pid and Stanley controller                                                              
# Author             : Hang Cui (hangcui3@illinois.edu)                                       
# Date created       : 08/08/2022                                                                 
# Date last modified : 03/14/2025                                                          
# Version            : 1.0                                                                    
# Usage              : rosrun gem_gnss_control gem_gnss_tracker_stanley_rtk.py                                                                      
# Python version     : 3.8   
# Longitudinal ctrl  : Ji'an Pan (pja96@illinois.edu), Peng Hang (penghan2@illinois.edu)                                                            
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
from std_msgs.msg import String, Bool, Float32, Float64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from sensor_msgs.msg import NavSatFix, Path
from septentrio_gnss_driver.msg import INSNavGeod
from tf.transformations import euler_from_quaternion, quaternion_from_euler

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

        self.rate   = rospy.Rate(30)

        self.olat   = 40.0928232 
        self.olon   = -88.2355788

        self.offset = 1.1 # meters

        # PID for longitudinal control
        self.desired_speed = 1  # m/s
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        # Goal reaching parameters
        self.goal_reached_threshold = 1  # meters
        self.goal_reached = False
        self.min_speed_threshold = 0.2  # m/s

        # Path data
        self.path_points_x = []
        self.path_points_y = []
        self.path_points_heading = []
        self.path_lock = threading.Lock()

        self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        # we replaced novatel hardware with septentrio hardware on e2
        self.gnss_sub   = rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, self.gnss_callback)
        self.ins_sub    = rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_callback)
        self.path_sub   = rospy.Subscriber("/waypoints", Path, self.path_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        self.stanley_pub = rospy.Publisher('/gem/stanley_gnss_cmd', AckermannDrive, queue_size=1)

        self.ackermann_msg                         = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        # Hang 
        self.steer = 0.0 # degrees
        self.steer_sub = rospy.Subscriber("/pacmod/parsed_tx/steer_rpt", SystemRptFloat, self.steer_callback)


    # Get GNSS information
    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude  # latitude
        self.lon     = inspva_msg.longitude # longitude
        self.heading = inspva_msg.azimuth   # heading in degrees

    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)
    
    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)
        


    # Get vehicle speed
    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s


    # Get value of steering wheel
    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.output),1)

    def path_callback(self, msg):
        """Callback for path updates"""
        with self.path_lock:
            # Clear current path data
            self.path_points_x = []
            self.path_points_y = []
            self.path_points_heading = []
            
            # Add new path points
            for pose in msg.poses:
                # Get original coordinates
                x = pose.pose.position.x
                y = pose.pose.position.y
                yaw = pose.pose.position.z
                
                # Convert to local coordinates
                local_x, local_y = self.wps_to_local_xy_stanley(x, y)
                
                self.path_points_x.append(local_x)
                self.path_points_y.append(local_y)
                self.path_points_heading.append(yaw)

    # Conversion of front wheel to steering wheel
    def front2steer(self, f_angle):
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle


    # Conversion of Lon & Lat to X & Y
    def wps_to_local_xy_stanley(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return -lon_wp_x, -lat_wp_y   


    # Conversion of GNSS heading to vehicle heading
    def heading_to_yaw_stanley(self, heading_curr):
        if (heading_curr >= 0 and heading_curr < 90):
            yaw_curr = np.radians(-heading_curr-90)
        else:
            yaw_curr = np.radians(-heading_curr+270)
        return yaw_curr


    # Get vehicle states: x, y, yaw
    def get_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # rct_errorerence point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_local_xy_stanley(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading_to_yaw_stanley(self.heading) 

        # rct_errorerence point is located at the center of front axle
        curr_x = local_x_curr + self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr + self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)


    # Find close yaw in predefined GNSS waypoint list
    def find_close_yaw(self, arr, val):
        diff_arr = np.array( np.abs( np.abs(arr) - np.abs(val) ) )
        idx = np.where(diff_arr < 0.5)
        return idx


    # Conversion to -pi to pi
    def pi_2_pi(self, angle):

        if angle > np.pi:
            return angle - 2.0 * np.pi

        if angle < -np.pi:
            return angle + 2.0 * np.pi

        return angle

    # Computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def stop_car(self):
        """Stop the car smoothly"""
        self.ackermann_msg.acceleration = 0.0
        self.ackermann_msg.steering_angle = 0.0
        self.stanley_pub.publish(self.ackermann_msg)
        rospy.loginfo("Goal reached! Car stopped.")

    def update_plot(self, frame):
        """Update the plot with new data"""
        if not self.plot_data['time']:  # If no data yet
            return self.line1, self.line2
            
        self.line1.set_data(self.plot_data['time'], self.plot_data['ct_error'])
        self.line2.set_data(self.plot_data['time'], self.plot_data['steering_angle'])
        
        # Update y-axis limits with some padding
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
        
        # Update x-axis limits
        if len(self.plot_data['time']) > 0:
            self.ax1.set_xlim(self.plot_data['time'][0], self.plot_data['time'][-1])
            self.ax2.set_xlim(self.plot_data['time'][0], self.plot_data['time'][-1])
        
        # Force redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        return self.line1, self.line2

    # Start Stanley controller
    def start_stanley(self):
        
        while not rospy.is_shutdown():
            with self.path_lock:
                if not self.path_points_x:  # If no path points available
                    self.rate.sleep()
                    continue

                self.path_points_x   = np.array(self.path_points_x)
                self.path_points_y   = np.array(self.path_points_y)
                self.path_points_yaw = np.array(self.path_points_heading)

            # coordinates of reference point (center of frontal axle) in global frame
            curr_x, curr_y, curr_yaw = self.get_gem_state()

            # Check if we've reached the goal
            if not self.goal_reached:
                # Calculate distance to final waypoint
                final_x = self.path_points_x[-1]
                final_y = self.path_points_y[-1]
                dist_to_goal = self.dist((curr_x, curr_y), (final_x, final_y))
                
                if dist_to_goal < self.goal_reached_threshold:
                    self.goal_reached = True
                    rospy.loginfo(f"Goal reached! Distance: {dist_to_goal:.2f}m")
                    self.stop_car()
                    break

            target_idx = self.find_close_yaw(self.path_points_yaw, curr_yaw)

            self.target_path_points_x   = self.path_points_x[target_idx]
            self.target_path_points_y   = self.path_points_y[target_idx]
            self.target_path_points_yaw = self.path_points_yaw[target_idx]

            # find the closest point
            dx = [curr_x - x for x in self.target_path_points_x]
            dy = [curr_y - y for y in self.target_path_points_y]

            # find the index of closest point
            target_point_idx = int(np.argmin(np.hypot(dx, dy)))

            if (target_point_idx != len(self.target_path_points_x) -1):
                target_point_idx = target_point_idx + 1

            vec_target_2_front    = np.array([[dx[target_point_idx]], [dy[target_point_idx]]])
            front_axle_vec_rot_90 = np.array([[np.cos(curr_yaw - np.pi / 2.0)], [np.sin(curr_yaw - np.pi / 2.0)]])

            # crosstrack error
            ct_error = np.dot(vec_target_2_front.T, front_axle_vec_rot_90)
            ct_error = float(np.squeeze(ct_error))

            # heading error
            theta_e = self.pi_2_pi(self.target_path_points_yaw[target_point_idx]-curr_yaw) 
            theta_e_deg = round(np.degrees(theta_e), 1)
            print("Crosstrack Error: " + str(round(ct_error,3)) + ", Heading Error: " + str(theta_e_deg))

            # Update plot data
            current_time = time.time() - self.start_time
            self.plot_data['time'].append(current_time)
            self.plot_data['ct_error'].append(ct_error)
            self.plot_data['steering_angle'].append(theta_e_deg)
            
            # Keep only last 100 points for better performance
            if len(self.plot_data['time']) > 100:
                self.plot_data['time'] = self.plot_data['time'][-100:]
                self.plot_data['ct_error'] = self.plot_data['ct_error'][-100:]
                self.plot_data['steering_angle'] = self.plot_data['steering_angle'][-100:]

            # --------------------------- Longitudinal control using PD controller ---------------------------
            filt_vel = np.squeeze(self.speed_filter.get_data(self.speed))
            
            # Adjust desired speed based on distance to goal
            dist_to_goal = self.dist((curr_x, curr_y), (self.path_points_x[-1], self.path_points_y[-1]))
            if dist_to_goal < 2.0:  # Start slowing down when within 2 meters
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
                throttle_percent = (a_expected+2.3501) / 7.3454

            if throttle_percent > self.max_accel:
                throttle_percent = self.max_accel
            elif throttle_percent < 0.3:
                throttle_percent = 0.37

            # -------------------------------------- Stanley controller --------------------------------------
            f_delta        = round(theta_e + np.arctan2(ct_error*0.4, filt_vel), 3)
            f_delta        = round(np.clip(f_delta, -0.61, 0.61), 3)
            f_delta_deg    = np.degrees(f_delta)
            steering_angle = self.front2steer(f_delta_deg)

            if (filt_vel < self.min_speed_threshold):
                self.ackermann_msg.acceleration   = throttle_percent
                self.ackermann_msg.steering_angle = 0
            else:
                self.ackermann_msg.acceleration   = throttle_percent
                self.ackermann_msg.steering_angle = round(steering_angle,1)

            self.stanley_pub.publish(self.ackermann_msg)
            self.rate.sleep()

    def __del__(self):
        """Cleanup when the object is destroyed"""
        plt.close('all')
        plt.ioff()  # Turn off interactive mode


def stanley_run():

    rospy.init_node('gnss_stanley_node', anonymous=True)
    stanley = Stanley()

    try:
        stanley.start_stanley()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    stanley_run()