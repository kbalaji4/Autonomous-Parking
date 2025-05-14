#!/usr/bin/env python3

#================================================================
# File name: gem_gnss_pp_tracker_pid.py                                                                  
# Description: gnss waypoints tracker using pid and pure pursuit                                                                
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 08/02/2021                                                                
# Date last modified: 03/14/2025                                                
# Version: 1.0                                                                   
# Usage: rosrun gem_gnss gem_gnss_pp_tracker.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from filters import OnlineFilter
from pid_controllers import PID


# ROS Headers
import alvinxy.alvinxy as axy # Import AlvinXY transformation module
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64, Int64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion


class PurePursuit(object):
    
    def __init__(self):
        # Add plotting variables
        self.plot_data = {
            'time': [],
            'ct_error': [],
            'steering_angle': []
        }
        self.start_time = time.time()

        self.last_max_goal_idx = 0  # Avoid reusing past goals
        
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Pure Pursuit Controller Performance Metrics')
        
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

        # Initialize CSV logging
        self.csv_file = open('vehicle_trajectory_latest.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Type', 'Time (s)', 'X (m)', 'Y (m)', 'Yaw (rad)'])  # CSV header

        self.csv_file1 = open('planner_path_data_latest.csv', mode='w', newline='')
        self.csv_writer1 = csv.writer(self.csv_file)
        self.csv_writer1.writerow(['X (m)', 'Y (m)', 'Yaw (rad)'])  # CSV header

        # Extra CSV for detailed metrics
        self.metrics_file = open('controller_metrics_log.csv', mode='w', newline='')
        self.metrics_writer = csv.writer(self.metrics_file)
        self.metrics_writer.writerow([
            'Time (s)', 'X', 'Y', 'Yaw (rad)', 'Speed (m/s)',
            'CT Error (m)', 'Steering Angle (deg)', 'Wheel Angle (deg)', 'Goal Index'
        ])

        self.rate       = rospy.Rate(10)

        self.look_ahead = 4
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.46 # meters

        self.gnss_sub_old   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        # we replaced novatel hardware with septentrio hardware on e2
        self.gnss_sub   = rospy.Subscriber("/septentrio_gnss/navsatfix", NavSatFix, self.gnss_callback)
        self.ins_sub    = rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self.ins_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0
        
        self.path_points_lon_x = []
        self.path_points_lat_y = []
        self.path_points_heading = []
        
        rospy.Subscriber("/waypoints", Path, self.path_callback)
        
        rospy.Subscriber("/detection_world_positions", PoseStamped, self.detection_callback)

        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        self.olat       = 40.0928563
        self.olon       = -88.2359994

        # read waypoints into the system 
        self.goal_reached_threshold = 4 # meters
        self.goal       = 0        
        self.goal_pub = rospy.Publisher('/current_goal_idx', Int64, queue_size=10) 
        #self.read_waypoints() 

        self.desired_speed = 0.9  # m/s, reference speed
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False
        
        

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = True
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 2.0 # radians/second
        
        self.closest_person_depth = np.inf


    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude  # latitude
        self.lon     = inspva_msg.longitude # longitude
        self.heading = inspva_msg.azimuth   # heading in degrees
    
    def ins_callback(self, msg):
        self.heading = round(msg.heading, 6)
    
    def gnss_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)
        
    def detection_callback(self,msg):
        if msg.pose.position.z == -10.0:
            self.closest_person_depth = np.inf
        else:
            self.closest_person_depth = msg.pose.position.z
        
    def path_callback(self, msg):
        # self.path_points_yaw = []
        i = 0
        for pose in msg.poses:
            print(pose)
            x = pose.pose.position.x
            y = pose.pose.position.y
            yaw = pose.pose.position.z
            self.path_points_lon_x.append(x)
            self.path_points_lat_y.append(y)
            self.path_points_heading.append(yaw) 
            self.csv_writer1.writerow([x, y, yaw])
            i += 1


    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr
    
    def stop_car(self):
        """Stop the car by applying brakes and disabling acceleration."""
        self.brake_cmd.f64_cmd = 0.5  # Apply full brake
        #self.accel_cmd.f64_cmd = 0.0  # Disable acceleration
        self.brake_pub.publish(self.brake_cmd)
        #self.accel_pub.publish(self.accel_cmd)
        rospy.loginfo("Car stopped.")

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

    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/hybrid_astar_path_original.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        # x towards East and y towards North
        self.path_points_lon_x   = [float(point[0]) for point in path_points] # longitude
        self.path_points_lat_y   = [float(point[1]) for point in path_points] # latitude
        self.path_points_heading = [float(point[2]) for point in path_points] # heading
        self.wp_size             = len(self.path_points_lon_x)
        self.dist_arr            = np.zeros(self.wp_size)

    def wps_to_local_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return lon_wp_x, lat_wp_y   

    def get_gem_state(self):

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

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

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

    def start_pp(self):
        
        
        while not rospy.is_shutdown():

            if (self.gem_enable == False):

                if(self.pacmod_enable == True):

                    # ---------- enable PACMod ----------
                    # enable forward gear
                    self.gear_cmd.ui16_cmd = 3

                    # enable brake
                    self.brake_cmd.enable  = True
                    self.brake_cmd.clear   = False
                    self.brake_cmd.ignore  = False
                    self.brake_cmd.f64_cmd = 0.0

                    # enable gas 
                    self.accel_cmd.enable  = True
                    self.accel_cmd.clear   = False
                    self.accel_cmd.ignore  = False
                    self.accel_cmd.f64_cmd = 0.0

                    self.gear_pub.publish(self.gear_cmd)
                    print("Foward Engaged!")

                    self.turn_pub.publish(self.turn_cmd)
                    print("Turn Signal Ready!")
                    
                    self.brake_pub.publish(self.brake_cmd)
                    print("Brake Engaged!")

                    self.accel_pub.publish(self.accel_cmd)
                    print("Gas Engaged!")

                    self.gem_enable = True

            if self.closest_person_depth < 10:
                self.stop_car()
                rospy.sleep(3) # flickering detections
                rospy.loginfo("Human Detected Stopping Car")
                continue
            else:
                self.brake_cmd.f64_cmd = 0.0  # Disable Break
                self.brake_pub.publish(self.brake_cmd)
                rospy.loginfo("Car Starting.")
                        
            self.path_points_x = np.array(self.path_points_lon_x)
            self.path_points_y = np.array(self.path_points_lat_y)

            curr_x, curr_y, curr_yaw = self.get_gem_state()

            # Log current position to CSV
            log_time = time.time() - self.start_time
            self.csv_writer.writerow(["actual", round(log_time, 2), curr_x, curr_y, np.degrees(curr_yaw)])

            self.dist_arr = np.zeros(len(self.path_points_x))
            
        
            if self.goal >= 0:#len(self.path_points_x) - 1:
                goal_x = self.path_points_x[-1]
                goal_y = self.path_points_y[-1]
                distance_to_goal = self.dist((curr_x, curr_y), (goal_x, goal_y))

            # Check if waypoint is behind the car
            waypoint_vector = [self.path_points_x[self.goal] - curr_x, self.path_points_y[self.goal] - curr_y]
            vehicle_heading_vector = [np.cos(curr_yaw), np.sin(curr_yaw)]
            angle_to_waypoint = self.find_angle(vehicle_heading_vector, waypoint_vector)
            
            # If waypoint is behind the car (angle > 90 degrees)
            # if abs(angle_to_waypoint) > np.pi / 2:
            #     if self.gear_cmd.ui16_cmd != 1:  # If not already in reverse
            #         rospy.loginfo("Waypoint is behind. Stopping the car to switch to reverse gear.")
            #         self.stop_car()

            #         rospy.sleep(4)
            #         self.brake_cmd.f64_cmd = 0.0  # Disable Break
            #         self.brake_pub.publish(self.brake_cmd)
            #         rospy.loginfo("Switching to reverse gear.")
            #         self.gear_cmd.ui16_cmd = 1  # Reverse gear
            #         self.gear_pub.publish(self.gear_cmd)
            #         # Adjust look-ahead distance for reverse
            #         self.look_ahead = 2.0  # Shorter look-ahead in reverse
            # else:
            #     if self.gear_cmd.ui16_cmd != 3:  # If not already in forward
            #         rospy.loginfo("Waypoint is ahead. Stopping the car to switch to forward gear.")
            #         self.brake_cmd.f64_cmd = 0.2  # Apply full brake
            #         #self.accel_cmd.f64_cmd = 0.0  # Disable acceleration
            #         self.brake_pub.publish(self.brake_cmd)
            #         #self.accel_pub.publish(self.accel_cmd)
            #         # self.stop_car()
            #         rospy.loginfo("Car stopped.")
            #         self.steer_cmd.angular_position = 0.0

            #         self.steer_pub.publish(self.steer_cmd)
            #         rospy.sleep(4)
            #         self.brake_cmd.f64_cmd = 0.0  # Disable Break
            #         self.brake_pub.publish(self.brake_cmd)
            #         rospy.loginfo("Switching to forward gear.")
            #         self.gear_cmd.ui16_cmd = 3  # Forward gear
            #         self.gear_pub.publish(self.gear_cmd)
            #         # Reset look-ahead distance for forward
            #         self.look_ahead = 4.0  # Normal look-ahead in forward

            print(f"goal :  {self.goal}")
            print(f"lookahead : {self.look_ahead}")
            # finding the distance of each way point from the current position
            for i in range(len(self.path_points_x)):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))
            # finding those points which are less than the look ahead distance (will be behind and ahead of the vehicle)
            # self.dist_arr = self.dist_arr[self.goal:]
            goal_arr = np.where( (self.dist_arr < self.look_ahead + 0.3) & (self.dist_arr > self.look_ahead - 0.3))[0]

            # goal_arr = np.where(
            #     (self.dist_arr < self.look_ahead + 0.3) &
            #     (self.dist_arr > self.look_ahead - 0.3) &
            #     (np.arange(len(self.dist_arr)) >= self.last_max_goal_idx)
            # )[0]
            
            print(self.dist_arr)
            # finding the goal point which is the last in the set of points less than the lookahead distance
            for idx in goal_arr:
                v1 = [self.path_points_x[idx]-curr_x , self.path_points_y[idx]-curr_y]
                v2 = [np.cos(curr_yaw), np.sin(curr_yaw)]
                temp_angle = self.find_angle(v1,v2)
                # find correct look-ahead point by using heading information
                # if self.gear_cmd.ui16_cmd == 1:
                #     if abs(temp_angle) > np.pi/2:
                #         self.goal = idx
                        # self.last_max_goal_idx = max(self.last_max_goal_idx, self.goal)
                        # break
                # else:
                if abs(temp_angle) < np.pi/2:
                    self.goal = idx
                    # self.last_max_goal_idx = max(self.last_max_goal_idx, self.goal)
                    break
            print(self.goal)

            self.goal_pub.publish(Int64(self.goal))
            # finding the distance between the goal point and the vehicle
            # true look-ahead distance between a waypoint and current position
            L = self.dist_arr[self.goal]

            # find the curvature and the angle 
            alpha = self.heading_to_yaw(self.path_points_heading[self.goal]) - curr_yaw

            # ----------------- tuning this part as needed -----------------
            k       = 0.41
            angle_i = math.atan((k * 2 * self.wheelbase * math.sin(alpha)) / L) 
            angle   = angle_i*2
            # ----------------- tuning this part as needed -----------------

            f_delta = round(np.clip(angle, -0.61, 0.61), 3)

            f_delta_deg = np.degrees(f_delta)

            # steering_angle in degrees
            steering_angle = self.front2steer(f_delta_deg)

            # Calculate cross-track error
            ct_error = round(np.sin(alpha) * L, 3)
            
            # Update plot data
            current_time = time.time() - self.start_time
            self.plot_data['time'].append(current_time)
            self.plot_data['ct_error'].append(ct_error)
            self.plot_data['steering_angle'].append(f_delta_deg)
            
            # Keep only last 100 points for better performance
            if len(self.plot_data['time']) > 100:
                self.plot_data['time'] = self.plot_data['time'][-100:]
                self.plot_data['ct_error'] = self.plot_data['ct_error'][-100:]
                self.plot_data['steering_angle'] = self.plot_data['steering_angle'][-100:]

            # Print debug info
            if(self.gem_enable == True):
                print("Current index: " + str(self.goal))
                print("Forward velocity: " + str(self.speed))
                print("Crosstrack Error: " + str(ct_error))
                print("Front steering angle: " + str(f_delta_deg) + " degrees")
                print("Steering wheel angle: " + str(steering_angle) + " degrees")
                print("\n")

            

            current_time = rospy.get_time()
            filt_vel     = self.speed_filter.get_data(self.speed)
            output_accel = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)
            # Log detailed metrics to separate CSV
            self.metrics_writer.writerow([
                round(current_time, 2),
                curr_x, curr_y, curr_yaw,
                self.speed,
                ct_error, f_delta_deg, steering_angle, self.goal
            ])
            if output_accel > self.max_accel:
                output_accel = self.max_accel

            if output_accel < 0.3:
                output_accel = 0.3

            if (f_delta_deg <= 30 and f_delta_deg >= -30):
                self.turn_cmd.ui16_cmd = 1
            elif(f_delta_deg > 30):
                self.turn_cmd.ui16_cmd = 2 # turn left
            else:
                self.turn_cmd.ui16_cmd = 0 # turn right

            self.accel_cmd.f64_cmd = output_accel
            # if self.gear_cmd.ui16_cmd == 1: # reverse angle needs reverse
            #     self.steer_cmd.angular_position = -np.radians(steering_angle)
            # else:
            self.steer_cmd.angular_position = np.radians(steering_angle)
            if distance_to_goal < self.goal_reached_threshold and len(self.path_points_heading) > 0:
                print("Stopping the car as goal is reached")
                self.brake_cmd.f64_cmd = 0.5  # Apply half brake
                #self.accel_cmd.f64_cmd = 0.0  # Disable acceleration
                self.brake_pub.publish(self.brake_cmd)
                if np.abs(self.heading - self.path_points_heading[-1]) < 1:
                    print("aligned with goal breaking")
                    break
                #self.accel_pub.publish(self.accel_cmd)
            else:
                self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)
            self.turn_pub.publish(self.turn_cmd)

            self.rate.sleep()

    def __del__(self):
        """Cleanup when the object is destroyed"""
        plt.close('all')
        plt.ioff()  # Turn off interactive mode
        self.csv_file.close()  # Close the log file
        self.metrics_file.close()


def pure_pursuit():

    rospy.init_node('gnss_pp_node', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()

