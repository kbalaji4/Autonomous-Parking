#!/usr/bin/env python3

import rospy
import math
import numpy as np
from numpy import linalg as la

from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion

from gazebo_msgs.srv import GetModelState

class PurePursuit(object):
    
    def __init__(self):
        self.rate = rospy.Rate(20)
        self.look_ahead = 6    # meters
        self.wheelbase  = 1.75 # meters
        self.goal = 0

        # Waypoint storage
        self.path_points_x = []
        self.path_points_y = []
        self.path_points_yaw = []
        self.dist_arr = np.zeros(1)

        # ROS Subscribers and Publishers
        rospy.Subscriber("/waypoints", Path, self.path_callback)
        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

        self.ackermann_msg = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

    def path_callback(self, msg):
        self.path_points_x = []
        self.path_points_y = []
        self.path_points_yaw = []

        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            quat = pose.pose.orientation
            _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

            self.path_points_x.append(x)
            self.path_points_y.append(y)
            self.path_points_yaw.append(yaw)

        self.dist_arr = np.zeros(len(self.path_points_x))
        rospy.loginfo(f"✅ Received {len(self.path_points_x)} waypoints.")

    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)

    def get_gem_pose(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            service_response = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            model_state = service_response(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service call failed: " + str(exc))
            return 0.0, 0.0, 0.0

        x = model_state.pose.position.x
        y = model_state.pose.position.y

        orientation_q = model_state.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        return round(x, 4), round(y, 4), round(yaw, 4)

    def start_pp(self):
        while not rospy.is_shutdown():

            if len(self.path_points_x) < 2:
                rospy.logwarn("⏳ Waiting for waypoints...")
                self.rate.sleep()
                continue

            curr_x, curr_y, curr_yaw = self.get_gem_pose()

            self.path_points_x = np.array(self.path_points_x)
            self.path_points_y = np.array(self.path_points_y)

            for i in range(len(self.path_points_x)):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))

            goal_arr = np.where((self.dist_arr < self.look_ahead + 0.3) & (self.dist_arr > self.look_ahead - 0.3))[0]

            found = False
            for idx in goal_arr:
                v1 = [self.path_points_x[idx]-curr_x , self.path_points_y[idx]-curr_y]
                v2 = [np.cos(curr_yaw), np.sin(curr_yaw)]
                temp_angle = self.find_angle(v1, v2)
                if abs(temp_angle) < np.pi/2:
                    self.goal = idx
                    found = True
                    break

            if not found:
                rospy.logwarn("⚠️ No valid goal point ahead.")
                self.rate.sleep()
                continue

            L = self.dist_arr[self.goal]

            gvcx = self.path_points_x[self.goal] - curr_x
            gvcy = self.path_points_y[self.goal] - curr_y
            goal_x_veh_coord = gvcx*np.cos(curr_yaw) + gvcy*np.sin(curr_yaw)
            goal_y_veh_coord = gvcy*np.cos(curr_yaw) - gvcx*np.sin(curr_yaw)

            alpha = self.path_points_yaw[self.goal] - curr_yaw
            k = 0.285
            angle_i = math.atan((2 * k * self.wheelbase * math.sin(alpha)) / L) 
            angle = angle_i * 2
            angle = round(np.clip(angle, -0.61, 0.61), 3)

            ct_error = round(np.sin(alpha) * L, 3)

            rospy.loginfo(f"[PP] Crosstrack Error: {ct_error:.3f} | Steering: {math.degrees(angle):.1f}° | Goal: {self.goal}")

            self.ackermann_msg.speed = 2.8
            self.ackermann_msg.steering_angle = angle
            self.ackermann_pub.publish(self.ackermann_msg)

            self.rate.sleep()


def pure_pursuit():
    rospy.init_node('pure_pursuit_sim_node', anonymous=True)
    pp = PurePursuit()
    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    pure_pursuit()
