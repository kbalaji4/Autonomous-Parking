#!/usr/bin/env python3

import rospy
import math
import numpy as np
from nav_msgs.msg import Path
from ackermann_msgs.msg import AckermannDrive
from gazebo_msgs.srv import GetModelState
from tf.transformations import euler_from_quaternion

class PurePursuit:
    def __init__(self):
        rospy.init_node('pure_pursuit_sim_node')
        self.look_ahead = 1.5  # ⬅️ Adjusted for tighter waypoint spacing
        self.wheelbase = 1.75
        self.goal = 0

        self.path_points_x = []
        self.path_points_y = []
        self.path_points_yaw = []
        self.dist_arr = np.zeros(1)

        rospy.Subscriber("/waypoints", Path, self.path_callback)
        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)
        self.rate = rospy.Rate(20)
        self.ackermann_msg = AckermannDrive()

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

    def get_pose(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)('gem', '')
            pos = model_state.pose.position
            ori = model_state.pose.orientation
            _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            return pos.x, pos.y, yaw
        except:
            return 0.0, 0.0, 0.0

    def run(self):
        while not rospy.is_shutdown():
            if len(self.path_points_x) < 2:
                rospy.loginfo("⌛ Waiting for waypoints...")
                self.rate.sleep()
                continue

            cx, cy, cyaw = self.get_pose()

            # Compute distances to all waypoints
            self.dist_arr = np.hypot(np.array(self.path_points_x) - cx,
                                     np.array(self.path_points_y) - cy)

            # Find the first point greater than lookahead distance
            goal_indices = np.where(self.dist_arr > self.look_ahead)[0]

            if len(goal_indices) > 0:
                self.goal = goal_indices[0]
            else:
                # Fallback: choose closest waypoint
                self.goal = np.argmin(self.dist_arr)
                rospy.logwarn("⚠️ No waypoint beyond lookahead — fallback to closest")

            goal_x = self.path_points_x[self.goal]
            goal_y = self.path_points_y[self.goal]
            goal_yaw = self.path_points_yaw[self.goal]

            dx = goal_x - cx
            dy = goal_y - cy
            L = math.hypot(dx, dy)

            # Transform goal to vehicle coordinates
            local_x = dx * math.cos(cyaw) + dy * math.sin(cyaw)
            local_y = dy * math.cos(cyaw) - dx * math.sin(cyaw)

            alpha = math.atan2(local_y, local_x)
            steering_angle = math.atan2(2 * self.wheelbase * math.sin(alpha), L)
            steering_angle = np.clip(steering_angle, -0.61, 0.61)

            # 🔍 Debug
            rospy.loginfo(f"[Drive] Car at ({cx:.2f}, {cy:.2f}) yaw: {math.degrees(cyaw):.1f}°")
            rospy.loginfo(f"[Drive] Waypoint #{self.goal} at dist {L:.2f} → steering {math.degrees(steering_angle):.1f}°")

            # Publish control
            self.ackermann_msg.steering_angle = steering_angle
            self.ackermann_msg.speed = 1.2
            self.ackermann_pub.publish(self.ackermann_msg)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        PurePursuit().run()
    except rospy.ROSInterruptException:
        pass
