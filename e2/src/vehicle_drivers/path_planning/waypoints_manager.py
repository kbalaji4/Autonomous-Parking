#!/usr/bin/env python3

# parking_waypoints = [(x, y, yaw)]
# this one calls hybrid_astar_node.py each time we switch to a new waypoint we want to go to

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from msg import Goal

class ParkingWaypointManager:
    def __init__(self):
        rospy.init_node('parking_waypoint_manager')
        
        # List of parking waypoints (x, y, yaw in radians)
        self.parking_spots = [
            # Just one test waypoint for now
            (-25.0, -2.0, 0.0)  # Adjust these coordinates based on your map
        ]
        """
        float64 x
        float64 y
        float64 yaw
        """
        
        self.current_spot_index = 0
        self.current_spot_occupied = False
        self.path_planning_in_progress = False
        
        # Publisher for goal waypoints
        self.goal_pub = rospy.Publisher('/parking_goal', Goal, queue_size=1)
        
        # Subscribe to path planning completion status
        # rospy.Subscriber('/path_planning_status', Bool, self.planning_status_callback)
        
        # Timer to check and publish waypoints
        rospy.Timer(rospy.Duration(1.0), self.check_waypoints)
        
        rospy.loginfo("Parking waypoint manager initialized")

    # def planning_status_callback(self, msg):
    #     """Callback for path planning completion status"""
    #     self.path_planning_in_progress = not msg.data
    #     if not self.path_planning_in_progress:
    #         rospy.loginfo("Path planning completed for current waypoint")

    def publish_next_waypoint(self):
        """Publish the next available parking waypoint"""
        if self.current_spot_index < len(self.parking_spots):
            x, y, yaw = self.parking_spots[self.current_spot_index]
            
            goal_msg = ParkingGoal()
            goal_msg.x = x
            goal_msg.y = y
            goal_msg.yaw = yaw
            
            self.goal_pub.publish(goal_msg)
            rospy.loginfo(f"Published parking waypoint {self.current_spot_index}: ({x}, {y}, {yaw})")
            self.path_planning_in_progress = True
            self.current_spot_index += 1
            return True
        return False

    def check_waypoints(self, event=None):
        """Timer callback to check and publish waypoints"""
        if not self.path_planning_in_progress:
            if not self.publish_next_waypoint():
                rospy.loginfo("No more parking waypoints available")

if __name__ == '__main__':
    try:
        manager = ParkingWaypointManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


