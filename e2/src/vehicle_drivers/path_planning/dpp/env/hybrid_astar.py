# import heapq
# import numpy as np
# import rospy

# class HybridAStar:
#     def __init__(self, env, start, goal, max_iter=1000, step_size=0.5, max_steer=0.5):
#         self.env = env
#         self.start = start
#         self.goal = goal
#         self.max_iter = max_iter
#         self.step_size = step_size
#         self.max_steer = max_steer
        
#         # Grid resolution
#         self.grid_res = 0.5  # Reduced from 1.0 for finer planning
        
#         # Cost weights
#         self.w_steer = 0.1  # Reduced from 0.2 to allow more steering
#         self.w_gear = 0.05  # Further reduced to encourage gear changes
#         self.w_obs = 1.0    # Increased from 0.5 to better avoid obstacles
#         self.w_goal = 0.5   # Reduced from 1.0 to focus more on path quality
#         self.w_reverse = 0.1  # New weight for reverse motion
        
#         # Parking-specific parameters
#         self.parking_mode = False
#         self.parking_angle_threshold = np.pi/4  # 45 degrees
#         self.parking_distance_threshold = 3.0   # meters
        
#         # Initialize grid
#         self.grid_size = (int(env.lx/self.grid_res), int(env.ly/self.grid_res))
#         self.grid = np.zeros(self.grid_size)
        
#         # Initialize closed set
#         self.closed = set()
        
#         # Initialize open set
#         self.open = []
#         heapq.heappush(self.open, (0, 0, start))
        
#         # Initialize node dictionary
#         self.nodes = {0: {'parent': None, 'cost': 0, 'steer': 0, 'gear': 1}}
        
#         # Initialize path
#         self.path = None
        
#         # Check if this is a parking maneuver
#         self._check_parking_maneuver()

#     def _check_parking_maneuver(self):
#         """Check if this is a parking maneuver based on start and goal positions"""
#         # Calculate angle between start and goal
#         dx = self.goal[0] - self.start[0]
#         dy = self.goal[1] - self.start[1]
#         angle = np.arctan2(dy, dx)
        
#         # Calculate distance to goal
#         distance = np.sqrt(dx*dx + dy*dy)
        
#         # Check if this looks like a parking maneuver
#         if (abs(angle) > self.parking_angle_threshold and 
#             distance < self.parking_distance_threshold):
#             self.parking_mode = True
#             # Adjust parameters for parking
#             self.step_size = 0.3  # Smaller steps for parking
#             self.grid_res = 0.3   # Finer grid for parking
#             self.w_steer = 0.05   # Allow more steering for parking
#             self.w_gear = 0.02    # Further reduced to encourage gear changes
#             self.w_obs = 1.5      # Higher obstacle weight for parking
#             self.w_reverse = 0.05  # Reduced reverse penalty for parking
#             rospy.loginfo("Parking mode activated")

#     def _get_neighbors(self, node):
#         """Get neighboring nodes with improved parking behavior"""
#         neighbors = []
        
#         # Define steering angles with more options for parking
#         if self.parking_mode:
#             steer_angles = np.linspace(-self.max_steer, self.max_steer, 11)  # More steering options
#         else:
#             steer_angles = np.linspace(-self.max_steer, self.max_steer, 7)
        
#         # Try both forward and reverse
#         for gear in [-1, 1]:
#             for steer in steer_angles:
#                 # In parking mode, don't skip any combinations
#                 if not self.parking_mode and abs(steer) > self.max_steer * 0.7 and gear == 1:
#                     continue
                
#                 # Calculate next state
#                 next_state = self._move(node, steer, gear)
                
#                 # Check if valid
#                 if self._is_valid(next_state):
#                     # Calculate cost with parking-specific adjustments
#                     cost = self._calculate_cost(node, next_state, steer, gear)
#                     neighbors.append((next_state, cost, steer, gear))
        
#         return neighbors

#     def _calculate_cost(self, current, next_state, steer, gear):
#         """Calculate cost with improved parking behavior"""
#         # Base cost
#         cost = 0
        
#         # Steering cost (reduced for parking)
#         cost += self.w_steer * abs(steer)
        
#         # Gear change cost (reduced for parking)
#         if self.nodes[current]['gear'] != gear:
#             cost += self.w_gear
        
#         # Reverse motion cost (reduced for parking)
#         if gear == -1:
#             cost += self.w_reverse
        
#         # Obstacle cost (increased for parking)
#         if self._is_collision(next_state):
#             cost += self.w_obs * 100  # Large penalty for collisions
        
#         # Goal cost with parking-specific adjustments
#         if self.parking_mode:
#             # In parking mode, prioritize alignment with goal orientation
#             angle_diff = abs(self._normalize_angle(next_state[2] - self.goal[2]))
#             pos_cost = self._distance(next_state, self.goal)
            
#             # If we're close to the goal, prioritize orientation
#             if pos_cost < 2.0:  # Within 2 meters of goal
#                 cost += self.w_goal * (angle_diff * 3)  # Triple the weight for angle difference
#             else:
#                 cost += self.w_goal * (pos_cost + angle_diff)
#         else:
#             # Normal mode: distance-based cost
#             cost += self.w_goal * self._distance(next_state, self.goal)
        
#         return cost

#     def _heuristic(self, state):
#         """Calculate heuristic with improved parking behavior"""
#         if self.parking_mode:
#             # In parking mode, consider both position and orientation
#             pos_cost = self._distance(state, self.goal)
#             angle_diff = abs(self._normalize_angle(state[2] - self.goal[2]))
            
#             # If we're close to the goal, prioritize orientation
#             if pos_cost < 2.0:  # Within 2 meters of goal
#                 return pos_cost + angle_diff * 3  # Triple the weight for angle difference
#             else:
#                 return pos_cost + angle_diff
#         else:
#             # Normal mode: just use distance
#             return self._distance(state, self.goal)

#     def _move(self, node, steer, gear):
#         """Move the vehicle according to the bicycle model with reverse capability"""
#         # Get current state
#         x, y, yaw = node
        
#         # Calculate motion based on bicycle model
#         # gear: 1 for forward, -1 for reverse
#         # step_size: distance to move
#         # steer: steering angle
        
#         # Calculate the radius of curvature
#         if abs(steer) < 1e-6:  # If steering angle is very small
#             # Straight line motion
#             x_new = x + self.step_size * np.cos(yaw) * gear
#             y_new = y + self.step_size * np.sin(yaw) * gear
#             yaw_new = yaw
#         else:
#             # Curved motion
#             # Using bicycle model with wheelbase = 1.75m (GEM e2 specs)
#             wheelbase = 1.75
#             radius = wheelbase / np.tan(steer)
            
#             # Calculate center of rotation
#             center_x = x - radius * np.sin(yaw)
#             center_y = y + radius * np.cos(yaw)
            
#             # Calculate angle to move
#             angle = (self.step_size * gear) / radius
            
#             # Calculate new position
#             x_new = center_x + radius * np.sin(yaw + angle)
#             y_new = center_y - radius * np.cos(yaw + angle)
#             yaw_new = self._normalize_angle(yaw + angle)
        
#         return (x_new, y_new, yaw_new)

#     def _is_valid(self, state):
#         """Check if the state is valid (within bounds and not in collision)"""
#         x, y, yaw = state
        
#         # Check if within bounds
#         if not (0 <= x <= self.env.lx and 0 <= y <= self.env.ly):
#             return False
        
#         # Check for collision
#         if self._is_collision(state):
#             return False
        
#         return True

#     def _is_collision(self, state):
#         """Check if the state is in collision with any obstacle"""
#         x, y, yaw = state
        
#         # Create a simple car model for collision checking
#         # Using GEM e2 dimensions: length = 2.62m, width = 1.41m
#         car_length = 2.62
#         car_width = 1.41
        
#         # Calculate car corners
#         cos_yaw = np.cos(yaw)
#         sin_yaw = np.sin(yaw)
        
#         # Car corners relative to center
#         corners = [
#             (car_length/2, car_width/2),   # front right
#             (car_length/2, -car_width/2),  # front left
#             (-car_length/2, -car_width/2), # rear left
#             (-car_length/2, car_width/2)   # rear right
#         ]
        
#         # Transform corners to world frame
#         world_corners = []
#         for corner in corners:
#             world_x = x + corner[0] * cos_yaw - corner[1] * sin_yaw
#             world_y = y + corner[0] * sin_yaw + corner[1] * cos_yaw
#             world_corners.append((world_x, world_y))
        
#         # Check if any corner is in collision with obstacles
#         for ob in self.env.obs:
#             ob_x, ob_y, ob_w, ob_h = ob
#             for corner in world_corners:
#                 if (ob_x <= corner[0] <= ob_x + ob_w and 
#                     ob_y <= corner[1] <= ob_y + ob_h):
#                     return True
        
#         return False

#     def _distance(self, state1, state2):
#         """Calculate distance between two states"""
#         x1, y1, _ = state1
#         x2, y2, _ = state2
#         return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#     def _normalize_angle(self, angle):
#         """Normalize angle to [-pi, pi]"""
#         return np.arctan2(np.sin(angle), np.cos(angle)) 