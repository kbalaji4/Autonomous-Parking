import numpy as np
import math
import heapq
from collections import deque
from scipy.spatial import distance

XY_GRID_RESOLUTION = 1.0
YAW_GRID_RESOLUTION = np.deg2rad(15.0)
MOVE_STEP_SIZE = 1.0
WB = 2.0
MAX_STEER = np.deg2rad(30.0)
N_STEER = 5

def reeds_shepp_path(start, goal, radius):
    d = distance.euclidean([start[0], start[1]], [goal[0], goal[1]])
    steps = int(d / 0.5)
    path = []
    for i in range(steps + 1):
        t = i / steps
        x = start[0] + t * (goal[0] - start[0])
        y = start[1] + t * (goal[1] - start[1])
        theta = start[2] + t * (goal[2] - start[2])
        path.append((x, y, theta))
    return path

class Node:
    def __init__(self, x_ind, y_ind, yaw_ind, direction, x, y, yaw, cost, parent_index):
        self.x_ind = x_ind
        self.y_ind = y_ind
        self.yaw_ind = yaw_ind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.cost = cost
        self.parent_index = parent_index
    def __lt__(self, other):
        return self.cost < other.cost

def calc_index(x, y, yaw):
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)
    return x_ind, y_ind, yaw_ind

def hybrid_astar(start, goal):
    start_index = calc_index(start[0], start[1], start[2])
    goal_index = calc_index(goal[0], goal[1], goal[2])
    
    open_list = []
    closed_set = set()
    node_map = {}

    start_node = Node(*start_index, True, start[0], start[1], start[2], 0.0, None)
    heapq.heappush(open_list, (start_node.cost, start_node))
    node_map[(start_index, True)] = start_node

    while open_list:
        _, current = heapq.heappop(open_list)
        key = ((current.x_ind, current.y_ind, current.yaw_ind), current.direction)

        if key in closed_set:
            continue
        closed_set.add(key)

        dx = current.x - goal[0]
        dy = current.y - goal[1]
        dist_to_goal = math.hypot(dx, dy)

        if dist_to_goal <= 2.0:
            return reconstruct_path(current, node_map, goal)

        for steer in np.linspace(-MAX_STEER, MAX_STEER, N_STEER):
            for direction in [True, False]:
                step = MOVE_STEP_SIZE if direction else -MOVE_STEP_SIZE
                next_x = current.x + step * math.cos(current.yaw)
                next_y = current.y + step * math.sin(current.yaw)
                next_yaw = current.yaw + step / WB * math.tan(steer)

                x_ind, y_ind, yaw_ind = calc_index(next_x, next_y, next_yaw)
                new_key = ((x_ind, y_ind, yaw_ind), direction)

                if new_key in closed_set:
                    continue

                cost = current.cost + MOVE_STEP_SIZE + (0.1 if direction != current.direction else 0.0)
                new_node = Node(x_ind, y_ind, yaw_ind, direction,
                                next_x, next_y, next_yaw, cost, key)

                if new_key not in node_map or node_map[new_key].cost > cost:
                    node_map[new_key] = new_node
                    heapq.heappush(open_list, (new_node.cost, new_node))

    return []

def reconstruct_path(goal_node, node_map, goal):
    path = deque()
    node = goal_node
    while node is not None:
        path.appendleft((node.x, node.y, node.yaw))
        node = node_map.get(node.parent_index)
    path.extend(reeds_shepp_path(path[-1], goal, radius=2.0))
    return list(path)

def plot_path(path, start, goal):
    import matplotlib.pyplot as plt
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    plt.plot(xs, ys, '-b')
    plt.plot(start[0], start[1], 'og', label='Start')
    plt.plot(goal[0], goal[1], 'xr', label='Goal')
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Hybrid A* Path")
    plt.show()
