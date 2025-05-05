#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import time

from dpp.env.environment import Environment
from dpp.env.car import SimpleCar
from dpp.methods.hybrid_astar import HybridAstar
from dpp.utils.coordinate_transform import CoordinateTransform



def setup_plot(grid_size):
    """Setup the matplotlib plot for visualization"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def plot_obstacles(ax, env):
    """Plot obstacles in the environment"""
    for ob in env.obs:
        ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))

def plot_car(ax, pos, yaw, color='blue'):
    """Plot the car at the given position and orientation"""
    # Simple car representation
    car_length = 4.0
    car_width = 2.0
    
    # Calculate car corners
    x, y = pos[0], pos[1]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # Car corners relative to center
    corners = np.array([
        [-car_length/2, -car_width/2],
        [car_length/2, -car_width/2],
        [car_length/2, car_width/2],
        [-car_length/2, car_width/2]
    ])
    
    # Rotate and translate corners
    rotated_corners = np.dot(corners, np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]]))
    rotated_corners += np.array([x, y])
    
    # Plot car
    car_patch = plt.Polygon(rotated_corners, closed=True, fill=True, color=color, alpha=0.5)
    ax.add_patch(car_patch)

def plot_path(ax, path, color='green'):
    """Plot the planned path"""
    if path is None:
        return
        
    x = [state.pos[0] for state in path]
    y = [state.pos[1] for state in path]
    ax.plot(x, y, color=color, linewidth=2)

def main():
    # Initialize coordinate transformer
    coord_transform = CoordinateTransform(
        center_lat=40.0928174,
        center_lon=-88.2356714,
        grid_size=80.0
    )
    
    # Create environment
    env = Environment()
    
    # Create car
    car = SimpleCar(env)
    
    # Initialize hybrid A* planner
    planner = HybridAstar(car, env.grid, reverse=True)
    
    # Modify weights for smoother paths
    planner.w1 = 0.95  # weight for astar heuristic
    planner.w2 = 0.05  # weight for simple heuristic
    planner.w3 = 0.50  # increased weight for steering angle change
    planner.w4 = 0.30  # increased weight for turning
    planner.w5 = 2.00  # weight for reversing
    
    # Set start position (center of grid)
    start_x, start_y = 40.0, 40.0  # Center of 80x80 grid
    start_yaw = 0.0
    car.start_pos = [start_x, start_y, start_yaw]
    
    # Set goal position
    goal_x, goal_y = 10.0, 10.0
    goal_yaw = 0.0
    car.end_pos = [goal_x, goal_y, goal_yaw]
    
    # Setup visualization
    fig, ax = setup_plot(80.0)
    plot_obstacles(ax, env)
    
    # Plot start and goal positions
    plot_car(ax, [start_x, start_y], start_yaw, color='blue')
    plot_car(ax, [goal_x, goal_y], goal_yaw, color='red')
    
    # Plan path
    print("Planning path...")
    start_time = time.time()
    path, closed = planner.search_path(heu=1, extra=True)
    end_time = time.time()
    print(f"Path planning took {end_time - start_time:.2f} seconds")
    
    if path is not None:
        print("Path found!")
        # Plot the path
        plot_path(ax, path)
        
        # Plot car positions along the path
        for i in range(0, len(path), 5):  # Plot every 5th position
            state = path[i]
            plot_car(ax, [state.pos[0], state.pos[1]], state.pos[2], color='green')
    else:
        print("No valid path found!")
    
    plt.title("Hybrid A* Path Planning Test")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main() 