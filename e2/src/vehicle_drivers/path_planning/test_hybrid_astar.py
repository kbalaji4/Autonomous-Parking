#!/usr/bin/env python3

import os
import csv
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from scipy.interpolate import splprep, splev
import matplotlib.animation as animation

from dpp.env.grid import Grid
from dpp.env.car import SimpleCar
from dpp.env.environment import Environment
from dpp.test_cases.cases import TestCase
from dpp.methods.hybrid_astar import HybridAstar

def smooth_path(path, smoothing=0.3):
    """Smooth the path using B-spline interpolation"""
    # Extract x, y coordinates
    x = [state.pos[0] for state in path]
    y = [state.pos[1] for state in path]
    
    # Check if we have enough points for smoothing
    if len(path) < 4:
        print("Warning: Path too short for smoothing, returning original path")
        return path
    
    try:
        # Create B-spline representation with increased smoothing
        tck, u = splprep([x, y], s=smoothing, k=3)  # k=3 for cubic spline
        
        # Generate more points for smoother curve
        # Increase the number of points by a factor of 4
        u_new = np.linspace(0, 1, len(path) * 4)
        x_new, y_new = splev(u_new, tck)
        
        # Calculate yaw angles for new points using central differences
        dx = np.gradient(x_new)
        dy = np.gradient(y_new)
        yaw_new = np.arctan2(dy, dx)
        
        # Create new path with smoothed points
        smoothed_path = []
        for i in range(len(x_new)):
            pos = [x_new[i], y_new[i], yaw_new[i]]
            state = path[0].__class__(pos, path[0].model)  # Create new state with same class
            smoothed_path.append(state)
        
        # Print statistics about the smoothing
        print(f"Original path points: {len(path)}")
        print(f"Smoothed path points: {len(smoothed_path)}")
        
        return smoothed_path
    except Exception as e:
        print(f"Warning: Path smoothing failed: {str(e)}")
        print("Returning original path")
        return path

def save_path_to_csv(path, filename):
    """Save path waypoints to a CSV file"""
    # Create waypoints directory if it doesn't exist
    os.makedirs('waypoints', exist_ok=True)
    
    # Full path to the CSV file
    filepath = os.path.join('waypoints', filename)
    
    # Write path data to CSV
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['x', 'y', 'yaw'])
        # Write waypoints
        for state in path:
            writer.writerow([state.pos[0], state.pos[1], state.pos[2]])

def plot_path(env, path, closed_):
    """Plot the path with animation"""
    # Extract path data
    xl, yl = [], []
    carl = []
    for state in path:
        xl.append(state.pos[0])
        yl.append(state.pos[1])
        carl.append(state.model[0])
    
    # Extract branches for visualization
    branches = []
    bcolors = []
    for node in closed_:
        for b in node.branches:
            branches.append(b[1:])
            bcolors.append('y' if b[0] == 1 else 'b')
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.lx)
    ax.set_ylim(0, env.ly)
    ax.set_aspect("equal")
    
    # Add grid lines
    ax.set_xticks(np.arange(0, env.lx + 1, 10))
    ax.set_yticks(np.arange(0, env.ly + 1, 10))
    ax.grid(True)
    
    # Plot obstacles
    for ob in env.obs:
        ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))
    
    # Plot branches
    for b, c in zip(branches, bcolors):
        x = [p[0] for p in b]
        y = [p[1] for p in b]
        ax.plot(x, y, color=c, linewidth=0.5, alpha=0.3)
    
    # Plot path
    _path, = ax.plot([], [], color='lime', linewidth=2)
    _carl = PatchCollection([])
    ax.add_collection(_carl)
    _car = PatchCollection([])
    ax.add_collection(_car)
    
    def animate(i):
        _path.set_data(xl[min(i, len(path)-1):], yl[min(i, len(path)-1):])
        
        sub_carl = carl[:min(i+1, len(path))]
        _carl.set_paths(sub_carl[::4])
        _carl.set_color('m')
        _carl.set_alpha(0.1)
        
        edgecolor = ['k']*5 + ['r']
        facecolor = ['y'] + ['k']*4 + ['r']
        _car.set_paths(path[min(i, len(path)-1)].model)
        _car.set_edgecolor(edgecolor)
        _car.set_facecolor(facecolor)
        _car.set_zorder(3)
        
        return _path, _carl, _car
    
    # Create animation
    frames = len(path) + 1
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
    plt.title("GEM e2 Path Planning Visualization")
    plt.show()

def main():
    # Create test case
    tc = TestCase()
    
    # Initialize environment and car
    env = Environment(tc.obs, lx=40.0, ly=40.0)  # Set environment size to 40x40
    car = SimpleCar(env, tc.start_pos, tc.end_pos)
    
    # Update car parameters to match GEM e2 specs
    car.l = 1.75  # Wheelbase: 69 in = 1.75m
    car.carl = 2.62  # Length: 103 in = 2.62m
    car.carw = 1.41  # Width: 55.5 in = 1.41m
    # Calculate max steering angle based on turning radius
    # turning_radius = wheelbase / tan(max_steering_angle)
    # 3.175 = 1.75 / tan(max_steering_angle)
    # max_steering_angle = arctan(1.75/3.175) ≈ 0.5 radians
    car.max_phi = 0.5  # Maximum steering angle
    
    grid = Grid(env, cell_size=0.25)  # 0.25m cell size
    
    # Initialize hybrid A* planner with modified parameters for smoother paths
    hastar = HybridAstar(car, grid, reverse=True)
    
    # Modify weights for smoother paths
    hastar.w1 = 0.95  # weight for astar heuristic
    hastar.w2 = 0.05  # weight for simple heuristic
    hastar.w3 = 0.50  # increased weight for steering angle change
    hastar.w4 = 0.30  # increased weight for turning
    hastar.w5 = 2.00  # weight for reversing
    
    # Plan path
    print("Planning path...")
    t = time()
    path, closed_ = hastar.search_path(heu=1, extra=True)
    print('Total time: {}s'.format(round(time()-t, 3)))
    
    if not path:
        print('No valid path found!')
        return
    
    # Downsample path for waypoints (use smaller step for shorter paths)
    step = max(1, len(path) // 30)  # Ensure we get at least 30 points
    path = path[::step] + [path[-1]]
    
    # Smooth the path
    print("Smoothing path...")
    smoothed_path = smooth_path(path, smoothing=0.2)  # Reduced smoothing for more detail
    
    # Save both original and smoothed paths to CSV
    save_path_to_csv(path, 'hybrid_astar_path_original.csv')
    save_path_to_csv(smoothed_path, 'hybrid_astar_path_smoothed.csv')
    print(f"Paths saved to waypoints/")
    
    # Print some statistics
    print(f"Number of waypoints (original): {len(path)}")
    print(f"Number of waypoints (smoothed): {len(smoothed_path)}")
    print(f"Start position: {path[0].pos}")
    print(f"Goal position: {path[-1].pos}")
    
    # Plot both paths
    print("Plotting original path...")
    plot_path(env, path, closed_)
    print("Plotting smoothed path...")
    plot_path(env, smoothed_path, closed_)

if __name__ == '__main__':
    main() 