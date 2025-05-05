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
from alvin import ll2xy, xy2ll

from dpp.env.grid import Grid
from dpp.env.car import SimpleCar
from dpp.env.environment import Environment
from dpp.test_cases.cases import TestCase
from dpp.methods.hybrid_astar import HybridAstar

def wps_to_local_xy(lon_wp, lat_wp, olat, olon):
    """Convert GNSS waypoints into local fixed frame represented in x and y"""
    x, y = ll2xy(lat_wp, lon_wp, olat, olon)
    return x, y

def local_xy_to_wps(x, y, olat, olon):
    """Convert local x,y coordinates back to GPS coordinates"""
    lat, lon = xy2ll(x, y, olat, olon)
    return lon, lat

def smooth_path(path, smoothing=0.3):
    """Smooth the path using B-spline interpolation"""
    # Extract x, y coordinates
    x = [state.pos[0] for state in path]
    y = [state.pos[1] for state in path]
    
    # Extract yaw angles and convert to radians for interpolation
    yaw = [np.radians(state.pos[2]) for state in path]
    
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
        
        # Ensure smooth yaw transition
        # Calculate the target yaw change
        start_yaw = np.radians(path[0].pos[2])
        end_yaw = np.radians(path[-1].pos[2])
        
        # Create a smooth yaw transition
        yaw_progress = np.linspace(0, 1, len(x_new))
        yaw_transition = start_yaw + (end_yaw - start_yaw) * yaw_progress
        
        # Blend between path-based yaw and target yaw
        blend_weight = np.exp(-5 * (yaw_progress - 0.5)**2)  # Gaussian blend
        yaw_new = (1 - blend_weight) * yaw_transition + blend_weight * yaw_new
        
        # Create new path with smoothed points
        smoothed_path = []
        for i in range(len(x_new)):
            pos = [x_new[i], y_new[i], np.degrees(yaw_new[i])]
            state = path[0].__class__(pos, path[0].model)  # Create new state with same class
            smoothed_path.append(state)
        
        # Ensure start and end yaw angles are exactly as specified
        smoothed_path[0].pos[2] = path[0].pos[2]
        smoothed_path[-1].pos[2] = path[-1].pos[2]
        
        # Print statistics about the smoothing
        print(f"Original path points: {len(path)}")
        print(f"Smoothed path points: {len(smoothed_path)}")
        
        return smoothed_path
    except Exception as e:
        print(f"Warning: Path smoothing failed: {str(e)}")
        print("Returning original path")
        return path

def save_path_to_csv(path, filename, olat, olon):
    """Save path waypoints to a CSV file with local coordinates and yaw in degrees"""
    # Create waypoints directory if it doesn't exist
    os.makedirs('waypoints', exist_ok=True)
    
    # Full path to the CSV file
    filepath = os.path.join('waypoints', filename)
    
    # Write path data to CSV
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['x', 'y', 'yaw_deg'])
        # Write waypoints
        for state in path:
            writer.writerow([round(state.pos[0], 3), round(state.pos[1], 3), round(state.pos[2], 3)])

def plot_path(env, path, closed_, olat, olon):
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
    # Set origin GPS coordinates
    olat = 40.0928563
    olon = -88.2359994
    
    # Create test case
    #tc = TestCase()
    
    
    
   



    print(wps_to_local_xy(-88.235190, 40.092727,  olat, olon))
    print(wps_to_local_xy(-88.236153, 40.092898,  olat, olon))

    
    # Define start and goal GPS coordinates and yaw angles (in degrees)
    slat =  olat
    slon = olon
    start_yaw_deg = 0.0  # Start yaw in degrees (facing East)
    
    glat = 40.0928328
    glon = -88.2353660
    goal_yaw_deg = 180  # Goal yaw in degrees (facing South)
    
    # Convert yaw angles from degrees to radians
    start_yaw_rad = np.radians(start_yaw_deg)
    goal_yaw_rad = np.radians(goal_yaw_deg)
    
    # Convert GPS coordinates to local coordinates
    start_x, start_y = wps_to_local_xy(slon, slat, olat, olon)
    goal_x, goal_y = wps_to_local_xy(glon, glat, olat, olon)
    
    # # Calculate environment size and center
    # dx = abs(goal_x - start_x)
    # dy = abs(goal_y - start_y)
    # env_size = max(dx, dy) * 2.0  # Make it twice as large as needed
    # env_size = max(env_size, 100.0)  # Ensure minimum size of 100m
    
    # # Calculate center point
    # center_x = (start_x + goal_x) / 2.0
    # center_y = (start_y + goal_y) / 2.0
    
    # # Shift coordinates relative to center
    # start_x_shifted = start_x - center_x + env_size/2
    # start_y_shifted = start_y - center_y + env_size/2
    # goal_x_shifted = goal_x - center_x + env_size/2
    # goal_y_shifted = goal_y - center_y + env_size/2
    
    # # Initialize environment and car with shifted coordinates and yaw angles
    # env = Environment(tc.obs, lx=100, ly=100)  # Set environment size based on coordinates
    
    grid_top_left = (-25, 10)
    grid_bottom_right = (85, -20)
    lx = grid_bottom_right[0] - grid_top_left[0]  # 110m
    ly = grid_top_left[1] - grid_bottom_right[1]  # 30m

    # Initialize environment with static grid dimensions
    obs = [[0 ,25.5, 110, 0.1],[94,0,0.1,30],[12,0,0.1,30],[0 ,5, 110, 0.1]]
    env = Environment(obs, lx=lx, ly=ly)
    print(start_y)
    print(goal_y)
    
    
    # Shift coordinates relative to the static grid's top-left corner
    start_x_shifted = start_x - grid_top_left[0]
    start_y_shifted = ly - (grid_top_left[1] - start_y)  # Flip y-axis
    goal_x_shifted = goal_x - grid_top_left[0]
    goal_y_shifted = grid_top_left[1] - goal_y  # Flip y-axis
    
    
    
    start_pos = [start_x_shifted, start_y_shifted, start_yaw_rad]  # Initial yaw in radians
    #goal_pos = [goal_x_shifted, goal_y_shifted, goal_yaw_rad]     # Final yaw in radians
    goal_pos = [90,10,goal_yaw_rad]
    car = SimpleCar(env, start_pos, goal_pos)
    
    # Update car parameters to match GEM e2 specs
    car.l = 1.75  # Wheelbase: 69 in = 1.75m
    car.carl = 2.62  # Length: 103 in = 2.62m
    car.carw = 1.41  # Width: 55.5 in = 1.41m
    car.max_phi = 0.5  # Maximum steering angle
    
    # Adjust grid size based on environment size
    cell_size = max(0.25, lx / 200)  # Ensure reasonable number of cells
    grid = Grid(env, cell_size=cell_size)
    
    # Initialize hybrid A* planner with modified parameters for smoother paths
    hastar = HybridAstar(car, grid, reverse=True)
    
    # #Modify weights to prioritize orientation
    # hastar.w1 = 0.8   # weight for astar heuristic
    # hastar.w2 = 0.2   # weight for simple heuristic
    # hastar.w3 = 0.8   # increased weight for steering angle change
    # hastar.w4 = 0.6   # increased weight for turning
    # hastar.w5 = 2   # weight for reversing
    
    # Plan path
    print("Planning path...")
    #print(f"Environment size: {env_size:.2f}m x {env_size:.2f}m")
    print(f"Cell size: {cell_size:.2f}m")
    #print(f"Environment center: x={center_x:.2f}, y={center_y:.2f}")
    print(f"Start position (local): x={start_x:.2f}, y={start_y:.2f}, yaw={start_yaw_deg:.3f}°")
    print(f"Goal position (local): x={goal_x:.2f}, y={goal_y:.2f}, yaw={goal_yaw_deg:.3f}°")
    print(f"Start position (shifted): x={start_x_shifted:.2f}, y={start_y_shifted:.2f}, yaw={start_yaw_deg:.3f}°")
    print(f"Goal position (shifted): x={goal_x_shifted:.2f}, y={goal_y_shifted:.2f}, yaw={goal_yaw_deg:.3f}°")
    t = time()
    path, closed_ = hastar.search_path(heu=1, extra=True)
    print('Total time: {}s'.format(round(time()-t, 3)))
    
    if not path:
        print('No valid path found!')
        return
    
    # Convert path back to original coordinates and yaw to degrees
    for state in path:
        state.pos[0] = state.pos[0] + grid_top_left[0]
        state.pos[1] = grid_top_left[1] - state.pos[1]
        # state.pos[0] = state.pos[0] + center_x - env_size/2
        # state.pos[1] = state.pos[1] + center_y - env_size/2
        state.pos[2] = np.degrees(state.pos[2])  # Convert yaw to degrees
        # Normalize yaw to [0, 360)
        state.pos[2] = state.pos[2] % 360.0
        state.pos[2] = (90-state.pos[2]) % 360.0
    
    # Downsample path for waypoints (use smaller step for shorter paths)
    step = max(1, len(path) // 50)  # Ensure we get at least 30 points
    path = path[::step] + [path[-1]]
    
    # Smooth the path
    #print("Smoothing path...")
    #smoothed_path = smooth_path(path, smoothing=0.2)  # Reduced smoothing for more detail
    
    # Ensure start and goal orientations are correct
   # smoothed_path[0].pos[2] = start_yaw_deg
    #smoothed_path[-1].pos[2] = goal_yaw_deg
    
    # Save both original and smoothed paths to CSV with local coordinates and yaw in degrees
    save_path_to_csv(path, 'hybrid_astar_path_original.csv', olat, olon)
    #save_path_to_csv(smoothed_path, 'hybrid_astar_path_smoothed.csv', olat, olon)
    print(f"Paths saved to waypoints/")
    
    # Print some statistics
    print(f"Number of waypoints (original): {len(path)}")
    #print(f"Number of waypoints (smoothed): {len(smoothed_path)}")
    #print(f"Start position: x={smoothed_path[0].pos[0]:.3f}, y={smoothed_path[0].pos[1]:.3f}, yaw={smoothed_path[0].pos[2]:.3f}°")
    #print(f"Goal position: x={smoothed_path[-1].pos[0]:.3f}, y={smoothed_path[-1].pos[1]:.3f}, yaw={smoothed_path[-1].pos[2]:.3f}°")
    
    # Plot both paths
    print("Plotting original path...")
    plot_path(env, path, closed_, olat, olon)
    #print("Plotting smoothed path...")
    #plot_path(env, smoothed_path, closed_, olat, olon)

if __name__ == '__main__':
    main() 