#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.animation as animation
from scipy.interpolate import splprep, splev
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpp.env.environment import Environment
from dpp.env.car import SimpleCar
from dpp.env.grid import Grid
from dpp.methods.hybrid_astar import HybridAstar
from dpp.utils.coordinate_transform import CoordinateTransform

def smooth_path(path, smoothing=0.3):
    """Smooth the path using B-spline interpolation"""
    # Extract x, y coordinates
    x = [state.pos[0] for state in path]
    y = [state.pos[1] for state in path]
    
    # Create B-spline representation
    tck, u = splprep([x, y], s=smoothing)
    
    # Generate more points for smoother curve
    u_new = np.linspace(0, 1, len(path) * 2)
    x_new, y_new = splev(u_new, tck)
    
    # Calculate yaw angles for new points
    dx = np.gradient(x_new)
    dy = np.gradient(y_new)
    yaw_new = np.arctan2(dy, dx)
    
    # Create new path with smoothed points
    smoothed_path = []
    for i in range(len(x_new)):
        pos = [x_new[i], y_new[i], yaw_new[i]]
        state = path[0].__class__(pos, path[0].model)  # Create new state with same class
        smoothed_path.append(state)
    
    return smoothed_path

class InteractivePathPlanner:
    def __init__(self):
        # Initialize coordinate transformer
        self.coord_transform = CoordinateTransform(
            center_lat=40.0928174,
            center_lon=-88.2356714,
            grid_size=80.0
        )
        
        # Create environment
        self.env = Environment(lx=80.0, ly=80.0)
        
        # Create grid from environment
        self.grid = Grid(self.env, cell_size=0.25)
        
        # Create car with GEM e2 specifications
        self.car = SimpleCar(self.env)
        # Update car parameters to match GEM e2 specs
        self.car.l = 1.75  # Wheelbase: 69 in = 1.75m
        self.car.carl = 2.62  # Length: 103 in = 2.62m
        self.car.carw = 1.41  # Width: 55.5 in = 1.41m
        # Calculate max steering angle based on turning radius
        # turning_radius = wheelbase / tan(max_steering_angle)
        # 3.175 = 1.75 / tan(max_steering_angle)
        # max_steering_angle = arctan(1.75/3.175) ≈ 0.5 radians
        self.car.max_phi = 0.5  # Maximum steering angle
        
        # Initialize hybrid A* planner
        self.planner = HybridAstar(self.car, self.grid, reverse=True)
        
        # Modify weights for smoother paths
        self.planner.w1 = 0.95  # weight for astar heuristic
        self.planner.w2 = 0.05  # weight for simple heuristic
        self.planner.w3 = 0.50  # increased weight for steering angle change
        self.planner.w4 = 0.30  # increased weight for turning
        self.planner.w5 = 2.00  # weight for reversing
        
        # Set start position (center of grid)
        self.start_x, self.start_y = 40.0, 40.0
        self.start_yaw = 0.0
        self.car.start_pos = [self.start_x, self.start_y, self.start_yaw]
        
        # Initialize goal position
        self.goal_x = None
        self.goal_y = None
        self.goal_yaw = 0.0
        
        # Initialize visualization variables
        self.path_line = None
        self.car_patches = []
        self.animation = None
        
        # Setup visualization
        self.fig, self.ax = self.setup_plot()
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def setup_plot(self):
        """Setup the matplotlib plot for visualization"""
        # Create figure and axes first
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Configure axes
        self.ax.set_xlim(0, 80.0)
        self.ax.set_ylim(0, 80.0)
        self.ax.set_aspect("equal")
        
        # Add grid lines
        self.ax.set_xticks(np.arange(0, 81, 10))
        self.ax.set_yticks(np.arange(0, 81, 10))
        self.ax.grid(True)
        
        # Plot obstacles
        for ob in self.env.obs:
            self.ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))
        
        # Plot start position
        self.plot_car(self.start_x, self.start_y, self.start_yaw, color='blue')
        
        plt.title("Click to set goal position\nPress 'q' to quit")
        return self.fig, self.ax
        
    def plot_car(self, x, y, yaw, color='blue'):
        """Plot the car at the given position and orientation"""
        # Use actual GEM e2 dimensions
        car_length = 2.62  # Length: 103 in = 2.62m
        car_width = 1.41   # Width: 55.5 in = 1.41m
        
        # Car corners relative to center
        corners = np.array([
            [-car_length/2, -car_width/2],
            [car_length/2, -car_width/2],
            [car_length/2, car_width/2],
            [-car_length/2, car_width/2]
        ])
        
        # Rotate and translate corners
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated_corners = np.dot(corners, rotation_matrix)
        rotated_corners += np.array([x, y])
        
        # Create and add car patch
        car_patch = Polygon(rotated_corners, closed=True, fill=True, color=color, alpha=0.5)
        self.ax.add_patch(car_patch)
        self.car_patches.append(car_patch)
        
    def plot_path(self, path):
        """Plot the planned path with animation"""
        if self.path_line is not None:
            self.path_line.remove()
            
        if path is None:
            return
            
        # Extract path data
        xl, yl = [], []
        carl = []
        for state in path:
            xl.append(state.pos[0])
            yl.append(state.pos[1])
            carl.append(state.model[0])
        
        # Plot path
        self.path_line, = self.ax.plot([], [], color='lime', linewidth=2)
        _carl = PatchCollection([])
        self.ax.add_collection(_carl)
        _car = PatchCollection([])
        self.ax.add_collection(_car)
        
        def animate(i):
            self.path_line.set_data(xl[min(i, len(path)-1):], yl[min(i, len(path)-1):])
            
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
            
            return self.path_line, _carl, _car
        
        # Create animation
        frames = len(path) + 1
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=frames, interval=50, blit=True
        )
        
    def clear_car_patches(self):
        """Clear all car position patches except the start position"""
        for patch in self.car_patches[1:]:
            patch.remove()
        self.car_patches = self.car_patches[:1]
        
    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return
            
        # Get clicked coordinates
        self.goal_x = event.xdata
        self.goal_y = event.ydata
        
        # Update car goal position
        self.car.end_pos = [self.goal_x, self.goal_y, self.goal_yaw]
        
        # Clear previous path and car positions
        if self.path_line is not None:
            self.path_line.remove()
            self.path_line = None
        if self.animation is not None:
            self.animation.event_source.stop()
        self.clear_car_patches()
        
        # Plot new goal position
        self.plot_car(self.goal_x, self.goal_y, self.goal_yaw, color='red')
        
        # Plan path
        print("Planning path...")
        start_time = time.time()
        path, closed_ = self.planner.search_path(heu=1, extra=True)
        end_time = time.time()
        print(f"Path planning took {end_time - start_time:.2f} seconds")
        
        if path is not None:
            print("Path found!")
            # Smooth the path
            smoothed_path = smooth_path(path, smoothing=0.3)
            # Plot the smoothed path
            self.plot_path(smoothed_path)
        else:
            print("No valid path found!")
            
        self.fig.canvas.draw()
        
    def run(self):
        """Run the interactive planner"""
        plt.show()

if __name__ == '__main__':
    planner = InteractivePathPlanner()
    planner.run() 