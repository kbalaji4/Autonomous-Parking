#!/usr/bin/env python3

import os
import csv
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.animation as animation
from alvin import ll2xy, xy2ll

from dpp.env.grid import Grid
from dpp.env.car import SimpleCar
from dpp.env.environment import Environment
from dpp.env.map import Map
from dpp.methods.hybrid_astar import HybridAstar

class InteractivePlanner:
    def __init__(self):
        # Set origin GPS coordinates
        self.olat = 40.0928563
        self.olon = -88.2359994
        
        # Initialize map and environment
        self.map = Map()
        self.map.add_walls()
        self.env = Environment(self.map.obs, lx=self.map.lx, ly=self.map.ly)
        
        # Set fixed start position (using the test case values)
        start_x, start_y = self.wps_to_local_xy(self.olon, self.olat, self.olat, self.olon)
        start_yaw_deg = 90  # Start yaw in degrees (facing East)
        start_yaw_rad = self.car_heading_to_planner_yaw(start_yaw_deg)
        
        # Shift start coordinates relative to the static grid's top-left corner
        self.start_x_shifted = start_x - self.map.grid_top_left[0]
        self.start_y_shifted = self.map.ly - (self.map.grid_top_left[1] - start_y)
        self.start_pos = [self.start_x_shifted, self.start_y_shifted, start_yaw_rad]
        
        # Initialize plot
        self.setup_plot()
        
        # Initialize car parameters
        self.car = None
        self.goal_pos = None
        self.path = None
        self.closed_ = None
        self.anim = None
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Show plot
        plt.show()
    
    def wps_to_local_xy(self, lon_wp, lat_wp, olat, olon):
        """Convert GNSS waypoints into local fixed frame represented in x and y"""
        x, y = ll2xy(lat_wp, lon_wp, olat, olon)
        return x, y

    def local_xy_to_wps(self, x, y, olat, olon):
        """Convert local x,y coordinates back to GPS coordinates"""
        lat, lon = xy2ll(x, y, olat, olon)
        return lon, lat
    
    def setup_plot(self):
        """Initialize the plot with map and static elements"""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Set plot limits based on map dimensions
        self.ax.set_xlim(0, self.map.lx)
        self.ax.set_ylim(0, self.map.ly)
        self.ax.set_aspect("equal")
        
        # Add grid lines
        self.ax.set_xticks(np.arange(0, self.map.lx + 1, 10))
        self.ax.set_yticks(np.arange(0, self.map.ly + 1, 10))
        self.ax.grid(True)
        
        # Plot obstacles
        for ob in self.map.obs:
            self.ax.add_patch(Rectangle((ob[0], ob[1]), ob[2], ob[3], fc='gray', ec='k'))
        
        # Plot start point
        self.start_marker = self.ax.plot([self.start_pos[0]], [self.start_pos[1]], 'go', markersize=10)[0]
        self.start_heading = self.ax.plot([], [], 'g-', linewidth=2)[0]
        self.update_heading_line(self.start_heading, self.start_pos[0], self.start_pos[1], self.start_pos[2])
        
        # Initialize goal marker and heading
        self.goal_marker = self.ax.plot([], [], 'ro', markersize=10)[0]
        self.goal_heading = self.ax.plot([], [], 'r-', linewidth=2)[0]
        
        # Initialize path visualization elements
        self.path_line = self.ax.plot([], [], color='lime', linewidth=2)[0]
        self.car_trail = PatchCollection([])
        self.ax.add_collection(self.car_trail)
        self.car_model = PatchCollection([])
        self.ax.add_collection(self.car_model)
        
        # Initialize branch visualization
        self.branch_lines = []
        
        plt.title("Interactive Path Planner\nClick anywhere on the map to set goal point")
    
    def car_heading_to_planner_yaw(self, yaw):
        """Convert car heading to planner yaw"""
        planner_yaw = 0.0
        if yaw <= 90.0:
            planner_yaw = 90 - yaw
        else:
            planner_yaw = 450 - yaw
        return np.radians(planner_yaw % 360.0)
    
    def planner_yaw_to_car_heading(self, yaw_rad):
        """Convert planner yaw to car heading"""
        yaw_deg = np.degrees(yaw_rad)
        car_heading = (90 - yaw_deg) % 360.0
        return car_heading
    
    def on_click(self, event):
        """Handle mouse clicks to set goal point"""
        if event.inaxes != self.ax:
            return
        
        # Get click coordinates
        x = event.xdata
        y = event.ydata
        
        # Check if click is within map bounds
        if not (0 <= x <= self.map.lx and 0 <= y <= self.map.ly):
            print("Click must be within map bounds")
            return
        
        # Check if click is on an obstacle
        for ob in self.map.obs:
            if (ob[0] <= x <= ob[0] + ob[2] and 
                ob[1] <= y <= ob[1] + ob[3]):
                print("Cannot set goal on obstacle")
                return
        
        # Set goal position (yaw will be set via command line)
        self.goal_pos = [x, y, 0]  # Initial yaw of 0
        self.goal_marker.set_data([x], [y])
        self.update_heading_line(self.goal_heading, x, y, 0)
        print(f"Goal point set at ({x:.2f}, {y:.2f})")
        
        # Ask for goal yaw angle
        while True:
            try:
                yaw_deg = float(input("Enter goal yaw angle (0-360 degrees): "))
                if 0 <= yaw_deg <= 360:
                    break
                print("Please enter a value between 0 and 360")
            except ValueError:
                print("Please enter a valid number")
        
        # Convert yaw to radians and update goal position
        yaw_rad = self.car_heading_to_planner_yaw(yaw_deg)
        self.goal_pos[2] = yaw_rad
        self.update_heading_line(self.goal_heading, x, y, yaw_rad)
        print(f"Goal yaw set to {yaw_deg:.1f}°")
        
        # Plan path
        self.plan_path()
    
    def update_heading_line(self, line, x, y, yaw):
        """Update the heading line for a point"""
        length = 2.0  # meters
        line.set_data(
            [x, x + length * np.cos(yaw)],
            [y, y + length * np.sin(yaw)]
        )
    
    def plan_path(self):
        """Plan path using Hybrid A*"""
        # Clear previous branch visualization
        for line in self.branch_lines:
            line.remove()
        self.branch_lines.clear()
        
        # Clear previous path visualization
        self.path_line.set_data([], [])
        self.car_trail.set_paths([])
        self.car_model.set_paths([])
        
        # Stop previous animation if it exists
        if self.anim is not None:
            self.anim.event_source.stop()
        
        # Initialize car with current start and goal positions
        self.car = SimpleCar(self.env, self.start_pos, self.goal_pos)
        
        # Update car parameters to match GEM e2 specs
        self.car.l = 1.75  # Wheelbase: 69 in = 1.75m
        self.car.carl = 2.62  # Length: 103 in = 2.62m
        self.car.carw = 1.41  # Width: 55.5 in = 1.41m
        self.car.max_phi = 0.5  # Maximum steering angle
        
        # Initialize grid
        grid = Grid(self.env, cell_size=self.map.cell_size)
        
        # Initialize hybrid A* planner
        hastar = HybridAstar(self.car, grid, reverse=False)
        
        # Modify weights to prioritize orientation
        hastar.w1 = 0.8   # weight for astar heuristic
        hastar.w2 = 0.2   # weight for simple heuristic
        hastar.w3 = 0.8   # increased weight for steering angle change
        hastar.w4 = 0.6   # increased weight for turning
        hastar.w5 = 2.0   # weight for reversing
        
        # Plan path
        print("Planning path...")
        t = time()
        self.path, self.closed_ = hastar.search_path(heu=1, extra=True)
        print('Total time: {}s'.format(round(time()-t, 3)))
        
        if not self.path:
            print('No valid path found!')
            return
        
        # Visualize branches
        self.visualize_branches()
        
        # Create animation
        self.animate_path()
        
        # Save path to CSV
        self.save_path_to_csv()
    
    def visualize_branches(self):
        """Visualize the explored branches"""
        if not self.closed_:
            return
            
        # Extract branches for visualization
        branches = []
        bcolors = []
        for node in self.closed_:
            for b in node.branches:
                branches.append(b[1:])
                bcolors.append('y' if b[0] == 1 else 'b')
        
        # Plot branches
        for b, c in zip(branches, bcolors):
            x = [p[0] for p in b]
            y = [p[1] for p in b]
            line = self.ax.plot(x, y, color=c, linewidth=0.5, alpha=0.3)[0]
            self.branch_lines.append(line)
    
    def animate_path(self):
        """Animate the car following the path"""
        if not self.path:
            return
            
        # Extract path data
        xl, yl = [], []
        carl = []
        for state in self.path:
            xl.append(state.pos[0])
            yl.append(state.pos[1])
            carl.append(state.model[0])
        
        def animate(i):
            self.path_line.set_data(xl[min(i, len(self.path)-1):], yl[min(i, len(self.path)-1):])
            
            # Update car trail
            sub_carl = carl[:min(i+1, len(self.path))]
            self.car_trail.set_paths(sub_carl[::4])
            self.car_trail.set_color('m')
            self.car_trail.set_alpha(0.1)
            
            # Update car model
            edgecolor = ['k']*5 + ['r']
            facecolor = ['y'] + ['k']*4 + ['r']
            self.car_model.set_paths(self.path[min(i, len(self.path)-1)].model)
            self.car_model.set_edgecolor(edgecolor)
            self.car_model.set_facecolor(facecolor)
            self.car_model.set_zorder(3)
            
            return self.path_line, self.car_trail, self.car_model
        
        # Create animation
        frames = len(self.path) + 1
        self.anim = animation.FuncAnimation(self.fig, animate, frames=frames, interval=50, blit=True)
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def save_path_to_csv(self):
        """Save path waypoints to a CSV file"""
        # Create waypoints directory if it doesn't exist
        os.makedirs('waypoints', exist_ok=True)
        
        # Full path to the CSV file
        filepath = os.path.join('waypoints', 'hybrid_astar_path_original.csv')
        
        # Write path data to CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['x', 'y', 'yaw_deg'])
            # Write waypoints
            for state in self.path:
                writer.writerow([round(state.pos[0], 3), round(state.pos[1], 3), round(state.pos[2], 3)])
        
        print(f"Path saved to {filepath}")

def main():
    planner = InteractivePlanner()

if __name__ == '__main__':
    main() 