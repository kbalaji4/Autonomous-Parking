
#!/usr/bin/env python3

"""
inputs so that we can rosrun
"""
import rospy
import math
import numpy as np
from sensor_msgs.msg import NavSatFix, Imu #e4
# Fix is for e2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion
import pyproj
import argparse

import sys
import os

import csv
import time
from threading import Lock
import matplotlib.pyplot as plt
# Ensure the `scripts` directory is at the very beginning of the Python path
# scripts_dir = os.path.dirname(__file__)
# if scripts_dir not in sys.path:
#     sys.path.insert(0, scripts_dir)

# print("printing paths in main(): ")
# print(sys.path)

from constants import STARTX, STARTY, STARTYAW

def plot_from_csv(planned_path_file, actual_path_file):
    """Plot both planned and actual paths from CSV files"""
    # Read planned path
    start = None
    goal = None
    planned_path = []
    with open(planned_path_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row[0] == 'start':
                start = tuple(map(float, row[1:]))
            elif row[0] == 'goal':
                goal = tuple(map(float, row[1:]))
            elif row[0] == 'path':
                planned_path.append(tuple(map(float, row[1:])))
    
    # Read actual vehicle trajectory
    actual_path = []
    target_indices = []
    with open(actual_path_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row[0] == 'actual':
                actual_path.append(tuple(map(float, row[2:5])))  # Skip timestamp
                if len(row) > 5:
                     # target waypoint idx is provided in this case
                     target_indices.append(int(row[5]))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot planned path
    planned_xs = [p[0] for p in planned_path]
    planned_ys = [p[1] for p in planned_path]
    planned_yaws = [p[2] for p in planned_path]
    ax.plot(planned_xs, planned_ys, '-b', label='Planned Path')
    ax.quiver(planned_xs, planned_ys, 
                   np.cos(planned_yaws), np.sin(planned_yaws), 
                   angles='xy', scale_units='xy', scale=1, color='r', label="Yaw")
    
    # Plot actual path
    actual_xs = [p[0] for p in actual_path]
    actual_ys = [p[1] for p in actual_path]
    actual_yaws = [p[2] for p in actual_path]
    ax.plot(actual_xs, actual_ys, '-r', label='Actual Path')
    ax.quiver(actual_xs, actual_ys, 
                   np.cos(actual_yaws), np.sin(actual_yaws), 
                   angles='xy', scale_units='xy', scale=1, color='b', label="Yaw")
    if target_indices:
         for i, (actual_pos, target_idx) in enumerate(zip(actual_path, target_indices)):
             if i % 5 == 0:  # every 5th connection
                 ax.plot([actual_pos[0], planned_path[target_idx][0]], 
                        [actual_pos[1], planned_path[target_idx][1]], 
                        'g--', alpha=0.75, linewidth=0.5)
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'og', label='Start')
    ax.plot(goal[0], goal[1], 'xr', label='Goal')
    
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title('Planned vs Actual Path')
    plt.show()


# if __name__ == "main":
    # change these
planned_path_file = "/home/gem/bkkw0/e2/planner_path_data_20250419-134535.csv"
actual_path_file = "/home/gem/bkkw0/e2/vehicle_trajectory_20250419-134532.csv"
plot_from_csv(planned_path_file, actual_path_file)
