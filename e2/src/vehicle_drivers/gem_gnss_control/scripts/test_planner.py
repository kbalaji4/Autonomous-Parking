import alvinxy.alvinxy as axy # Import AlvinXY transformation module
import numpy as np
import rospy
import numpy as np
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod
import math
import csv

lat = None
lon = None
heading = None


def wps_to_local_xy(lon_wp, lat_wp):
    global lat, lon
        # convert GNSS waypoints into local fixed frame reprented in x and y
    lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, lon, lat)
    return lon_wp_x, lat_wp_y   
    
def get_gem_state():
    global lat, lon, heading

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
    local_x_curr, local_y_curr = wps_to_local_xy(lon, lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
    curr_yaw = heading_to_yaw(heading) 
    #

        # reference point is located at the center of rear axle
    curr_x = local_x_curr - 0.46 * np.cos(curr_yaw)
    curr_y = local_y_curr - 0.46 * np.sin(curr_yaw)

    return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

def gnss_callback(msg):
    global lat, lon
    lat = round(msg.latitude, 6)
    lon = round(msg.longitude, 6)

def ins_callback( msg):
    global heading
    heading = round(msg.heading, 6)
    
def heading_to_yaw(heading_curr):
    if (heading_curr >= 270 and heading_curr < 360):
        yaw_curr = np.radians(450 - heading_curr)
    else:
        yaw_curr = np.radians(90 - heading_curr)
    return yaw_curr

def wait_for_pose():
    global lon, lat, heading
    while not rospy.is_shutdown() and (lon is None or lat is None or heading is None):
        rospy.sleep(0.1)

def generate(curr_x, curr_y, curr_yaw, distance, step=0.5, output_file="waypoints_test.csv"):
    waypoints = []

    for i in range(int(distance / step) + 1):
        x = curr_x + i * step * math.cos(curr_yaw)
        y = curr_y + i * step * math.sin(curr_yaw)
        waypoints.append((round(x, 3), round(y, 3), round(curr_yaw, 4)))

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "yaw"])  # Header
        writer.writerows(waypoints)

    print(f"Waypoints saved to {output_file}")
        
def run(clat, clon, cheading):
    global lat, lon, heading
    lat = clat   
    lon = clon
    heading = cheading
    curr_x, curr_y, curr_yaw = get_gem_state()
    print(f"Current position: x={curr_x}, y={curr_y}, yaw={curr_yaw}")
    generate(curr_x, curr_y, curr_yaw, 10.0, step=0.5, output_file="waypoints_test.csv")
    
if __name__ == "__main__":
    run()
    
    