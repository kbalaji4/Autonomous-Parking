Simulator
source ~/Desktop/GEM/devel/setup.bash
roslaunch gem_launch gem_init.launch world_name:="highbay_track.world" x:=-1.5 y:=-21 yaw:=3.1416
This is in the simulator repo, launches gazebo
roslaunch gem_launch gem_sensor_info.launch
rosrun hybrid_a_star_sim hybrid_astar_rs_node.py --goal_x -28 --goal_y -28.0 --goal_yaw 180
Hybrid a\*
rosrun hybrid_a_star_sim pure_pursuit.py
Rosrun <package_name aka the dir> <py file>
Catkin_make clean

GEMe2

source ~/Desktop/GEM/devel/setup.bash
roslaunch basic_launch sensor_init.launch
roslaunch basic_launch visualization.launch
roslaunch basic_launch dbw_joystick.launch

# Map

rosrun path_planning/plot_map_node.py

# Path Planning

rosrun path_planning/hybrid_astar_node.py

# Controller

rosrun gem_gnss_control controller.py

# Human Detection

rosrun camera_detect_node.py

# Python Testing

Can test path planning in non ros enviornment by running test_hybrid_astar.py

### Video Link:

https://www.youtube.com/watch?v=JsVNtm1iyrs
