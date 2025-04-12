cmake_minimum_required(VERSION 2.8.3)
project(hybrid_a_star_sim)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  nav_msgs
  sensor_msgs
  geometry_msgs
  std_msgs
  ackermann_msgs
  visualization_msgs
)

catkin_package(
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/hybrid_astar_rs_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


