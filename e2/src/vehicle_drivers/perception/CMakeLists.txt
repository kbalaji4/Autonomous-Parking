cmake_minimum_required(VERSION 3.0.2)
project(perception)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  nav_msgs
  sensor_msgs
  geometry_msgs
  novatel_gps_msgs
  nav_msgs
  std_msgs
  visualization_msgs
  tf
  filters
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  camera_detection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 