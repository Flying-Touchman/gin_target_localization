cmake_minimum_required(VERSION 2.8.3)
project(gin)

add_definitions(-std=c++11)


find_package(catkin REQUIRED COMPONENTS rospy )

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    )

include_directories(include ${catkin_INCLUDE_DIRS})
