cmake_minimum_required(VERSION 2.8)
project(human_track)
#SET(CMAKE_CXX_FLAGS_RELEASE"$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_BULIE_TYPE RELEASE)
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
#add_definitions(-std=c++11)

#add_definitions(-D_USE_OPENCV)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(catkin REQUIRED COMPONENTS
  roslib
  cv_bridge
  rosbag
  tf
  image_transport
  roscpp
  std_msgs
  laser_geometry
  angles
  sensor_msgs
  filters
  laser_filters
  autoscrubber_services
  message_filters
  pluginlib
  geometry_msgs
  image_geometry
  nav_msgs
  camera_calibration_parsers
  message_generation
  costmap_2d
  map_msgs
  actionlib_msgs
)

# Enable OPENCV in darknet
add_definitions(-DOPENCV)
add_definitions(-O4 -g)

#ADD_LIBRARY(newmat STATIC IMPORTED)
#SET_TARGET_PROPERTIES(newmat PROPERTIES IMPORTED_LOCATION /usr/lib/libnewmat.a)
ADD_LIBRARY(csm STATIC IMPORTED)
SET_TARGET_PROPERTIES(csm PROPERTIES IMPORTED_LOCATION /opt/ros/kinetic/lib/libcsm-static.a)
ADD_LIBRARY(jsoncpp STATIC IMPORTED)
SET_TARGET_PROPERTIES(jsoncpp PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libjsoncpp.a)
ADD_LIBRARY(newmat STATIC IMPORTED)
SET_TARGET_PROPERTIES(newmat PROPERTIES IMPORTED_LOCATION /usr/lib/libnewmat.a)


find_package(PkgConfig REQUIRED)
pkg_check_modules(GSL REQUIRED gsl)
find_library(NEWMAT newmat /usr/lib/libnewmat.a REQUIRED)



#find_package(PkgConfig REQUIRED)
#pkg_check_modules(GSL REQUIRED gsl)
#set("OpenCV_INCLUDE_DIR" /usr/include/opencv2/)
#find_library(NEWMAT newmat /usr/lib/libnewmat.a REQUIRED)
## Generate messages in the 'msg' folder
add_message_files(
   FILES
   states.msg
   ClusterClass.msg
   ClusteredScan.msg
   HumansFeatClass.msg
   HumansFeat.msg
   IcpPose2d.msg
   BoundingBox.msg
   BoundingBoxes.msg
 )



## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   geometry_msgs sensor_msgs std_msgs actionlib_msgs#   move_base_msgs#   nav_msgs#   visualization_msgs
 )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need


# Resolve system dependency on yaml-cpp, which apparently does not
# provide a CMake find_package() module.

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}_lib
  CATKIN_DEPENDS
    cv_bridge
    roscpp
    std_msgs
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    image_transport
  DEPENDS
    Boost
)

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
#include_directories(${catkin_INCLUDE_DIRS})
#include_directories(${Boost_INCLUDE_DIRS})
include_directories(${NEWMAT_INCLUDE_DIRS})
include_directories(${GSL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${GLS_LIBRARY_DIRS})

file(GLOB extra_files ./lib/*.cpp ./lib/*.c ./lib/*.cc ./src/*.cpp ./src/*.c)
#add_library(libExtra STATIC ${extra_files})
#message("sources files: \n ${extra_files}")

add_library(${PROJECT_NAME}_lib
    libSSD/SSD_Detector.cpp
    libSSD/image_interface.cpp
    libSSD/fp16.c
    libSSD/ncs_param.cpp
    libSSD/facenet.cpp
    ${extra_files}
  )

  target_link_libraries(${PROJECT_NAME}_lib
    m
    pthread
    stdc++
    -lmvnc
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )



set(header ./include/)
include_directories(${header})
message("header files path: ${header}")

## Declare a cpp library
# add_library(trajectory_follower
#   src/${PROJECT_NAME}/trajectory_follower.cpp
# )

# file(GLOB src_files ./src/*.cpp ./src/*.c)
# add_library(libSrc STATIC ${src_files})
# message("sources files: \n ${src_files}")

# Declare a cpp executable  # EXECUTABLE NODES
add_executable(human_track main.cc)

#target_link_libraries(${PROJECT_NAME}
#  ${PROJECT_NAME}_lib
#)

#target_link_libraries(laserTracker ${OpenCV_LIBS} libSrc)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
target_link_libraries(human_track   ${PROJECT_NAME}_lib
                                    ${catkin_LIBRARIES}
                                    ${OpenCV_LIBS}
                                    ${GSL_LIBRARIES}
                                    ${Boost_LIBRARIES}
                                    #libKcf
                                    #libAdd
                                    newmat
                                    csm
				                    jsoncpp
                                    )


## Specify libraries to link a library or executable target against
# target_link_libraries(trajectory_follower_node
#   ${catkin_LIBRARIES}
# )

# LINK THE LIBRARIES
