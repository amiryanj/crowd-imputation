cmake_minimum_required(VERSION 2.8.3)
project(repbot)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_python_setup()

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )


catkin_package(CATKIN_DEPENDS rospy std_msgs message_runtime)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
        scripts/simulation_node.py
        scripts/fusion_node.py
        scripts/tracking_node.py
        scripts/robot_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


