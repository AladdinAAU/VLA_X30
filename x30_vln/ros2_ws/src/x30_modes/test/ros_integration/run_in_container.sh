#!/bin/bash
# usage: run_in_container.sh [group ...]   (inside the container, stack already started by start_stack.sh)
. /opt/ros/humble/setup.bash; . /tmp/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=30 X30_CONFIG_DIR=/x30/configs PYTHONUNBUFFERED=1
exec python3 /x30/ros2_ws/src/x30_modes/test/ros_integration/run_scenarios.py "$@"
