#!/usr/bin/env bash
set -e

# The parent image has both ROS versions; initialize a Noetic-only environment.
unset PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH ROS_DISTRO
source /opt/ros/noetic/setup.bash
exec "$@"
