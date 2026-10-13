#!/bin/bash
# Replay profile check inside the framework container (ROS_DOMAIN_ID=50). One background process: the launch.
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-50} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp X30_CONFIGS=/x30/configs X30_ROOT=/x30
unset CYCLONEDDS_URI
export PYTHONPATH=/ws/src/x30_hw:${PYTHONPATH:-}
ros2 launch x30_hw replay.launch.py > /tmp/replay_launch.log 2>&1 &
LAUNCH=$!
trap 'kill -INT $LAUNCH 2>/dev/null; sleep 2; pkill -9 -f "ros2 bag play"; true' EXIT
sleep 4
python3 /x30/scripts/hw/replay_check.py
exit $?
