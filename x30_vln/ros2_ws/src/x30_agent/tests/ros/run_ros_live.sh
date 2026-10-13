#!/bin/bash
# Like run_ros_check.sh but the agent nodes use the real model server of configs/models.yaml (http://vlm:8000/v1 on x30_net).
. /opt/ros/humble/setup.bash; . /tmp/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-60} X30_CONFIG_DIR=/x30/configs PYTHONUNBUFFERED=1 X30_RESULTS=/tmp/results
[ -f /etc/x30/cyclonedds.xml ] && export CYCLONEDDS_URI=file:///etc/x30/cyclonedds.xml
T=/x30/ros2_ws/src/x30_agent/tests/ros
mkdir -p /tmp/logs /tmp/results
pkill -f "[f]ake_stack.py"; pkill -f "[l]ib/x30_agent"; sleep 1
python3 $T/fake_stack.py --configs /x30/configs --port 8123 > /tmp/logs/fake_stack.log 2>&1 &
ros2 run x30_agent reasoner_agent --ros-args -p run_name:=ros_live > /tmp/logs/reasoner_live.log 2>&1 &
ros2 run x30_agent instruction_follower --ros-args -p run_name:=ros_live > /tmp/logs/follower_live.log 2>&1 &
sleep 8
cd $T && python3 ros_live.py /x30/results/agent/ros_live.json "$@"
RC=$?
pkill -f "[f]ake_stack.py"; pkill -f "[l]ib/x30_agent"
exit $RC
