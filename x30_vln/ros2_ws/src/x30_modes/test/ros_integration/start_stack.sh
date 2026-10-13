#!/bin/bash
# usage: start_stack.sh [world_json]   (run inside the x30-framework container after building x30_modes into /tmp/ws)
. /opt/ros/humble/setup.bash; . /tmp/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=30 X30_CONFIG_DIR=/x30/configs PYTHONUNBUFFERED=1
pkill -f fake_robot.py; pkill -f fake_vlm.py; pkill -f stub_agent.py; pkill -f "lib/x30_modes"; pkill -f modes.launch ; sleep 1
mkdir -p /tmp/logs
T=/x30/ros2_ws/src/x30_modes/test/ros_integration
export PYTHONPATH=$T:$PYTHONPATH
WORLD=()
[ -n "$1" ] && WORLD=(--ros-args -p "world_json:=$1")
nohup python3 $T/fake_robot.py "${WORLD[@]}" > /tmp/logs/fake.log 2>&1 &
nohup python3 $T/stub_agent.py > /tmp/logs/agent.log 2>&1 &
nohup python3 $T/fake_vlm.py 8123 chair > /tmp/logs/vlm.log 2>&1 &
export X30_REASONER_URL=http://127.0.0.1:8123/v1 X30_REASONER_MODEL=fake
sleep 1
nohup ros2 launch x30_modes modes.launch.py profile:=sim > /tmp/logs/modes.log 2>&1 &
sleep 6
ros2 node list; ros2 action list | head -30
