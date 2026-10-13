#!/bin/bash
# Inside the x30-framework container (workspace already built by run_ros_check.sh, x30_modes too: colcon build --packages-select x30_modes).
# Real mode_manager + skill_server of x30_modes on the fake robot of x30_modes; our reasoner_agent delegated to through /x30/navigate.
. /opt/ros/humble/setup.bash; . /tmp/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-60} X30_CONFIG_DIR=/x30/configs PYTHONUNBUFFERED=1
[ -f /etc/x30/cyclonedds.xml ] && export CYCLONEDDS_URI=file:///etc/x30/cyclonedds.xml
M=/x30/ros2_ws/src/x30_modes/test/ros_integration; T=/x30/ros2_ws/src/x30_agent/tests/ros
export PYTHONPATH=$M:$PYTHONPATH
mkdir -p /tmp/logs /tmp/results
pkill -f "[f]ake_robot.py"; pkill -f "[c]hain_vlm.py"; pkill -f "[l]ib/x30_modes"; pkill -f "[l]ib/x30_agent"; sleep 1
nohup python3 $M/fake_robot.py > /tmp/logs/fake_robot.log 2>&1 &
nohup python3 $T/chain_vlm.py 8123 > /tmp/logs/chain_vlm.log 2>&1 &
export X30_REASONER_URL=http://127.0.0.1:8123/v1 X30_REASONER_MODEL=fake X30_RESULTS=/tmp/results
nohup ros2 launch x30_modes modes.launch.py profile:=sim > /tmp/logs/modes.log 2>&1 &
nohup ros2 run x30_agent reasoner_agent --ros-args -p run_name:=chain > /tmp/logs/reasoner.log 2>&1 &
sleep 10
python3 $T/chain_client.py
RC=$?
pkill -f "[f]ake_robot.py"; pkill -f "[c]hain_vlm.py"; pkill -f "[l]ib/x30_modes"; pkill -f "[l]ib/x30_agent"; pkill -f "[m]odes.launch"
exit $RC
