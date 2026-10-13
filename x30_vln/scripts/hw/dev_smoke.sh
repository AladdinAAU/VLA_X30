#!/bin/bash
# Quick manual smoke test inside the framework container: fake_x30 + x30_nav_bridge on loopback, stand, drive, e-stop.
# Never touches a real address (fake_x30 refuses anything but loopback).
source /opt/ros/humble/setup.bash; source /ws/install/setup.bash
export PYTHONPATH=/ws/src/x30_hw:$PYTHONPATH RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 -m x30_hw.tools.fake_x30 --time-scale 5 > /tmp/fake.log 2>&1 &
sleep 1
ros2 run x30_nav_bridge nav_bridge_node --ros-args -p motion_host_ip:=127.0.0.1 -p max_vx_forward:=0.4 > /tmp/nb.log 2>&1 &
sleep 3
ros2 service call /nav_bridge_node/acquire_control std_srvs/srv/Trigger
ros2 service call /nav_bridge_node/stand std_srvs/srv/Trigger
ros2 service call /nav_bridge_node/enable_motion std_srvs/srv/SetBool "{data: true}"
ros2 topic pub -r 20 /x30/cmd_vel_hw geometry_msgs/msg/Twist "{linear: {x: 0.3, y: 0.05}, angular: {z: 0.2}}" > /dev/null &
sleep 4
timeout 3 ros2 topic echo --once /x30/leg_odom --field twist.twist
python3 -c "
from x30_hw.tools.fake_x30 import control_request
import json; print(json.dumps(control_request({'cmd':'status'})))"
ros2 service call /nav_bridge_node/soft_estop std_srvs/srv/Trigger
sleep 1
timeout 3 ros2 topic echo --once /x30/estop_active
pkill -f nav_bridge_node; pkill -f fake_x30; pkill -f "ros2 topic"
echo ---- NB LOG; tail -30 /tmp/nb.log
