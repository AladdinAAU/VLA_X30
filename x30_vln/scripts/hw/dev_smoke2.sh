#!/bin/bash
source /opt/ros/humble/setup.bash; source /ws/install/setup.bash
export PYTHONPATH=/ws/src/x30_hw:$PYTHONPATH RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 -m x30_hw.tools.fake_x30 --time-scale 5 > /tmp/fake.log 2>&1 &
sleep 1
ros2 run x30_nav_bridge nav_bridge_node --ros-args -p motion_host_ip:=127.0.0.1 --log-level debug > /tmp/nb.log 2>&1 &
sleep 3
ros2 service call /nav_bridge_node/stand std_srvs/srv/Trigger
python3 -c "
from x30_hw.tools.fake_x30 import control_request
import json; print(json.dumps(control_request({'cmd':'status'})))"
pkill -f nav_bridge_node; pkill -f fake_x30
grep -a -E "0x2101|发送控制" /tmp/nb.log | head; grep -a -c . /tmp/nb.log
