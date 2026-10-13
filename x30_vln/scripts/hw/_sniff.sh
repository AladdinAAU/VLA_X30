source /opt/ros/humble/setup.bash; source /ws/install/setup.bash
python3 /x30/scripts/hw/_sniff.py &
sleep 0.5
ros2 run x30_nav_bridge nav_bridge_node --ros-args -p motion_host_ip:=127.0.0.1 --log-level debug > /tmp/nb.log 2>&1 &
sleep 2
timeout 7 ros2 service call /nav_bridge_node/stand std_srvs/srv/Trigger > /dev/null
wait %1
pkill -9 -f nav_bridge_node

