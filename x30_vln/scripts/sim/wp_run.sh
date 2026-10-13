#!/bin/bash
# usage: results_wp_run.sh <backend> <explore_s> <goals>
source /opt/ros/humble/setup.bash; source /ws/install/setup.bash
export ROS_DOMAIN_ID=20
B=$1; O=/x30/results/waypoints; mkdir -p $O
pkill -f ros2 2>/dev/null; pkill -f x30_navigation 2>/dev/null; sleep 2
ros2 service call /sim/set_robot_pose x30_msgs/srv/Json "{request: '{\"x\": 4.08, \"y\": 0.92, \"yaw\": 0.0}'}" | tail -1
nohup ros2 launch x30_navigation navigation.launch.py profile:=sim use_sim_time:=true odom_tf:=false odom_source:=leg planner_backend:=$B > $O/nav_$B.log 2>&1 &
nohup ros2 run x30_navigation cmd_relay > $O/relay_$B.log 2>&1 &
sleep 25
ros2 service call /nav/explore/start std_srvs/srv/Trigger >/dev/null
sleep $2
ros2 service call /nav/explore/stop std_srvs/srv/Trigger 2>&1 | tail -1
sleep 5
ros2 run x30_navigation waypoint_bench --gt-map /x30/ros2_ws/src/x30_worlds/worlds/replica_apartment_1/map.yaml --map-origin 4.08,0.92,0.0 --goals $3 --out $O/bench_$B.json 2>&1 | tail -$(($3+2))
