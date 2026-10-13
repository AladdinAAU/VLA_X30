#!/usr/bin/env bash
# start_world.sh <world_id> [seconds]  - boot the simulator headless (domain 70, own gz partition), kinematic, all sensors
export ROS_DOMAIN_ID=70 GZ_PARTITION=worlds70
source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash
nohup ros2 launch x30_sim sim.launch.py world:=$1 locomotion:=kinematic > /tmp/sim_worlds.log 2>&1 &
