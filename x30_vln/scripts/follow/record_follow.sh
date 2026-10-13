#!/usr/bin/env bash
# Record a follow_person episode as a rosbag and run it.   bash scripts/follow/record_follow.sh <name> [extra run_follow_episode args]
# Bag: results/bags/<name> (camera, /gt/pose, /gt/person, /follow/status, /tf*, /odom, /cmd_vel*, /x30/safety, /x30/mode, /map, /clock). Needs the stack from docs/FOLLOW_PERSON.md.
export MSYS_NO_PATHCONV=1
N="${1:-follow_rec1}"; shift
ENVS="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80;"
docker exec e2e-fw bash -lc "rm -rf /x30/results/bags/$N"
docker exec -d e2e-fw bash -lc "$ENVS ros2 bag record -o /x30/results/bags/$N --compression-mode message --compression-format zstd \
  /camera/color/image_raw /camera/color/camera_info /gt/pose /gt/person /gt/people /follow/status /tf /tf_static /odom /cmd_vel /cmd_vel_nav /x30/safety /x30/mode /map /clock > /x30/results/follow/bag_$N.log 2>&1"
sleep 6
docker exec e2e-fw bash -lc "$ENVS python3 /x30/scripts/follow/run_follow_episode.py --name $N $* > /x30/results/follow/$N.log 2>&1"
sleep 3
docker exec e2e-fw bash -lc "pkill -INT -f '[r]os2 bag record'; sleep 6; du -sh /x30/results/bags/$N"
