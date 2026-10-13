#!/usr/bin/env bash
# Record a guided tour as a rosbag and run it.   bash scripts/teach/record_tour.sh <name>     (stack: run_e2e.sh aws_hospital, run_follow_stack.sh configs/follow/tour1.json, teach node)
export MSYS_NO_PATHCONV=1
N="${1:-tour_rec1}"
ENVS="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80;"
docker exec e2e-fw bash -lc "rm -rf /x30/results/bags/$N"
docker exec -d e2e-fw bash -lc "$ENVS ros2 bag record -o /x30/results/bags/$N --compression-mode message --compression-format zstd \
  /camera/color/image_raw /gt/pose /gt/person /gt/people /follow/status /teach/reply /tf /tf_static /odom /cmd_vel /x30/safety /x30/mode /map /clock > /x30/results/teach/bag_$N.log 2>&1"
sleep 6
docker exec e2e-fw bash -lc "$ENVS python3 /x30/scripts/teach/run_tour.py --name $N > /x30/results/teach/$N.log 2>&1"
sleep 3
docker exec e2e-fw bash -lc "pkill -INT -f '[r]os2 bag record'; sleep 6; du -sh /x30/results/bags/$N"
