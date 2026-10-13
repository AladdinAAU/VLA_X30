#!/usr/bin/env bash
# Record one room episode as a rosbag (zstd) and run it.   bash scripts/e2e/record_episode.sh <Room letter> [tag]
# Bag: results/bags/room_<R>_<tag>  (camera, camera_info, /gt/pose, /tf, /tf_static, /odom, /cmd_vel*, /x30/safety, /x30/mode, /map, /nav/known_map_info, /clock).
# No lidar cloud (about 8 MB/s). Episode result: results/e2e/room_<R>_<tag>/result.json. Needs the aws_hospital stack (run_e2e.sh aws_hospital).
export MSYS_NO_PATHCONV=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
R="${1:-C}"; TAG="${2:-rec1}"; N="room_${R}_${TAG}"
ENVS="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80;"
docker exec e2e-fw bash -lc "rm -rf /x30/results/bags/$N"
docker exec -d e2e-fw bash -lc "$ENVS ros2 bag record -o /x30/results/bags/$N --compression-mode message --compression-format zstd \
  /camera/color/image_raw /camera/color/camera_info /gt/pose /tf /tf_static /odom /cmd_vel /cmd_vel_nav /x30/safety /x30/mode /nav/known_map_info /map /clock > /x30/results/e2e/logs/bag_$N.log 2>&1"
sleep 6
docker exec e2e-fw bash -lc "$ENVS python3 /x30/scripts/e2e/run_episode.py --name $N --world aws_hospital --instruction 'Go to Room $R.' --start 0.93 11.48 0.0 --load-map /x30/results/maps/aws_hospital.gt.json --rooms /x30/configs/rooms/aws_hospital.yaml --target-room 'Room $R' --policy-name locate_nav --timeout 900 > /x30/results/e2e/logs/$N.log 2>&1"
sleep 3
docker exec e2e-fw bash -lc "pkill -INT -f '[r]os2 bag record'; sleep 6; ls -la /x30/results/bags/$N | tail -4; du -sh /x30/results/bags/$N"
