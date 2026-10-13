#!/usr/bin/env bash
# Host helper: sim_capture_world.sh <world_id> [n_poses]  -> boots the world in x30-sim (domain 70), captures frames, measures RTF, stops it.
w=$1; n=${2:-20}; cd "$(dirname "$0")/../.."
python -I datasets/scripts/make_view_poses.py "$w" "$n" 1
docker exec x30-sim bash /x30/datasets/scripts/sim_start_world.sh "$w"
docker exec x30-sim bash -c "source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=70 GZ_PARTITION=worlds70
  until ros2 topic list 2>/dev/null | grep -q /gt/objects; do sleep 5; done; sleep 40
  python3 /x30/scripts/sim/measure_rtf.py --seconds 20 --interval 20 2>&1 | tail -2 | cut -c1-260
  python3 /x30/datasets/scripts/sim_capture.py /x30/results/worlds_check/$w 2>&1 | tail -1"
docker exec x30-sim bash /x30/scripts/sim/kill_sim.sh >/dev/null 2>&1
