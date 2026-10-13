#!/usr/bin/env bash
# Old (planner assist off) vs new on the corner scenario: bash scripts/follow/run_corner_ab.sh   (stack up; takes about 2 x 6 min wall)
export MSYS_NO_PATHCONV=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
bash "$ROOT/scripts/follow/run_follow_stack.sh" configs/follow/corner1.json > /dev/null 2>&1
docker exec e2e-fw bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80; python3 /x30/scripts/follow/run_follow_episode.py --name corner2_base --no-planner --timeout 105 > /x30/results/follow/corner2_base.log 2>&1"
bash "$ROOT/scripts/follow/record_follow.sh" corner2_plan --timeout 105 > "$ROOT/results/follow/record_corner2_plan.log" 2>&1
echo done > "$ROOT/results/follow/corner2.done"
