#!/usr/bin/env bash
# Start the follow-specific nodes on top of a running aws_hospital stack (bash scripts/e2e/run_e2e.sh aws_hospital first):
#   bash scripts/follow/run_follow_stack.sh [crowd file under /x30, default configs/follow/crowd1.json]
# - follow_person node (framework container, Humble), - person_mover (sim container, Jazzy, GZ_PARTITION=e2e) spawning the crowd
export MSYS_NO_PATHCONV=1
CROWD="${1:-configs/follow/crowd1.json}"
ENVS="source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80;"
mkdir -p "$(dirname "$0")/../../results/follow"
docker exec e2e-fw bash -lc "pkill -f '[f]ollow_person'; true"
docker exec e2e-sim bash -lc "pkill -f '[p]erson_mover'; true"
sleep 2
docker exec -d e2e-fw bash -lc "$ENVS ros2 launch x30_follow follow_person.launch.py use_sim_time:=true > /x30/results/follow/follow_node.log 2>&1"
docker exec -d e2e-sim bash -lc ". /usr/local/bin/x30-env; export ROS_DOMAIN_ID=80 GZ_PARTITION=e2e; python3 /ws/src/x30_sim/x30_sim/person_mover.py --ros-args -p crowd_file:=/x30/$CROWD > /x30/results/follow/mover.log 2>&1"
sleep 20
grep -E "verified|missing|failed|Error" "$(dirname "$0")/../../results/follow/mover.log" | grep -v rcutils | tail -3 | cut -c1-220
