#!/usr/bin/env bash
# Use the taught places: from the spawn, "Go to <name>." with locate_nav (the model reports where, the planner drives); the semantic map holds ONLY the places and
# objects taught/seen during the tour (results/maps/<tour>.json), no ground truth.   bash scripts/teach/run_use.sh [tour name, default tour_2]
export MSYS_NO_PATHCONV=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; T="${1:-tour_2}"
run() { # name instruction target-room
  docker exec e2e-fw bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80; python3 /x30/scripts/e2e/run_episode.py --name use_${T}_$1 --world aws_hospital --instruction \"$2\" --start 0.93 11.48 0.0 --load-map-raw /x30/results/maps/$T.json --rooms /x30/configs/rooms/aws_hospital_tour.yaml --rooms-score-only --target-room \"$3\" --policy-name locate_nav --timeout 900 > /x30/results/e2e/logs/use_${T}_$1.log 2>&1"
}
run marcus "Go to Marcus." "Room C"
run alexander "Go to Alexander's room." "Room E"
run delivery "Take this package to the delivery point." "Delivery spot"
echo done > "$ROOT/results/e2e/use_${T}.done"
