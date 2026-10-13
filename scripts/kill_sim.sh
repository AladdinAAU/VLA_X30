#!/usr/bin/env bash
# Stop every process the X30 sim starts (stray gz/nav2 processes break the next run).
pats=("ros2 launch x30_" "gz sim" "ruby.*gz" "parameter_bridge" "robot_state_publisher"
      "x30_teleop_bridge" "nav2_" "opennav" "slam_toolbox" "controller_manager/spawner" "frontier")
for p in "${pats[@]}"; do pgrep -f "$p" | grep -vx "$$" | xargs -r kill -TERM 2>/dev/null; done; sleep 3
for p in "${pats[@]}"; do pgrep -f "$p" | grep -vx "$$" | xargs -r kill -KILL 2>/dev/null; done
echo "x30 sim stopped"
