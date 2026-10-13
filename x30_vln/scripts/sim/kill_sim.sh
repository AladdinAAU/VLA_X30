#!/usr/bin/env bash
# Stop every process the X30 simulator starts (stray gz / ros nodes break the next run). Run inside the sim container.
pats=("ros2 launch x30_sim" "gz sim" "ruby.*gz" "parameter_bridge" "robot_state_publisher" "x30_sim/lib/x30_sim"
      "controller_manager/spawner" "static_transform_publisher" "ros_gz_sim/create")
me=$$
for sig in TERM KILL; do
    for p in "${pats[@]}"; do
        for pid in $(pgrep -f "$p"); do [ "$pid" != "$me" ] && [ "$pid" != "$PPID" ] && kill -"$sig" "$pid" 2>/dev/null; done
    done
    sleep 2
done
echo "x30 sim stopped"
