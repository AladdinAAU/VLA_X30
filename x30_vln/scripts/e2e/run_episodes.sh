#!/usr/bin/env bash
# ep1 (sofa, from spawn) was run by hand: python3 /x30/scripts/e2e/run_episode.py --name ep1_sofa --instruction "Find the sofa and stop next to it." --label sofa
# The three episodes of docs/END_TO_END.md, one after the other (each ~10-15 min wall at RTF 0.33). Start poses are world-frame x y yaw.
export MSYS_NO_PATHCONV=1
run() { docker exec e2e-fw bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80; python3 /x30/scripts/e2e/run_episode.py $* > /x30/results/e2e/logs/$2.log 2>&1"; }
run --name ep2_chair --instruction \"Find a chair and stop next to it.\" --label chair --start 3.6 -4.6 3.14 --timeout 900
run --name ep3_tv --instruction \"Find the tv and stop next to it.\" --label tv --start 4.09 -2.71 0.0 --timeout 900
