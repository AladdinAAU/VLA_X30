#!/bin/bash
# Dry-run integration test inside the framework container (loopback only, ROS_DOMAIN_ID=50):
#   fake_x30 + x30_nav_bridge + real_adapter + camera_ingest (synthetic source) + static TF, then integration_dryrun.py.
# Usage (host):  docker exec x30-hw-test bash /x30/scripts/hw/run_dryrun.sh
# Exactly one background process (the launch) is started and it is always stopped on exit.

source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-50} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp X30_CONFIGS=/x30/configs
unset CYCLONEDDS_URI
export PYTHONPATH=/ws/src/x30_hw:${PYTHONPATH:-}
ros2 launch x30_hw real.launch.py dry_run:=true stage:=B fake_time_scale:=5 camera_url:=synthetic://640x480@10 > /tmp/dryrun_launch.log 2>&1 &
LAUNCH=$!
trap 'kill -INT $LAUNCH 2>/dev/null; sleep 2; pkill -9 -f nav_bridge_node; pkill -9 -f fake_x30; true' EXIT
sleep 6
python3 /x30/scripts/hw/integration_dryrun.py
RC=$?
echo "integration exit code $RC"
exit $RC
