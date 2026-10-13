#!/bin/bash
# Inside an x30-framework container on network x30_net with the simulator running (ROS_DOMAIN_ID=30, kinematic locomotion).
#   run_sim_check.sh                      -> sim_check.py
#   run_sim_check.sh approach_trace.py 0.4
. /opt/ros/humble/setup.bash; . /tmp/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=30 X30_CONFIG_DIR=/x30/configs PYTHONUNBUFFERED=1
export CYCLONEDDS_URI=file:///etc/x30/cyclonedds.xml
T=/x30/ros2_ws/src/x30_modes/test/ros_integration; C=/x30/configs/modes
export PYTHONPATH=$T:$PYTHONPATH
mkdir -p /tmp/logs
pkill -f "[l]ib/x30_modes"; sleep 1
for n in cmd_vel_mux safety_guard mode_manager skill_server; do
    nohup ros2 run x30_modes $n --ros-args --params-file $C/sim.yaml --params-file $T/sim_overrides.yaml > /tmp/logs/$n.log 2>&1 &
done
sleep 6
SCRIPT=${1:-sim_check.py}
shift
python3 $T/$SCRIPT "$@"
RC=$?
pkill -f "[l]ib/x30_modes"
exit $RC
