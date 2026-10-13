#!/bin/bash
# Inside an x30-framework container with the project mounted at /x30:  bash run_ros_check.sh [build]
# Builds x30_msgs, x30_common, x30_agent into /tmp/ws (once), starts fake_stack.py and both agent nodes, runs ros_check.py.
. /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-60} X30_CONFIG_DIR=/x30/configs PYTHONUNBUFFERED=1
[ -f /etc/x30/cyclonedds.xml ] && export CYCLONEDDS_URI=file:///etc/x30/cyclonedds.xml
T=/x30/ros2_ws/src/x30_agent/tests/ros
if [ "$1" = build ] || [ ! -f /tmp/ws/install/setup.bash ]; then
    mkdir -p /tmp/ws/src && for p in x30_msgs x30_common x30_agent; do ln -sfn /x30/ros2_ws/src/$p /tmp/ws/src/$p; done
    (cd /tmp/ws && colcon build --symlink-install --packages-select x30_msgs x30_common x30_agent --cmake-args -DBUILD_TESTING=OFF 2>&1 | tail -5) || exit 1
fi
. /tmp/ws/install/setup.bash
mkdir -p /tmp/logs /tmp/results
pkill -f "[f]ake_stack.py"; pkill -f "[l]ib/x30_agent"; sleep 1
python3 $T/fake_stack.py --configs /x30/configs --port 8123 > /tmp/logs/fake_stack.log 2>&1 &
export X30_REASONER_URL=http://127.0.0.1:8123/v1 X30_REASONER_MODEL=fake X30_RESULTS=/tmp/results
ros2 run x30_agent reasoner_agent --ros-args -p run_name:=ros_check > /tmp/logs/reasoner.log 2>&1 &
ros2 run x30_agent instruction_follower --ros-args -p run_name:=ros_check > /tmp/logs/follower.log 2>&1 &
sleep 8
python3 $T/ros_check.py /x30/results/agent/ros_check.json
RC=$?
pkill -f "[f]ake_stack.py"; pkill -f "[l]ib/x30_agent"
exit $RC
