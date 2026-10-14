#!/bin/bash
# Entry point of the ros1_bridge image (runs after /ros_entrypoint.sh has sourced Noetic and Foxy).
#   bridge   (default) wait for the ROS 1 master, load the topic list, run the bridge (BRIDGE_MODE=parameter|dynamic)
#   ros1 <cmd...>  run a ROS 1 tool (roscore, rosbag play ..., rostopic ...) in a Noetic-only environment: the Foxy variables
#                  that /ros_entrypoint.sh also sources would shadow Noetic's python modules (rosgraph_msgs) and break them
#   anything else is executed as given (bash, ros2 ...)
# Env: ROS_MASTER_URI (required for "bridge"), ROS_IP or ROS_HOSTNAME (address the ROS 1 side advertises; must be reachable
# from the master/robot), BRIDGE_TOPICS (yaml), BRIDGE_MASTER_TIMEOUT (s, default 120), BRIDGE_NODE_NAME (optional ROS 1 name).
set -e
if [ "${1:-bridge}" = "ros1" ]; then
    shift
    unset PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH ROS_DISTRO
    . /opt/ros/noetic/setup.bash
    if [ -f /opt/livox/ros1/setup.bash ]; then . /opt/livox/ros1/setup.bash; fi
    exec "$@"
fi

# Noetic's overlay resets PATH; restore Foxy before loading the ROS 2 overlays.
if [ -f /opt/livox/ros1/setup.bash ]; then
    . /opt/livox/ros1/setup.bash
    . /opt/ros/foxy/setup.bash
fi
for setup in /opt/livox/ros2/local_setup.bash /opt/livox/bridge/local_setup.bash; do
    if [ -f "$setup" ]; then . "$setup"; fi
done
if [ "${1:-bridge}" != "bridge" ]; then exec "$@"; fi

: "${ROS_MASTER_URI:?ROS_MASTER_URI must be set, e.g. http://192.168.1.106:11311}"
TIMEOUT="${BRIDGE_MASTER_TIMEOUT:-120}"
HOST_PORT="${ROS_MASTER_URI#*//}"
M_HOST="${HOST_PORT%%:*}"; M_PORT="${HOST_PORT##*:}"; M_PORT="${M_PORT%%/*}"

echo "[bridge] ROS_MASTER_URI=${ROS_MASTER_URI} ROS_IP=${ROS_IP:-} ROS_HOSTNAME=${ROS_HOSTNAME:-} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} RMW=${RMW_IMPLEMENTATION}"
echo "[bridge] waiting for the ROS 1 master at ${M_HOST}:${M_PORT} (max ${TIMEOUT}s)"
for ((i = 0; i < TIMEOUT; i++)); do
    if timeout 2 bash -c "exec 3<>/dev/tcp/${M_HOST}/${M_PORT}" 2>/dev/null; then break; fi
    sleep 1
done
timeout 2 bash -c "exec 3<>/dev/tcp/${M_HOST}/${M_PORT}" 2>/dev/null || { echo "[bridge] ERROR: master not reachable"; exit 1; }

if [ "${BRIDGE_MODE:-parameter}" = "dynamic" ]; then
    echo "[bridge] dynamic_bridge (bridges every topic that has a publisher AND a subscriber on opposite sides)"
    exec ros2 run ros1_bridge dynamic_bridge ${BRIDGE_ARGS:-}
fi

TOPICS="${BRIDGE_TOPICS}"
if [ ! -f "$TOPICS" ]; then
    echo "[bridge] $TOPICS not found, using the image default /opt/x30/topics.default.yaml"
    TOPICS=/opt/x30/topics.default.yaml
fi
echo "[bridge] topic list: $TOPICS"; grep -E '^\s+topic:' "$TOPICS" | sed 's/^/[bridge]   /'

# rosparam needs a Noetic-only environment (the ROS 2 variables would shadow its python modules): load the list in a
# subshell, then run the bridge from the combined environment (same order as Rob7)
( unset PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH ROS_DISTRO
  . /opt/ros/noetic/setup.bash
  rosparam load "$TOPICS" )
REMAP=()
if [ -n "${BRIDGE_NODE_NAME:-}" ]; then
    REMAP+=("__name:=${BRIDGE_NODE_NAME}")
fi
exec ros2 run ros1_bridge parameter_bridge "${REMAP[@]}"
