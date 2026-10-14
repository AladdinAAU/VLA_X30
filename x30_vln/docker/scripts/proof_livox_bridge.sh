#!/usr/bin/env bash
# ROS 1 native CustomMsg -> Foxy custom bridge -> Humble, on loopback only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="x30-livox-proof-$$"
MASTER="$PREFIX-master"; BRIDGE="$PREFIX-bridge"; SOURCE="$PREFIX-source"; RECEIVER="$PREFIX-receiver"
cleanup() { docker rm -f "$RECEIVER" "$SOURCE" "$BRIDGE" "$MASTER" >/dev/null 2>&1 || true; }
trap cleanup EXIT
ROS1=(-e ROS_MASTER_URI=http://127.0.0.1:${LIVOX_TEST_MASTER_PORT:-11321} -e ROS_IP=127.0.0.1)
DDS=(-e ROS_DOMAIN_ID=${LIVOX_TEST_DOMAIN:-69} -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
     -e CYCLONEDDS_URI=file:///etc/x30/cyclonedds-local.xml -v "$ROOT/docker/dds:/etc/x30:ro")
docker run -d --name "$MASTER" --network host "${ROS1[@]}" x30-livox-bridge:latest ros1 roscore -p "${LIVOX_TEST_MASTER_PORT:-11321}" >/dev/null
docker run -d --name "$BRIDGE" --network host "${ROS1[@]}" "${DDS[@]}" -e BRIDGE_NODE_NAME=x30_livox_proof \
    -e BRIDGE_TOPICS=/x30/configs/ros1_bridge_topics_livox.yaml -v "$ROOT/configs:/x30/configs:ro" x30-livox-bridge:latest >/dev/null
docker exec "$BRIDGE" x30-bridge-entrypoint ros2 run ros1_bridge dynamic_bridge --print-pairs | rg 'livox_ros_driver2'
sleep 4
docker run -d --name "$SOURCE" --network host "${ROS1[@]}" -v "$ROOT/docker/scripts:/probe:ro" \
    x30-livox-bridge:latest ros1 python3 /probe/livox_bridge_source.py >/dev/null
sleep 6
docker logs "$BRIDGE"
docker run -d --name "$RECEIVER" --network host "${DDS[@]}" -v "$ROOT/docker/scripts:/probe:ro" \
    x30-livox-framework:latest sleep infinity >/dev/null
docker exec "$RECEIVER" bash -lc 'python3 /probe/livox_bridge_receiver.py'
docker exec "$RECEIVER" bash -lc 'timeout 10 ros2 topic echo /livox/lidar livox_ros_driver2/msg/CustomMsg --no-daemon --field point_num --once'
docker exec "$RECEIVER" bash -lc 'ros2 interface show livox_ros_driver2/msg/CustomMsg'
