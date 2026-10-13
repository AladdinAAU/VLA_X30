#!/usr/bin/env bash
# Proof (b): real-robot chain without the robot. Noetic rosbag play -> ros1_bridge (Foxy, CycloneDDS) -> Humble framework container.
# All on the Docker network x30_net. Containers: x30-infra-master (roscore), x30-infra-bridge, x30-infra-bagplay, x30-infra-humble.
# Usage: proof_bridge.sh [bagfile on host] [loop|once]
set -u
export MSYS_NO_PATHCONV=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if command -v cygpath >/dev/null 2>&1; then ROOT="$(cygpath -m "$ROOT")"; fi
BAG="${1:-$ROOT/../2026-10-06-17-05-52.bag}"
MODE="${2:-once}"
BAGDIR="$(dirname "$BAG")"; BAGNAME="$(basename "$BAG")"
OUT="$ROOT/results/infra"; mkdir -p "$OUT"
MASTER=http://x30-infra-master:11311
DDS=(-e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e CYCLONEDDS_URI=file:///etc/x30/cyclonedds.xml -v "$ROOT/docker/dds:/etc/x30:ro")

docker rm -f x30-infra-master x30-infra-bridge x30-infra-bagplay >/dev/null 2>&1
docker run -d --name x30-infra-master --network x30_net -e ROS_MASTER_URI=$MASTER -e ROS_HOSTNAME=x30-infra-master x30-ros1-bridge:latest ros1 roscore >/dev/null
sleep 4
docker run -d --name x30-infra-bridge --network x30_net -e ROS_MASTER_URI=$MASTER -e ROS_HOSTNAME=x30-infra-bridge -e BRIDGE_TOPICS=/nonexistent "${DDS[@]}" -e CYCLONEDDS_URI=file:///etc/x30/cyclonedds-foxy.xml x30-ros1-bridge:latest >/dev/null
sleep 8
echo "== bridge log"; docker logs x30-infra-bridge 2>&1 | tail -n 12
if ! docker ps --format '{{.Names}}' | grep -qx x30-infra-humble; then
  docker run -d --name x30-infra-humble --network x30_net --hostname humble --network-alias framework "${DDS[@]}" -e ROS_DOMAIN_ID=0 -v "$ROOT/docker/scripts:/probe:ro" x30-framework:latest sleep infinity >/dev/null
fi
docker exec -e CYCLONEDDS_URI=file:///etc/x30/cyclonedds.xml x30-infra-humble bash -lc "python3 /probe/count_topic.py /lidar_points --seconds $([ "$MODE" = loop ] && echo 40 || echo 90) --idle-exit $([ "$MODE" = loop ] && echo 0 || echo 6)" > "$OUT/bridge_count_$MODE.txt" 2>&1 &
CNT=$!
sleep 4
PLAYARGS=""; [ "$MODE" = loop ] && PLAYARGS="-l"
docker run -d --name x30-infra-bagplay --network x30_net -e ROS_MASTER_URI=$MASTER -e ROS_HOSTNAME=x30-infra-bagplay -v "$BAGDIR:/data:ro" x30-ros1-bridge:latest ros1 rosbag play -q $PLAYARGS "/data/$BAGNAME" >/dev/null
sleep 8
docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' x30-infra-bridge x30-infra-bagplay x30-infra-humble | tee "$OUT/bridge_stats_$MODE.txt"
wait $CNT
echo "== receiver (framework container)"; cat "$OUT/bridge_count_$MODE.txt"
echo "== bag play log"; docker logs x30-infra-bagplay 2>&1 | tail -n 6
docker rm -f x30-infra-master x30-infra-bridge x30-infra-bagplay >/dev/null 2>&1
