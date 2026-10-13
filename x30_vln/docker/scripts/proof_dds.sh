#!/usr/bin/env bash
# Proof (a): Humble framework container <-> stock ros:jazzy container over CycloneDDS on the user-defined bridge network x30_net.
# Traffic in both directions at once (see dds_probe.py). Containers x30-infra-humble / x30-infra-jazzy are created by this script
# and stopped at the end. Usage: proof_dds.sh [best_effort|reliable] [duration_s] [dds_file]
set -u
export MSYS_NO_PATHCONV=1
QOS="${1:-best_effort}"; DUR="${2:-20}"; DDSFILE="${3:-cyclonedds.xml}"
TAG="${DDSFILE%.xml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if command -v cygpath >/dev/null 2>&1; then ROOT="$(cygpath -m "$ROOT")"; fi
OUT="$ROOT/results/infra"; mkdir -p "$OUT"
ENVS=(-e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e "CYCLONEDDS_URI=file:///etc/x30/$DDSFILE")
MNT=(-v "$ROOT/docker/dds:/etc/x30:ro" -v "$ROOT/docker/scripts:/probe:ro")

if ! docker ps --format '{{.Names}}' | grep -qx x30-infra-humble; then
    docker run -d --name x30-infra-humble --network x30_net --hostname humble --network-alias framework "${ENVS[@]}" "${MNT[@]}" x30-framework:latest sleep infinity >/dev/null
fi
if ! docker ps --format '{{.Names}}' | grep -qx x30-infra-jazzy; then
    docker run -d --name x30-infra-jazzy --network x30_net --hostname jazzy --network-alias sim "${ENVS[@]}" "${MNT[@]}" ros:jazzy-ros-base sleep infinity >/dev/null
fi
# the stock jazzy image has no CycloneDDS RMW; this is the only change made to it (the sim image installs the same package)
docker exec x30-infra-jazzy bash -c 'dpkg -s ros-jazzy-rmw-cyclonedds-cpp >/dev/null 2>&1 || (apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq --no-install-recommends ros-jazzy-rmw-cyclonedds-cpp >/dev/null 2>&1)'

docker exec -e "CYCLONEDDS_URI=file:///etc/x30/$DDSFILE" x30-infra-humble bash -lc "python3 /probe/dds_probe.py --pub h2j --sub j2h --duration $DUR --warmup 6 --qos $QOS" > "$OUT/dds_humble_${QOS}_$TAG.json" 2> "$OUT/dds_humble_${QOS}_$TAG.err" &
P1=$!
docker exec -e "CYCLONEDDS_URI=file:///etc/x30/$DDSFILE" x30-infra-jazzy bash -c "source /opt/ros/jazzy/setup.bash; python3 /probe/dds_probe.py --pub j2h --sub h2j --duration $DUR --warmup 6 --qos $QOS" > "$OUT/dds_jazzy_${QOS}_$TAG.json" 2> "$OUT/dds_jazzy_${QOS}_$TAG.err" &
P2=$!
sleep $((6 + DUR / 2))
docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' x30-infra-humble x30-infra-jazzy | tee "$OUT/dds_stats_${QOS}_$TAG.txt"
wait $P1; wait $P2
echo "== humble (publishes h2j, receives j2h)"; cat "$OUT/dds_humble_${QOS}_$TAG.json"
echo "== jazzy  (publishes j2h, receives h2j)"; cat "$OUT/dds_jazzy_${QOS}_$TAG.json"
