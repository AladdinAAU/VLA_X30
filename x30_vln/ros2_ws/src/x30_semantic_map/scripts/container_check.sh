#!/bin/bash
# Runs INSIDE the x30-framework container (see run_ros_integration.sh): builds x30_msgs, x30_common and x30_semantic_map
# from the read-only source mount, runs the pytest suite, starts the semantic_mapper node and the ROS integration check.
set -eo pipefail
. /usr/local/bin/x30-env
export ROS_DOMAIN_ID="${SEMMAP_DOMAIN:-77}" PYTHONDONTWRITEBYTECODE=1
cd /ws
colcon build --packages-up-to x30_semantic_map --event-handlers console_cohesion- 2>&1 | tail -15
. /ws/install/setup.bash
PKG=/ws/src/x30_semantic_map
echo "== pytest (installed package, ROS python environment)"
cd /tmp
[ -n "${SEMMAP_SKIP_PYTEST:-}" ] || python3 -m pytest -q -p no:cacheprovider -p no:anyio "$PKG/test" --ignore="$PKG/test/ros_integration_check.py" 2>&1 | tail -8 && PYRC=${PIPESTATUS[0]} || PYRC=$?
echo "== semantic_mapper + integration check"
ros2 run x30_semantic_map semantic_mapper --ros-args -p vlm_enabled:=false -p config_file:="${X30_CONFIGS:-/x30/configs}/semantic_map.yaml" \
    > /tmp/semantic_mapper.log 2>&1 &
MAPPER=$!
trap 'kill $MAPPER 2>/dev/null || true' EXIT
python3 "$PKG/test/ros_integration_check.py" --rate "${SEMMAP_RATE:-5}" --out /tmp/semantic_map_integration.json --png /tmp/semantic_map_ros.png "$@" && RC=0 || RC=$?
echo "== mapper log (tail)"; tail -n 8 /tmp/semantic_mapper.log
cat /tmp/semantic_map_integration.json
if [ -d "${SEMMAP_OUT:-/nonexistent}" ]; then cp /tmp/semantic_map_ros.png "$SEMMAP_OUT/asm_ros_integration.png" 2>/dev/null || true; fi
if [ -d "${SEMMAP_JSON_OUT:-/nonexistent}" ]; then cp /tmp/semantic_map_integration.json "$SEMMAP_JSON_OUT"/ 2>/dev/null || true; fi
[ "${PYRC:-0}" = 0 ] || { echo 'pytest FAILED'; RC=1; }
exit $RC
