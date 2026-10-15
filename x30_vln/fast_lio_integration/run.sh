#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
compose=(docker compose -f docker/compose.yml -f docker/compose.real.yml -f docker/compose.livox.yml -f fast_lio_integration/compose.yml --profile real)
case "${1:-}" in
  build)
    # A Windows checkout may give the existing Livox image scripts CRLF endings.
    scripts=(
      docker/images/livox_bridge/build-bridge.sh
      docker/images/livox_bridge/humble-env
      docker/images/ros1_bridge/bridge-entrypoint.sh
    )
    backup=$(mktemp -d)
    for i in "${!scripts[@]}"; do cp "${scripts[$i]}" "$backup/$i"; done
    restore_scripts() {
      for i in "${!scripts[@]}"; do cp "$backup/$i" "${scripts[$i]}"; done
      rm -r -- "$backup"
    }
    trap restore_scripts EXIT
    python3 - <<'PY'
from pathlib import Path
for name in ("docker/images/livox_bridge/build-bridge.sh", "docker/images/livox_bridge/humble-env", "docker/images/ros1_bridge/bridge-entrypoint.sh"):
    path = Path(name)
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
PY
    bridge_source=docker/images/livox_bridge/upstream/ros1_bridge
    if [[ ! -f "$bridge_source/package.xml" ]]; then
      mkdir -p "$(dirname "$bridge_source")"
      git clone --depth 1 --branch 0.9.7 https://github.com/ros2/ros1_bridge.git "$bridge_source"
    fi
    "${compose[@]}" build ros1_bridge framework
    "${compose[@]}" build fast_lio
    ;;
  live)
    : "${ROS_IP:?Set ROS_IP to the workstation robot-network address}"
    export FAST_LIO_SIM_TIME=false
    "${compose[@]}" up -d ros1_bridge framework fast_lio
    ;;
  bag)
    bag="${2:?Give a rosbag2 directory under datasets/real}"
    [[ -d "$bag" ]] || { echo "Bag directory does not exist: $bag" >&2; exit 2; }
    python3 fast_lio_integration/check_bag.py "$bag"
    export FAST_LIO_SIM_TIME=true
    "${compose[@]}" stop ros1_bridge 2>/dev/null || true
    "${compose[@]}" up -d framework fast_lio
    compatible=$("${compose[@]}" exec -T framework python3 /x30/docker/scripts/bag_compat.py "/x30/${bag#./}")
    "${compose[@]}" exec -T -e "BAG_PATH=$compatible" framework bash -lc 'ros2 bag play "$BAG_PATH" --clock --delay 3'
    ;;
  stop) "${compose[@]}" down ;;
  *) echo 'Usage: run.sh build|live|bag datasets/real/BAG|stop' >&2; exit 2 ;;
esac
