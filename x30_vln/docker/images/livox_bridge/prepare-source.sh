#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$HERE/upstream/ros1_bridge"
if [ ! -f "$SOURCE/package.xml" ]; then
    mkdir -p "$HERE/upstream"
    if docker image inspect x30-livox-bridge:latest >/dev/null 2>&1; then
        CONTAINER=$(docker create x30-livox-bridge:latest)
        trap 'docker rm "$CONTAINER" >/dev/null 2>&1 || true' EXIT
        docker cp "$CONTAINER:/tmp/livox_bridge/src/ros1_bridge" "$SOURCE"
    else
        git clone --depth 1 --branch 0.9.7 https://github.com/ros2/ros1_bridge.git "$SOURCE"
    fi
fi
python3 -c 'import sys, xml.etree.ElementTree as ET; p=ET.parse(sys.argv[1]).getroot(); assert p.findtext("name")=="ros1_bridge" and p.findtext("version")=="0.9.7", "Expected ros1_bridge 0.9.7"' "$SOURCE/package.xml"
echo "ros1_bridge 0.9.7 source ready for offline builds: $SOURCE"
