#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
source /opt/livox/ros1/setup.bash
source /opt/ros/foxy/setup.bash
source /opt/livox/ros2/local_setup.bash

generated=/tmp/livox_bridge/build/ros1_bridge/generated
previous=$(mktemp -d)
if [ -d "$generated" ]; then cp -a "$generated/." "$previous/"; fi
mkdir -p "$generated"
cd /tmp/livox_bridge/src/ros1_bridge
python3 bin/ros1_bridge_generate_factories --output-path "$generated" --template-dir resource

# Refresh mappings while keeping unchanged converters eligible for the existing C++ build cache.
while IFS= read -r -d '' file; do
    relative=${file#"$generated"/}
    if [ -f "$previous/$relative" ] && cmp -s "$file" "$previous/$relative"; then
        touch -r "$previous/$relative" "$file"
    fi
done < <(find "$generated" -type f -print0)
cd /tmp/livox_bridge
MAKEFLAGS=-j${BUILD_JOBS:-2} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS:-2} colcon build \
    --merge-install --install-base /opt/livox/bridge --packages-select ros1_bridge --cmake-force-configure \
    --event-handlers console_direct+ --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
