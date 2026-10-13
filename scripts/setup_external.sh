#!/usr/bin/env bash
# Fetch the two upstream repos at the commits this project was built/tested against.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fetch() {  # url dir commit
    [ -d "$ROOT/external/$2/.git" ] || git clone "$1" "$ROOT/external/$2"
    git -C "$ROOT/external/$2" checkout -q "$3"
}
mkdir -p "$ROOT/external"
fetch https://github.com/legubiao/quadruped_ros2_control.git quadruped_ros2_control 5434c5810d1a7fe223bcfd04550e9d3bfdd4b458
fetch https://github.com/mil-as/quadruped_navigation_ros2.git quadruped_navigation_ros2 3c12fabf51614695ea763f8015e51b3b3d75c822
