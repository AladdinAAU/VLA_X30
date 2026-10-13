#!/usr/bin/env bash
# Builds only what the X30 sim needs (not the whole quadruped_ros2_control repo).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"$ROOT/scripts/setup_external.sh"
source /opt/ros/jazzy/setup.bash
set -u

# Controller tuning for the X30 (sim-time gait, gait period, hold gain, inertia).
CTRL="$ROOT/external/quadruped_ros2_control"
PATCH="$ROOT/scripts/unitree_guide_x30.patch"
if git -C "$CTRL" apply --check "$PATCH" 2>/dev/null; then
    git -C "$CTRL" apply "$PATCH" && echo "==> Applied controller patch"
elif ! git -C "$CTRL" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "WARNING: controller patch neither applies nor is applied; resolve manually" >&2
fi

colcon build --symlink-install --packages-up-to \
    x30_sim_bringup x30_data_capture leg_pd_controller unitree_guide_controller x30_description \
    keyboard_input autonomous_navigation "$@"
