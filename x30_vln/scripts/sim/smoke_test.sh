#!/usr/bin/env bash
# Boot the simulator headless, check the whole robot contract, drive 3 m, save images / scan, measure RTF and CPU/RAM.
#   [WORLD=<id>] scripts/sim/smoke_test.sh [kinematic|legged|both] [measure_seconds]
# Runs on the host (Git Bash / Linux: starts the x30-sim container if needed and runs itself inside it) or directly in the container.
# Results: results/sim_smoke/<mode>/ (rgb.png, thermal.png, lidar.pcd, lidar_bev.png, report.json, rtf.csv, launch.log). Exit code != 0 on failure.
set +u
modes=${1:-kinematic}; secs=${2:-120}
if ! command -v ros2 >/dev/null 2>&1; then
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    docker compose -f "$root/docker/compose.sim.yml" --profile sim up -d sim >/dev/null 2>&1 || { echo "cannot start the sim container"; exit 2; }
    MSYS_NO_PATHCONV=1 docker exec -e ROS_DOMAIN_ID="${SIM_DOMAIN_ID:-10}" x30-sim bash -lc "/x30/scripts/sim/smoke_test.sh $modes $secs"
    exit $?
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f /ws/install/setup.bash ] || x30-build >/dev/null 2>&1
. /usr/local/bin/x30-env
[ "$modes" = both ] && modes="kinematic legged"
rc=0
for mode in $modes; do
    out=/x30/results/sim_smoke/$mode
    rm -rf "$out"; mkdir -p "$out"
    bash "$here/kill_sim.sh" >/dev/null 2>&1
    echo "=== $mode ==="
    nohup ros2 launch x30_sim sim.launch.py world:=${WORLD:-default_arena} locomotion:=$mode > "$out/launch.log" 2>&1 &
    python3 "$here/smoke_check.py" --out "$out" --locomotion "$mode" | tee "$out/check.log" | grep -E "PASS|FAIL|checks passed"
    [ "${PIPESTATUS[0]}" -ne 0 ] && rc=1
    echo "--- RTF / CPU / RAM over $secs s ---"
    python3 "$here/measure_rtf.py" --seconds "$secs" --interval 10 --csv "$out/rtf.csv" | tail -1
    bash "$here/kill_sim.sh" >/dev/null 2>&1
done
exit $rc
