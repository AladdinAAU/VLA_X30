#!/bin/bash
# Host-side runner (Git Bash / Linux): one throw-away x30-framework container named x30-semmap-test, removed on exit.
#   scripts/run_ros_integration.sh [check args, e.g. --seed 1]
# Build products live in the named volume x30-semmap-ws (remove with: docker volume rm x30-semmap-ws).
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
OUT="${SEMMAP_OUT_DIR:-$ROOT/docs/figures/semantic_map}"       # PNG of the annotated map
OUT_JSON="${SEMMAP_JSON_DIR:-$ROOT/docs/reports}"             # measured numbers
mkdir -p "$OUT" "$OUT_JSON"
MSYS_NO_PATHCONV=1 docker run --rm --name x30-semmap-test \
    -v x30-semmap-ws:/ws -v "$ROOT/ros2_ws/src:/ws/src:ro" -v "$ROOT/configs:/x30/configs:ro" -v "$OUT:/out" -v "$OUT_JSON:/out_json" \
    -e SEMMAP_OUT=/out -e SEMMAP_JSON_OUT=/out_json -e SEMMAP_SKIP_PYTEST="${SEMMAP_SKIP_PYTEST:-}" -e SEMMAP_RATE="${SEMMAP_RATE:-5}" -e X30_CONFIGS=/x30/configs \
    x30-framework:latest bash /ws/src/x30_semantic_map/scripts/container_check.sh "$@"
