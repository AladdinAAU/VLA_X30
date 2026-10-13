#!/usr/bin/env bash
# Stop and remove everything run_e2e.sh started. Volumes and images stay. Pass "keep-models" to leave the VLM/perception running.
export MSYS_NO_PATHCONV=1
docker rm -f e2e-fw e2e-sim >/dev/null 2>&1
[ "${1:-}" = keep-models ] || docker stop x30-vlm x30-perception >/dev/null 2>&1
docker ps --format '{{.Names}} {{.Status}}'
