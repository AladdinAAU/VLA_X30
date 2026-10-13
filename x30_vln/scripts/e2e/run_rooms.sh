#!/usr/bin/env bash
# Room-goal batch for aws_hospital: "Go to Room X." from the spawn (yaw 0), oracle object map + named rooms, policy locate_nav (model reports WHERE, planner drives).
#   bash scripts/e2e/run_rooms.sh [tag] [rooms...]     default tag r1, rooms A B C D E F G.   Summary: results/e2e/rooms_<tag>.jsonl
# Each episode ~2-5 min wall at RTF 0.4. Requires the stack from run_e2e.sh aws_hospital (see docs/END_TO_END.md).
export MSYS_NO_PATHCONV=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:-r1}"; shift
ROOMS=("$@"); [ ${#ROOMS[@]} -eq 0 ] && ROOMS=(A B C D E F G)
OUT="$ROOT/results/e2e/rooms_${TAG}.jsonl"; : > "$OUT"
for R in "${ROOMS[@]}"; do
  N="room_${R}_${TAG}"
  docker exec e2e-fw bash -lc "source /opt/ros/humble/setup.bash; source /ws/install/setup.bash; export ROS_DOMAIN_ID=80; python3 /x30/scripts/e2e/run_episode.py --name $N --world aws_hospital --instruction 'Go to Room $R.' --start 0.93 11.48 0.0 --load-map /x30/results/maps/aws_hospital.gt.json --rooms /x30/configs/rooms/aws_hospital.yaml --target-room 'Room $R' --policy-name locate_nav --timeout 900 > /x30/results/e2e/logs/$N.log 2>&1"
  python -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))))" "$ROOT/results/e2e/$N/result.json" >> "$OUT" 2>/dev/null || echo "{\"name\": \"$N\", \"error\": \"no result.json\"}" >> "$OUT"
done
echo done >> "$OUT"
