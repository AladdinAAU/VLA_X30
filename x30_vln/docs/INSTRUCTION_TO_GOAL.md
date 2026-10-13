# Instruction -> place -> planner (the `locate_nav` pipeline)

Status 2026-10-11. Everything below was run in the simulator (world `aws_hospital`, kinematic locomotion, software rendering, RTF about 0.4). The numbers are from single runs (n = 1 per room, one start pose, one model seed): they show that the pipeline works, they are not rates.

## 1. Division of labour
```
"Go to Room C."  ──▶ reasoner (Qwen3-VL-8B, policy configs/agent/locate_nav.yaml)
                        reads: semantic-map summary (objects with ids and x,y; Rooms list with each centre as at (x,y))
                        decides ONLY where: report_target(object_id=..)  or  report_target(x=.., y=..)   [or found=false]
                     ──▶ harness (core/reasoner.py _report_target): ONE skill call, then the episode ends
                        object_id -> goto_object(standoff 1.0 m)      x,y -> goto_pose(tolerance 1.0 m)
                     ──▶ planner behind the navigate_to_pose action (known_map_nav in the sim; Nav2 or another planner on the robot)
                     ──▶ cmd_vel_nav -> cmd_vel_mux -> safety_guard -> /cmd_vel
```
The model never sees the camera, never issues motion, is not asked again after `report_target`. It may call `query_map`, `search`, `explore`, `scan` first (to find or map the target). The older multi-turn policy (`object_nav`: search, goto_object, approach, verify, finish) is still available (`policy_name` default).

## 2. Where the semantic map comes from
* **Online** (`object_nav` policy and the first e2e runs): built while the robot explores (`search`/`explore`), YOLOE detections lifted with the lidar. Weak on CAD worlds (false 'bed' objects: railings, a reception desk); see `END_TO_END.md` (g).
* **Oracle map** (used for the room experiment): `python scripts/e2e/gt_to_semmap.py --world aws_hospital` writes `results/maps/aws_hospital.gt.json` (every ground-truth object as a confirmed map object, world frame). `run_episode.py --load-map` converts it to the episode's map frame (`known_map_nav`'s map frame is the start pose: map = R(-yaw)(world - start), `scripts/e2e/mapframe.py`).
* **Rooms**: the semantic mapper can take operator-defined rooms (`/semantic_map/set_rooms`, rectangles in the map frame, name kept, replaces the automatic segmentation) and lists them in the summary the reasoner sees. Definitions: `configs/rooms/aws_hospital.yaml` (7 rooms A-G of the central column, drawn on `docs/figures/rooms/aws_hospital_rooms.png`). The automatic segmentation of the hospital is poor (36 regions, the corridor swallows the rest; `docs/figures/rooms/aws_hospital_regions.png`).

## 3. How to run
Stack: `bash scripts/e2e/run_e2e.sh aws_hospital` (see `END_TO_END.md` (b)). Then one room: 
```
python3 /x30/scripts/e2e/run_episode.py --name room_C --world aws_hospital --instruction 'Go to Room C.' --start 0.93 11.48 0.0 \
  --load-map /x30/results/maps/aws_hospital.gt.json --rooms /x30/configs/rooms/aws_hospital.yaml --target-room 'Room C' --policy-name locate_nav
```
(inside `e2e-fw`, `ROS_DOMAIN_ID=80`; `--start` yaw must be 0 with `--rooms`). All rooms: `bash scripts/e2e/run_rooms.sh <tag> [A B ...]` (the jsonl summary writer in that script fails on Windows paths: read `results/e2e/room_<R>_<tag>/result.json`). Replay GIF/MP4 of an episode: `python scripts/e2e/replay_episode.py results/e2e/<name> --world aws_hospital --rooms configs/rooms/aws_hospital.yaml --target-room 'Room C' --goal 1.7 -3.25`. If a killed client leaves a skill running every new skill answers 'Busy': `python3 /x30/scripts/e2e/stop_skill.py`.
Success = stop reason `finished` AND the final `/gt/pose` inside the room rectangle (world frame).

## 4. Measured (aws_hospital, from the spawn 0.93/11.48, oracle object map, 7 rooms)
| room | result | path | sim time | note |
|---|---|---|---|---|
| A | OK (r4) | 13.4 m | 44 s | centre is inside the nurse-station desk: goal moved to the nearest free spot |
| B | OK (r1, r3) | 16.6 m | 71 s | |
| C | OK (r1, r3) | 15.6 m | 60 s | |
| D | OK (r3) | 24.4 m | 110 s | centre inside furniture, moved 0.6 m |
| E | OK (r3) | 23.6 m | 100 s | same |
| F | **FAIL** (r3, r4) | 33.9 m | 124 s | ends 0.66 m outside: only about 2.5 m2 of the room is reachable in the planner's map even at the loosest clearance, the nearest reachable cell lies in the corridor beyond the wall |
| G | OK (r1, r3) | 31.6 m | 133 s | |

6/7 with the final code. Caveats: the rooms were not all run in one batch with the final code (r3: B-E, G; r4: A, F; the last change only affects how an arrival is reported, not the motion); n = 1 each; the reasoner needed 2 turns (3 for G in r1) because it sometimes first calls `report_target` without x and y (the harness answers with an error and it retries); the object map is ground truth, so this measures instruction -> place -> navigation, not detection.

## 5. What had to be fixed to get here (all in the repo)
1. `known_map_nav` had no cancel callback: cancelled goals kept driving and filled the 4 executor threads, new goals got no answer (`nav2_no_response`). Now cancels are accepted and a new goal pre-empts the old one.
2. Planner: an unreachable goal (inside furniture or an enclosed pocket) is moved to the nearest reachable cell within 3 m (`KnownMap.nearest_reachable`, parameter `goal_snap_m`); `goto_pose` first moves a goal that lies inside an obstacle to the nearest free spot within 2 m and says so in its message.
3. Safety guard (sim/e2e profile only, `configs/modes/e2e.yaml`): footprint front/rear 0.52, half-width 0.26 (was 0.55/0.32), lateral margin 0.02, stop margin 0.12, hard stop 0.03 (were 0.04/0.20/0.06). The guard held the robot with every motion blocked at doorways 0.9 m wide. The real-robot config is unchanged and must be tuned on the robot.
4. The harness accepts the planner's arrival at its own (moved) goal, because the skill compared the robot with the original point.
5. Summary now shows room centres as coordinates, `explore` accepts both frontier answer formats, stale `frontier_explorer` nodes removed by restarting the containers.

## 6. Known problems and limits
* Doorways about 0.9 m wide: the planner treats the body as a circle (tiers 0.55/0.42/0.32 m) and the smoothed path may cross a door diagonally; the guard's footprint is a rectangle. With the relaxed guard Room C was reached, but the planner is not doorway-aware (no door-axis alignment). A guard 'escape' rule (allow motions that increase clearance when held) is not implemented.
* Rooms whose interior is blocked on the 2D map (F) cannot be entered; the experiment should use rooms that are reachable, or the map should be fixed.
* Not tested: object targets with this policy (the multi-turn `object_nav` runs used the detector-built map), rooms from other starts, other worlds, repeated runs, real-robot guard parameters, replacing `known_map_nav` with Nav2 (the action interface is the same, Nav2 was unreliable in the simulator: 0/20 and 2/10 earlier).

## 7. Recording a demo from a rosbag
`bash scripts/e2e/record_episode.sh C rec1` records `results/bags/room_C_rec1` (zstd, about 42 MB per 180 s; camera, camera_info, /gt/pose, /tf, /tf_static, /odom, /cmd_vel, /cmd_vel_nav, /x30/safety, /x30/mode, /map, /nav/known_map_info, /clock; no lidar cloud) while running the episode. Then, inside `e2e-fw`: `python3 /x30/scripts/e2e/bag_export.py /x30/results/bags/room_C_rec1 /x30/results/bags/room_C_rec1_export` (needs `SequentialCompressionReader`: the bag is compressed), and on the host `python scripts/e2e/bag_demo.py results/bags/room_C_rec1_export results/e2e/room_C_rec1 --world aws_hospital --rooms configs/rooms/aws_hospital.yaml --target-room "Room C"` -> `demo.mp4` (sim-time paced, 10 fps). The earlier `replay_episode.py` GIFs show the camera only at agent turns; use the bag demo for anything shown to others. Measured for Room C (rec1): inside the room, 15.6 m, 59 s sim, 148 s wall.
