# Handoff: X30 VLN framework (state at 2026-10-11)

> **STATUS AT A GLANCE (2026-10-12, end of the long session):** sections 10-14 at the end of this file are the newest and supersede sections 3, 4, 6 and 9 where they differ. Built in the simulator: instruction -> place -> planner (`locate_nav`), `follow_person` with lock-on / appearance / ambiguity / crowd / twin tests, the learner phase (`x30_teach`, taught places), rosbag demo tooling. Unfinished: planner-assisted following around corners (section 14, fair A/B still to run). The ready-to-paste prompt for a new chat is `docs/NEW_CHAT_PROMPT.md`.

Read this first in a new chat, then `docs/ARCHITECTURE.md`, `docs/INTERFACES.md` (binding contract) and `docs/END_TO_END.md`.
Project root: `C:\Users\LAB USER 1\Desktop\x30\x30_vln` (Windows 11, Docker Desktop/WSL2, RTX 3090 Ti 24 GB, 24 logical CPUs, WSL capped at 24 GB, swap off).
User: works with the DEEP Robotics X30 quadruped; wants a modular vision-language navigation framework, trials in simulation first, then the real robot, same nodes and topics for both.

## 0. IMPORTANT CORRECTION (2026-10-11): what 'velocity_follow' meant
The user's two modes were: (1) FOLLOWING A PERSON using velocity commands, and (2) object navigation. We misread (1) and built language-instruction following (NaVILA style, VLM emits 'move forward 50 cm / turn left 30 degrees', executed as velocity primitives) under the name `velocity_follow`. The user confirmed that mode is ALSO wanted. So: keep it (rename later to `instruction_follow`), and ADD the missing `follow_person` mode (section 9).

## 1. What the user asked for
Framework with SLAM, segmentation, a semantic map built from fused sensors (4 Livox lidars, RGB camera, optional thermal), a reasoning VLM, planners, and two switchable modes: `velocity_follow` (instruction following by velocity primitives) and `object_nav` (find an object via semantic map plus navigation). Docker based, structured, simulation resembling the real robot, datasets of real worlds with ground truth, fine-tuning/trials harness before real deployment. Later asks: simple "go to waypoint" navigation, known-map navigation, screenshots/gallery and documentation of progress.

## 2. Decisions already made (do not re-litigate without a reason)
- Framework container = ROS 2 Humble (matches the user's earlier Rob7 repo: foxy ros1_bridge image + Humble). Simulator container = ROS 2 Jazzy + Gazebo Harmonic (the user's earlier controller prework). They talk over CycloneDDS with standard messages only.
- Real X30 has NO depth camera: object 3D positions come from projecting `/lidar_points` into the RGB image. Thermal camera is NOT stock on the X30: every thermal consumer is optional.
- The user's earlier bridge (Rob7 compose: `ros:foxy-ros1-bridge`, master 192.168.1.106, ROS_IP 192.168.1.131; `nav_bridge` UDP node from ggbond-control) is the base for the real-robot path; `x30-ros1-bridge` image reproduces it and was tested on the recorded bag.
- Software rendering (llvmpipe) is the default for Gazebo under WSL (GPU path leaks ~1.2 MB/s).
- Real-scan worlds (Replica, ReplicaCAD, Habitat test scenes) are the priority worlds; procedural worlds are only for fast tests. The user approved only those three downloads; HM3D/MP3D/ScanNet need the user's own licence acceptance (importers exist).

## 3. What exists (all under `ros2_ws/src`, configs in `configs/`, docs in `docs/`)
| Area | Package / path | State (measured, from the reports in `docs/reports/`) |
|---|---|---|
| Interfaces | `x30_msgs` | done |
| Docker | `docker/` (compose.yml, compose.models.yml, compose.real.yml, compose.sim.yml, images framework/perception/ros1_bridge/sim, dds/*.xml), `scripts/x30.ps1` | done. Humble<->Jazzy DDS no drops; ROS1 bag through bridge 177/178 msgs at 10 Hz; VLM 0.35 s/reply |
| Simulator | `x30_sim`, `docker/images/sim` | 29/29 smoke checks in kinematic and legged; RTF 0.74-0.99 (light worlds), 0.3-0.4 on heavy scan worlds; lidar matches the real bag's distribution roughly (sim has more valid returns: 0.60 vs 0.47) |
| Worlds | `datasets/`, `x30_worlds` | 13 real-scan worlds (7 Replica, 4 ReplicaCAD, 2 Habitat test) plus 10 procedural plus 4 AWS; ground truth objects, episodes (object_nav, goal_pose, velocity_follow) generated |
| Perception | `x30_perception` (+ server) | works on synthetic data; detector finds only ~23% of visible objects in real-scan worlds (YOLOE) |
| Semantic map | `x30_semantic_map` | 31 ROS checks, 58 tests; recall 0.90 on synthetic scenario; VLM label check is a weak veto (8/15) |
| Odometry/mapping/Nav2 | `x30_navigation` | lidar odometry good in stub (0.13 m RMSE/105 m); 2D map precision 0.94/recall 0.79 in sim; Nav2 goals unreliable in sim (0/20 lab_01, 2/10 replica apartment) |
| Known-map navigation | `x30_navigation/known_map*.py`, `docs/NAVIGATION_KNOWN_MAP.md` | 12/12 waypoints in lab_01 and in replica_frl_apartment_0, 0 collisions (kinematic, static map, GT-aligned odometry) |
| Modes/safety/skills | `x30_modes`, `configs/skills.yaml`, `configs/modes/` | 80 tests, 79 fake-robot checks, 14 sim checks; guard margin shrinks at 0.6 m/s |
| Reasoner/follower | `x30_agent`, `x30_common`, `configs/models.yaml`, `configs/agent/` | fake world: object_nav 20/20; velocity_follow 6/20 fresh seeds (plain Qwen3-VL-8B is weak at metric language actions) |
| End-to-end | `docs/END_TO_END.md`, `scripts/e2e/` | first run: 3 object_nav episodes, 0/3 successes under a centre-distance criterion (causes below) |
| Gallery | `scripts/make_gallery.py` -> `docs/PROGRESS.html` | 153 images |

## 4. NOT finished
- **`x30_eval`** (runner, metrics, sweeps, fine-tune export): code written (23 files), progress in `docs/reports/eval-progress.md`; never finished or verified, no final report.
- **`x30_hw` and `x30_nav_bridge`** (real robot: patched nav_bridge fork, `fake_x30` UDP fake robot, real_adapter, camera ingest, replay launch, runbook): partly written, no final report, nothing verified. Must be built on the user's Rob7 compose and the ggbond nav_bridge; patches needed are in `docs/robot/x30-interfaces.md` section 4.3 (speed table wrong vs API appendix, 5 s stale cmd, heartbeat/ownership, `network.toml` registration of the dev PC, do not bridge `/cmd_vel` into ROS1).
- Nothing has run on the real robot. Never contact 192.168.x.x from tests; the user runs real checks (`scripts/check_real_network.ps1`, runbook).
- Optional lab world from the user's real bag (`datasets/real/lab_2026-10-06`, ROS 2 copy of `2026-10-06-17-05-52.bag`): not started.

## 5. Findings and known problems (honest list)
1. End-to-end episodes (see `docs/END_TO_END.md`): sofa 2.19 m from centre (sofa is 2.2 m long, agent stopped 1.3 m from its surface: scoring problem); chair failed from detector false positives and `search` giving up early; tv found within 0.5 m but robot wedged, guard blocked rotations, `goto_object` found no free standoff cell.
2. Detector recall is the weakest link (23%); real textures removed garbage labels but doors, windows, tables mostly missed. An oracle-detector ablation was NOT built.
3. Known-map mode publishes no `/costmap_cloud`; the guard reads `/lidar_points` through `configs/modes/e2e.yaml`. `next_viewpoint` node (replaces frontiers on a known map) is not unit-tested.
4. Nav2 in sim: costmap stall root-caused to the LaserScan observation source and fixed; sim map corruption in room areas still open; sim lidar-odometry ATE not measured; odometry yaw vs ground truth once off by 1.5 rad (unexplained).
5. Camera/thermal/IMU noise, latency, accel limits, lidar extrinsic angles (25/46/5/56 deg) are assumptions until measured on the robot. Camera-lidar extrinsic must be calibrated to about 2 deg (perception report).
6. ReplicaCAD worlds use ~7 GB (peak 14 GB) Gazebo memory, cause unknown. Concurrent test runs need distinct `ROS_DOMAIN_ID` (and GZ_PARTITION).
7. Licences: Replica, ReplicaCAD (LICENSE.txt says CC BY-NC 4.0, HF card says CC BY 4.0), Habitat test scenes: treat as non-commercial research.

## 6. Recommended next steps, in order (NEW: person following, section 9, is a top priority alongside these)
1. Fix `goto_object` standoff (use the uninflated grid / nearest reachable cell), publish a self-filtered `/costmap_cloud` in known-map mode, let `search` continue after a rejected candidate.
2. Surface-distance success criterion (distance to the object's box surface) in `x30_eval`; finish `x30_eval`.
3. Build the oracle detector (publish `Detection3DArray` from `/gt/objects` in FOV) to separate detector errors from reasoning errors; then >= 10 episodes per world, dev/heldout split, repeats.
4. Improve detection: per-task vocabulary, thresholds, maybe a stronger open-vocabulary detector; consider a Habitat-sim track for fast VLM-side evaluation (not started; discussed only).
5. Instruction following: plain Qwen3-VL-8B is weak. Options: larger/other VLM, fine-tune on logged sim trajectories (exporter planned in `x30_eval`), Qwen-RobotNav weights if published (check licence).
6. Finish `x30_hw`/`x30_nav_bridge` against the fake robot, then the runbook-driven staged real-robot bring-up by the user.

## 7. How to work (lessons that cost tokens)
- The user hit the session limit repeatedly. Do NOT run many agents at once (at most 2 at a time), do NOT resume long-running agents (each step re-sends the whole transcript: one resumed agent burned ~400k tokens in a minute); start FRESH agents with a short handoff and a hard budget (e.g. "about 40 tool calls"). Keep tool outputs short.
- Agents write progress notes to `docs/reports/<area>-progress.md`; finished work is in `docs/reports/<area>.md`. Cross-agent requests live in `docs/requests/`.
- Agent rules are in `docs/AGENT-BRIEF.md` (own directories only, run what you claim, no contact with 192.168.x.x, build lock `.locks/docker-build`, one background command at a time).
- Docker: start only via `powershell -ExecutionPolicy Bypass -File "C:\Users\LAB USER 1\Documents\ros2-docker-setup\docker-clean-start.ps1"`. Use 127.0.0.1, not `localhost`, for the model servers (localhost adds ~2 s per request on this host). Shell scripts written from Python get CRLF and fail in containers. In Git Bash prefix `docker exec` with `MSYS_NO_PATHCONV=1` for `/x30/...` args.
- The user wants progress visible: rebuild `docs/PROGRESS.html` (`python scripts/make_gallery.py`) and send it after milestones; agents save screenshots to `docs/figures/<area>/` with `CAPTIONS.md`.
- Report honestly: say what was measured, what was not, no claims without numbers.

## 8. Quick start (simulator, end to end)
See `docs/END_TO_END.md` (launch sequence, about 35 s bring-up) and `scripts/e2e/run_e2e.sh`. Containers: `x30-sim` (Jazzy), `x30-framework` (Humble), `x30-vlm` (Qwen3-VL-8B, 8000), `x30-perception` (8100). Operator manual: `docs/INFRA.md`. Worlds: `docs/WORLDS.md`.

## 9. Design notes for the missing `follow_person` mode (not built)
- Behaviour: acquire a target person (default: nearest person in front within 4 m; or described in words and chosen by the VLM from crops), then keep a set distance (suggest 2.0 m; the X30 manual p.36 says people must stay >= 2 m from the robot during autonomy, so never close below that without the user changing it) and heading, using velocity commands on `/cmd_vel_nav` (vx from distance error, wz from bearing, optional vy), through the existing mux and safety guard.
- Perception: 'person' is a class the detector handles well; track IDs over time (ByteTrack-style, or reuse the semantic map's dynamic-object handling); person 3D position from lidar projection (`x30_perception`); thermal can help only if a thermal camera exists (not stock). Re-identification after occlusion by appearance crop (VLM or embedding).
- Motion: follow the person's breadcrumb trail or a local planner target, avoid obstacles with the lidar costmap; do not cut through walls; slow or stop when the person stops; lost-target recovery: go to last seen position, rotate/scan, ask the VLM to re-find, give up and stop after a timeout.
- Simulation: Replica/ReplicaCAD worlds are static: add a moving person actor (Gazebo actor with a scripted trajectory, or a model driven by a script, with a warm `<temperature>` for thermal) and ground truth pose for it; metrics: fraction of time tracked, mean distance error vs target distance, lost events, collisions, minimum separation.
- Integration: new node `follow_person` (package `x30_modes` or a new `x30_follow`), skill `follow_person` in `configs/skills.yaml`, mode in `mode_manager`, episode type `follow_person` in the episode schema. Papers digest has little on this (CausalNav's dynamic-object tracking is the closest).

## 10. Update 2026-10-11 (session 2): instruction -> place -> planner, aws_hospital rooms
Full description, commands, measurements and caveats: `docs/INSTRUCTION_TO_GOAL.md`. Summary:
* New policy `locate_nav` (configs/agent): the reasoner only reports WHERE (`report_target` object id or x,y), the harness hands it to the planner in one skill call. Rooms are operator-defined (`configs/rooms/aws_hospital.yaml`, `/semantic_map/set_rooms`) and shown with coordinates in the map summary. Oracle object map from `scripts/e2e/gt_to_semmap.py`. Result: 6/7 rooms reached in aws_hospital (n = 1 each); room F unreachable in the 2D map.
* Fixed: known_map_nav cancel bug (goals leaked, `nav2_no_response`), unreachable goals moved to the nearest reachable cell, goto_pose snaps goals inside obstacles, sim guard footprint/margins relaxed (e2e profile only), harness accepts planner arrival at a moved goal.
* Findings: online object_nav in aws_hospital (find the bed, 2 runs) failed 0/2 because search did not reach the true beds and the detector produced false beds (railing, reception desk); verify rejected them correctly. Replica FRL apartment 2D map is tiny (35.6 m2 free): mapping tours there fail (wedging on furniture, 8/62 objects). Doorways of 0.9 m + circular planner body + rectangular guard footprint = wedging at doors; doorway-aware planning and a guard escape rule are NOT done.
* The decision to keep: planner (known_map_nav now, Nav2 on the robot) owns routing; the model owns the target. Next candidates: object targets with locate_nav on a detector-built map in aws_hospital/lab_01, more starts and repeats (n >= 10), doorway-aware planning, fixing room F (or choosing reachable rooms), real-robot guard tuning.

## 11. Update 2026-10-11 (session 2, later): follow_person built (simulator)
Full description, launch sequence, metrics and limits: `docs/FOLLOW_PERSON.md`. New package `x30_follow` (ROS-free `tracker.py` + `controller.py` with 14 tests, node `follow_person`, action `/x30/follow_person/run`, `/follow/status`), mode `follow_person` in x30_modes (mode_logic, mux_logic, mode_manager `person_action`), scripted walking person `x30_sim/person_mover.py` (moves the model `person_07` of aws_hospital with `set_pose`, `/gt/person`), path generator `scripts/follow/make_person_path.py`, runner/scoring `scripts/follow/run_follow_episode.py`, rosbag demo `scripts/follow/record_follow.sh` + `follow_demo.py`.
Measured (aws_hospital, one 35.5 m scripted walk): person 0.5 m/s, robot cap 0.6: tracked 100 %, mean distance error 0.46 m (steady 2.94 m for a 2.5 m target: 0.16 m surface-centroid bias + proportional lag), min separation 2.52 m (floor 2.0 m never violated), no lost events; person 0.8 m/s with a 0.5 m/s cap: robot falls behind (experiment design error). NOT done: lost-target recovery exercised (no occlusion test), planner-assisted following around corners, several people, description-based target choice, other worlds, real robot.
The same session built `docs/INSTRUCTION_TO_GOAL.md` (section 10 above) and the rosbag demo tooling (`scripts/e2e/record_episode.sh`, `bag_export.py`, `bag_demo.py`).

## 12. Update 2026-10-12: lock-on, appearance, crowd (simulator)
`docs/FOLLOW_PERSON.md` sections 8-10. `/follow/lock` service (operator command while the person is in front), colour appearance signature (crop histograms), ambiguity rule (hold, never guess), outfits (`scripts/follow/make_outfits.py`), crowd mover (`configs/follow/crowd1.json`, `/gt/people`), identity metrics (wrong_person_frac, identity_switches, min_sep_other_robot_closing_m). Crowd run crowd_rec3: 100 % tracked, 0 identity switches, 2.53 m behind the target, 5 distinct outfits; the contact criterion was revised after the fact (documented). The ambiguity rule and the look-alike case are unit-tested only. The next planned phase is the 'learner' tour: follow a person who names places ('this is room G', 'we deliver here') and store them as named regions/points in the semantic map (design agreed in chat, not started; input typed/scripted first).

Update 2026-10-12 (later): twin/look-alike test and a person_mover respawn bug (models silently missing; fixed by scene verification) are in docs/FOLLOW_PERSON.md section 12. crowd_4 and twin_rec2 were void. Lock: segment position model + adaptive ambiguity margin (24 tests). twin_rec4: lock stayed within 0.32 m of the target, 2 holds; preset criteria not met (documented as metric/scenario artifacts, criteria not changed).

## 13. Update 2026-10-12: learner phase (simulator)
`docs/TEACH_PLACES.md`. New package `x30_teach` (`/teach/say`, LLM interpreter with 4 tools, reply, log), taught places in the semantic mapper (`places.py`, services, summary block, saved with the map, line-of-sight room rule), scripted guided tour (`scripts/teach/run_tour.py`, nurse leads, robot follows with the lock, sentences sent when she stands in the place and the robot tracks her), use of the taught map (`scripts/teach/run_use.sh`), rosbag tour demo. Measured: model alone 21/22 (prompt tuned on them) and 10/10 on 10 fresh sentences; tour_2: 3/3 places in the right ground-truth room with the right kind/owner/function; "Go to Marcus", "Go to Alexander's room", "Take this package to the delivery point": 3/3 arrived. Limits: text only (no ASR), one tour, the robot was 7.6 m behind the person at the first sentence (anchor = person position), room extent = free-space region (7 m2 / 3.9 m2 for rooms of about 24 m2). Tests: x30_semantic_map 63, x30_teach 3, x30_follow 24.

## 14. Update 2026-10-12 (end of session): planner-assisted following, UNFINISHED
Problem reported by the user: when the person takes a fast corner the robot sees the straight line to her as blocked and stops instead of going around. Built: `x30_follow/trail.py` (breadcrumb trail + grid line of sight, 3 tests), node mode PLANNED (when the line is blocked it sends `navigate_to_pose` goals on her trail d_target behind her; lost: goal d_target short of her last seen place; below the floor it backs off), switch `planner_follow` (params_json false = old behaviour). 27 follow tests pass. NOT yet measured fairly:
* Tour scenario (nurse into Room C and back out the same 0.9 m door): old behaviour: robot avg 4.2 m behind, max 7.6 m, 57 % of the time > 3.5 m. New: avg 3.2 m, max 5.8 m, but the nurse walked through the robot at the door (0.05 m, scripted head-on) and the run ended 'lost the person' at 57 s (not analysed with the lock diagnostics).
* Corner scenario (`configs/follow/corner1.json`, `scripts/follow/run_corner_ab.sh`): the A/B result (corner2_base / corner2_plan) is INVALID: leftover `crowd_*` people of earlier scenarios were still standing in the world (the mover only managed models of the current file). Fixed in `person_mover.remove_leftovers` (verified: only person_07 in the scene). Earlier twin/tour/crowd results after the first crowd run may have had such static extra people around: tracking results were not obviously affected but this was uncontrolled.
* NEXT: re-run `bash scripts/follow/run_corner_ab.sh` (about 12 min wall) with the clean world, compare distance statistics (max distance, share of time > 3.5 m / > 5 m, lost events), analyse any 'lost' with the status `why` diagnostics (record_follow.sh bag), then document in docs/FOLLOW_PERSON.md.
