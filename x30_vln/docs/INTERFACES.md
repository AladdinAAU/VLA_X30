# Interface contract

Binding for all packages. Change it only by editing this file AND telling the integrator (the main session). Version 0.1.

## 1. Frames (REP-105)
| Frame | Meaning |
|---|---|
| `map` | global, from SLAM; may be rotated vs the simulator world (evaluation aligns by yaw/SE2) |
| `odom` | smooth local frame |
| `base_link` | X30 body centre (the torso link; static identity to prework `TORSO`). x forward, y left, z up |
| `lidar_link` | frame of the **merged** Livox cloud (this id is what the real robot/bag uses) |
| `imu_link` | body IMU |
| `camera_link`, `camera_optical_frame` | RGB camera (optical: z forward, x right, y down) |
| `thermal_link`, `thermal_optical_frame` | thermal camera |
| `lidar_fl_link`, `lidar_fr_link`, `lidar_rl_link`, `lidar_rr_link` | the four individual Livox sensors (front-left, front-right, rear-left, rear-right) |

TF `odom -> base_link` is published by exactly one node (the active odometry source, section 3). `map -> odom` by SLAM (or a static identity in `slam: none`). Static sensor transforms come from the robot description (sim) or `x30_hw` (real) using the same numbers from `configs/robot_*.yaml`. Sim time: all framework nodes use `use_sim_time` in `sim` profile, wall time in `real`/`replay` unless the bag player publishes `/clock`.

## 2. Robot-facing topics (sim, real and replay all provide these)
| Topic | Type | Rate | Notes |
|---|---|---|---|
| `/lidar_points` | `sensor_msgs/PointCloud2` | 10 Hz | frame `lidar_link`, merged 4 lidars, ~80k pts. Fields in order: `x,y,z,intensity` float32 (offsets 0,4,8,12), `tag` uint8 (16), `line` uint8 (17), `timestamp` float64 (18), `point_step` 26, `is_dense` true. Identical in sim, real, bag |
| `/imu/data` | `sensor_msgs/Imu` | >=100 Hz | frame `imu_link`; orientation filled if available |
| `/odom` | `nav_msgs/Odometry` | >=30 Hz | leg/proprioceptive odometry (real) or ground-truth-with-optional-noise (sim); frame `odom` -> `base_link` |
| `/camera/color/image_raw` | `sensor_msgs/Image` rgb8 | 10-15 Hz | frame `camera_optical_frame` |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | same | real intrinsics when calibrated |
| `/thermal/image_raw` | `sensor_msgs/Image` mono8 | 5-10 Hz | frame `thermal_optical_frame`, normalised display image (hot = bright) |
| `/thermal/temperature` | `sensor_msgs/Image` 32FC1 | same | degrees Celsius per pixel; only when radiometric data exists, else absent |
| `/thermal/camera_info` | `sensor_msgs/CameraInfo` | same | |
| `/camera/depth/image_raw` | `sensor_msgs/Image` 32FC1 (m) | 10-15 Hz | **SIM ONLY, ground-truth aid; the stack must not depend on it** (`depth_source: sim_depth` is for ablations) |
| `/cmd_vel` | `geometry_msgs/Twist` | 20 Hz | **input** to locomotion (sim controller or nav_bridge). Produced only by the safety guard |
| `/x30/ready` | `std_msgs/Bool` (transient local) | | true when the robot can be commanded (standing, trotting/walking) |
| `/x30/battery` | `sensor_msgs/BatteryState` | 1 Hz | |
| `/x30/estop` | `std_msgs/Bool` | | input: true = soft e-stop requested |
| `/x30/health` | `diagnostic_msgs/DiagnosticArray` | 1 Hz | one status per subsystem |
| `/clock` | `rosgraph_msgs/Clock` | | sim (and bag replay with `--clock`) |

Sim-only ground truth (never used by the stack except through `x30_eval`): `/gt/pose` (`geometry_msgs/PoseStamped`, frame `world`), `/gt/person` (PoseStamped, `world`, the scripted walking person of `x30_sim/person_mover.py`, controlled by `/sim/person/start|reset`), `/gt/objects` latched `x30_msgs/SemanticObjectArray` with true object poses and names, `/gt/scene_info` latched `std_msgs/String` JSON (scene name, bounds, landmark list).

Real-robot facts that bind everyone (from `docs/robot/x30-interfaces.md`): there is **one wide-angle RGB camera** (USB on the perception host; intrinsics, FOV, mount pose NOT documented, must be calibrated; a field-found stream is `rtsp://192.168.1.105:8554/test`); a **thermal camera is not part of the stock X30** (only an optional third-party payload) so every thermal consumer must work when `/thermal/*` is absent; the robot already publishes ROS1 `/leg_odom` (nav_msgs/Odometry, 2D, ~200 Hz), `/imu/data` (200 Hz), `/lidar_points`, `/livox/imu_192_168_2_20x`; `x30_hw` maps `/leg_odom` -> `/odom`. The four lidar mounts (front-up, front-down, rear-up, rear-down) have documented body-frame positions and angles (manual p.66-69).

Sim notes (from docs/requests/sim-to-interfaces.md): the per-lidar frames named `lidar_fl/fr/rl/rr_link` are really front-up, front-down, rear-up, rear-down. About half the points of the real merged cloud (47% in the bag) are no-return points kept at the sensor origin with intensity 0, and the sim reproduces that: lidar odometry/mapping code MUST drop intensity-0 points at the origin before use. In `sim` the simulator itself publishes TF `odom->base_link`; the framework must not run a second publisher there (`odom_source: gt` means 'use the sim one'). Thermal default rate is 5 Hz.

Velocity limits (defaults, per backend in config): sim `legged`: vx +-0.4, vy +-0.3, wz +-0.5 (controller limits); sim `kinematic` default: vx 0.6, vy 0.3, wz 0.8. Real robot initial trial limits (digest section 5.1): vx 0.4 fwd / 0.15 back, vy 0.2, wz 0.5, accel <= 0.5 m/s2, command timeout <= 0.5 s; raise only after measured stopping distances. Real robot limits come from the docs digest and are enforced in `x30_hw`.

## 3. Localization and mapping topics (framework)
| Topic | Type | Notes |
|---|---|---|
| `/odometry/lidar` | `nav_msgs/Odometry` | lidar odometry (KISS-ICP or FAST-LIO2), frame `odom` |
| `/odometry/active` | `nav_msgs/Odometry` | the odometry selected by param `odom_source` in `configs/modes/*.yaml` (`gt`, `leg`, `lidar`, `fused`); this one drives TF |
| `/map` | `nav_msgs/OccupancyGrid` | 2D, transient local; from slam_toolbox or the occupancy mapper |
| `/scan` | `sensor_msgs/LaserScan` | derived from `/lidar_points` (height slice) for slam_toolbox/Nav2 |
| `/costmap_cloud` | `sensor_msgs/PointCloud2` | obstacle points (ground removed) for Nav2 local costmap |
| `/map_3d` | `sensor_msgs/PointCloud2` | optional accumulated map, low rate |

## 4. Semantic map
Topics: `/semantic_map/objects` (`x30_msgs/SemanticObjectArray`, 1 Hz, transient local), `/semantic_map/markers` (`visualization_msgs/MarkerArray`), `/semantic_map/annotated_map` (`sensor_msgs/Image`, top-down raster with text labels, MapNav-style), `/perception/detections_image` (debug overlay).
Services (all `x30_msgs/srv/Json`, request = JSON string, response = `ok` + JSON string, so schemas can evolve without rebuilding):
- `/semantic_map/query` `{ "label": "chair", "near": [x,y], "radius": 5, "attributes": ["warm"], "limit": 10 }`
- `/semantic_map/find` `{ "text": "the red chair near the window" }` (ranks instances, may call the mapper VLM)
- `/semantic_map/relations` `{ "id": 12 }` -> left/right/near/on/in-front-of with distances
- `/semantic_map/summary` `{ "pose": [x,y,yaw], "max_objects": 30 }` -> compact text for the reasoner
- `/semantic_map/reset`, `/semantic_map/save`, `/semantic_map/load` `{ "path": ... }`
Object record (`SemanticObject`): `id, label, confidence, position (map frame), size, point_count, observations, first_seen, last_seen, caption, attributes[] (e.g. "warm","hot","person-like"), room, best_view (sensor_msgs/CompressedImage), bearing_only`.
Perception output: `/perception/detections` (`x30_msgs/Detection3DArray`, one message per processed RGB frame; positions already in `map` through TF, lifted by projecting `/lidar_points` into the image and taking the mask's points; objects without enough points are `bearing_only`; thermal statistics per mask when the thermal camera is available). The semantic mapper consumes only this topic. Also: `/perception/diagnostics` (latencies, frame drop counts) and service `/perception/vocabulary` (`x30_msgs/srv/Json`, set per-task phrases `{"add":[...],"remove":[...]}`). Centroids are on the visible surface of the object, not its volumetric centre. Known limit (docs/reports/perception.md): the real lidar has no returns below about -10.5 deg elevation, so the lower part of a horizontal camera image cannot be lifted, and camera-lidar extrinsic error above about 4 deg makes thin objects snap to the wall behind them: calibrate to within 2 deg.
Perception server HTTP (JSON, base64 JPEG): `POST /segment {image, vocab:[...], conf, imgsz}` -> `{instances:[{label, score, box:[x1,y1,x2,y2], mask_rle:{size,counts}}], ms}`; `GET /health`; `POST /embed` optional. Port 8100.

## 5. Modes, tasks, skills
- Action `/x30/navigate` (`x30_msgs/action/Navigate`): the single entry point for a task. Fields in `x30_msgs`.
- Service `/x30/set_mode` (`x30_msgs/srv/SetMode`): `idle|teleop|velocity_follow|object_nav|goal_pose|explore` plus `follow_person` (person following by velocity commands, built: action `/x30/follow_person/run`, status `/follow/status`, docs/FOLLOW_PERSON.md; `velocity_follow` is language-instruction following and will be renamed `instruction_follow`). Topic `/x30/mode` (`std_msgs/String`, transient local) reports it.
- Topics `/cmd_vel_nav`, `/cmd_vel_teleop` (Twist) -> `cmd_vel_mux` (priority teleop > nav, timeouts 0.5 s) -> `safety_guard` -> `/cmd_vel`. `/cmd_vel_raw` (after mux, before guard) exists for debugging. Guard behaviour: scale or stop when `/costmap_cloud`/`/scan` shows an obstacle inside the stopping distance along the commanded direction; publish `/x30/safety` (`x30_msgs/SafetyStatus`).
- Skills: action `/x30/skill/<name>` (`x30_msgs/action/Skill`, goal `{name, args_json, timeout_s}`, result `{success, message, data_json}`) served by the skill server (one process, many names). Required skills: `goto_pose`, `goto_object`, `approach`, `explore`, `search`, `turn_to`, `move_relative`, `scan` (rotate in place), `verify`, `stop`. Skill args are JSON dicts; every skill returns a short human-readable `message` that is fed back to the reasoner.
- Reasoner tools (tool-calling schema) are generated from the skill registry. The reasoner can only finish with a `finish` tool call, subject to the finish guard (a `verify` yes + distance check).
- Velocity primitives for `velocity_follow` (NaVILA-style language actions): `move_forward(distance_cm)`, `move_backward`, `strafe_left/right(distance_cm)`, `turn_left/right(degrees)`, `stop`. The follower converts to timed `Twist` on `/cmd_vel_nav`, closes the loop on odometry, and aborts on guard stop.

## 6. Episodes and results
Episode file (YAML or JSONL; one object per episode):
```yaml
- id: house1_chair_01
  scene: aws_small_house         # scene id known to datasets/ registry
  mode: object_nav               # object_nav | velocity_follow | goal_pose
  instruction: "Find the chair in the living room and stop next to it."
  start: {x: 0.0, y: 0.0, yaw: 0.0}          # world frame
  goal: {type: object, label: chair, instance: chair_3, radius: 1.5}   # or {type: pose, x, y, radius}
  time_limit_s: 300
  split: dev                     # dev | heldout
  reference_path: []             # optional list of waypoints (for nDTW)
```
Results per episode are one JSON line in `results/<run>/episodes.jsonl`: `episode_id, success, oracle_success, final_distance_m, path_length_m, spl, duration_s, collisions, interventions, agent_turns, stop_reason, mode, config_hash, git/seed`. Per-turn traces in `results/<run>/traces/<episode>.jsonl`. Metrics: SR, OSR, SPL, DTG (distance to goal), nDTW when reference given, collision count, ATE of the SLAM pose vs ground truth, semantic-map position error.

## 7. Configuration files (`configs/`)
- `robot_sim.yaml`, `robot_real.yaml`: sensor extrinsics, intrinsics, limits, topic names, odometry source. Same schema; the sim file mirrors the real one and cites its source for every number.
- `models.yaml`: `perception: {url}`, `mapper_vlm: {base_url, model, max_tokens, temperature}`, `reasoner_vlm: {...}`; env vars override.
- `modes/<mode>.yaml`: parameters for each mode and skill.
- Launch: `ros2 launch x30_bringup framework.launch.py profile:=sim|real|replay mode:=object_nav` (sim container: `ros2 launch x30_sim sim.launch.py world:=<id> locomotion:=legged|kinematic`).

## 8. Conventions
Python 3.10 (framework) / 3.12 (sim), type hints, `ruff`-clean, no hard-coded absolute paths, parameters through ROS params with defaults from the config files, logs through `get_logger`. Each package has `README.md` (what/how to run/tests), unit tests runnable **without ROS** where possible (`pytest`), and an integration check script in `scripts/` or `test/` that returns non-zero on failure. DDS: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` in all containers with the shared peer file from `docker/` (decided by the infra agent; do not set your own).

## Teaching (added 2026-10-12)
Service `/teach/say` (x30_msgs/srv/Json, request `{"text": "...", "x"?, "y"?, "at"?: "robot"}`, response JSON with `ok`, `action` save|rename|forget|ignore|error, `reply`, `interpretation`, `applied`, `context`, `position`, `position_source`), topic `/teach/reply` (std_msgs/String JSON). Mapper services `/semantic_map/add_place|places|remove_place|place_context` (Json). Map summary block `Places (taught by people...)`. Docs: docs/TEACH_PLACES.md.
