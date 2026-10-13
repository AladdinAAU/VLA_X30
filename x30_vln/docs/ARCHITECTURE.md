# X30 VLN framework: architecture

Status: design contract (2026-10-09). Every package must follow `INTERFACES.md`. Anything marked *(verify)* has not been tested here.

## 1. Goal

A modular framework to run vision-language navigation trials on the DEEP Robotics X30 quadruped, first in simulation
(Gazebo), then on the real robot **with the same nodes, topics and configs**. Nothing is trained; the language models
decide *what* to do, geometry (SLAM, costmaps, planners) decides *how*.

Navigation modes, switchable at runtime. NAMING CORRECTION (2026-10-11): the user's 'following using velocity commands' means FOLLOWING A PERSON (`follow_person`, not built yet). What was built and is still needed is language-instruction following, currently named `velocity_follow` (to be renamed `instruction_follow` in a later cleanup; keep the old name until then):

| Mode | Input | Output to robot | Idea (papers) |
|---|---|---|---|
| `velocity_follow` | free-form instruction ("go down the corridor, turn left at the door") | short velocity primitives on `/cmd_vel_nav` (forward x cm, turn y deg, stop) | NaVILA mid-level language actions + geometric safety layer |
| `follow_person` (built 2026-10-11, simulator only; docs/FOLLOW_PERSON.md) | a target person (nearest in front, or described: 'the person in the red jacket') | velocity commands on `/cmd_vel_nav` keeping a set distance and bearing | person detection + tracking + lidar obstacle avoidance; see docs/HANDOFF.md section 9 |
| `object_nav` | object / place ("find the fire extinguisher", "go to the chair near the window") | Nav2 goals + approach/verify skills | MapNav / OrionNav / SysNav: semantic map + planner + VLM verification |

Also available: `goal_pose` (Nav2 to a pose), `explore` (frontier), `teleop`, `idle`.

## 2. Design principles

1. **Sim/real parity.** The robot-facing contract is a small set of topics (INTERFACES.md section 2). Gazebo, the real robot (through the ROS1 bridge), and rosbag replay all publish exactly these. Downstream nodes never know which one they talk to.
2. **Sensors as on the real X30.** Four Livox lidars (2 front, 2 back; merged by the robot's perception host into `/lidar_points`, ~80k points at 10 Hz), an RGB camera, a thermal camera, IMUs, leg odometry. **There is no depth camera on the real robot**: 3D positions of segmented objects come from projecting lidar points into the camera image, not from depth images. The simulator can publish a depth image for ground-truth checks, but the stack must default to `depth_source: lidar`.
3. **Two model roles, many backends.** A *mapper* VLM/LLM annotates the semantic map (object captions, attributes, room types, fused with thermal); a *reasoner* VLM does task-level reasoning and tool calling. Both are OpenAI-compatible HTTP endpoints (llama.cpp / vLLM / hosted), configured in `configs/models.yaml`. They may be the same server.
4. **Everything is a skill behind one interface.** Planners and behaviours are skills (`x30_msgs/action/Skill`), the reasoner chooses skills by tool calling, and the mode manager can also run skills directly without any LLM (for ablations and tuning).
5. **One safety path.** All velocity commands pass `cmd_vel_mux` and `safety_guard` before the locomotion backend. The real-robot backend adds watchdog, speed limits and e-stop (see `docs/robot/x30-interfaces.md`).
6. **Honest evaluation.** Fixed episode files, ground truth from the simulator, repeats, dev/held-out split, no tuning on held-out (lesson from the drone project, `PROJECT-SUMMARY.md` there).

## 3. Containers

| Container | Base | Role | GPU |
|---|---|---|---|
| `sim` | ROS 2 Jazzy + Gazebo Harmonic (the working prework: legged controller, X30 model) | world, robot, simulated sensors, locomotion, ground truth. Publishes the robot contract only | rendering (software by default, see risks) |
| `framework` | ROS 2 **Humble** (same as the real-robot workstation in `Rob7_Gr760_P7`) | SLAM/odometry, Nav2, perception client, semantic map, modes, agent, eval | no |
| `ros1_bridge` | `ros:foxy-ros1-bridge` (as in Rob7) | real robot only: ROS1 Noetic <-> ROS 2 for lidar, IMU, camera, tf | no |
| `perception` | CUDA + PyTorch + Ultralytics | HTTP server: open-vocabulary segmentation (YOLOE-seg), optional SAM-class refinement | yes |
| `vlm` | llama.cpp server (CUDA) | Qwen3-VL (reasoner, and mapper if one model) | yes |
| `tools` (optional) | foxglove bridge / web console | visualisation, operator UI | no |

Why Humble for the framework: the real-robot path (Foxy ros1_bridge image + Humble workstation) is the one already proven in `Rob7_Gr760_P7`; Humble<->Foxy DDS interop is the well-trodden route, and the existing perception/VLM images are Humble-compatible. The sim stays Jazzy because that is where the controller work was done; sim and framework talk over DDS using only standard messages (`sensor_msgs`, `nav_msgs`, `geometry_msgs`, `tf2_msgs`). If the controller turns out to build on Humble, the two can be merged later. *(verify Humble<->Jazzy interop in Docker Desktop first; it is the first test of the infra agent.)*

Compose profiles: `sim` (sim + framework + models), `real` (ros1_bridge + framework + models), `replay` (bag playback + framework), `models` (perception + vlm only).

## 4. Node graph (framework container)

```
 SIM or REAL or REPLAY                       framework
 /lidar_points ─────────┬──▶ lidar_odometry (KISS-ICP | FAST-LIO2 | gt | leg) ─▶ TF odom→base_link, /odometry/lidar
 /imu/data              │
 /odom (leg or gt) ─────┤
                        ├──▶ occupancy_mapper (3D voxel → 2D grid, ground/obstacle split) ─▶ /map, /local_costmap cloud
                        │    slam_toolbox (2D, optional loop closure) ─▶ TF map→odom
                        │
 /camera/color/image_raw┼──▶ perception_client ──HTTP──▶ [perception server: YOLOE-seg (+SAM)]
 /thermal/image_raw     │          │ masks + labels
                        │          ▼
                        └──▶ semantic_mapper  (lift masks to 3D by lidar→image projection, thermal fusion,
                                   │            instance association, VLM annotation via [mapper VLM])
                                   ▼
                          /semantic_map/objects, /semantic_map/annotated_image (MapNav-style top-down),
                          services: query / find / relations / region_summary
                                   │
 instruction ─▶ mode_manager ──────┼──▶ velocity_follow (reasoner VLM → primitives) ─▶ /cmd_vel_nav
   (Navigate action)               ├──▶ object_nav (reasoner agent + skills) ─▶ Nav2 NavigateToPose ─▶ /cmd_vel_nav
                                   └──▶ explore / goal_pose / teleop
 /cmd_vel_nav, /cmd_vel_teleop ─▶ cmd_vel_mux ─▶ safety_guard (lidar) ─▶ /cmd_vel ─▶ locomotion
                                                                          (sim controller | nav_bridge UDP)
```

## 5. Packages (ROS 2 workspace `ros2_ws/src`)

| Package | Owner area | Contents |
|---|---|---|
| `x30_msgs` | contract | messages, services, actions |
| `x30_sim` (Jazzy) | simulation | robot model + sensors, worlds, locomotion (`legged` and fast `kinematic`), lidar merge, ground truth, scene importers |
| `x30_hw` | real robot | ROS1 bridge config, nav_bridge integration, camera/thermal ingest, health, replay tools, pre-flight |
| `x30_navigation` | localization/planning | lidar odometry, 2D mapping, Nav2 config for the X30, frontier explorer |
| `x30_perception` | perception | perception client node, perception server, thermal processing, lidar-camera projection |
| `x30_semantic_map` | mapping | semantic object map, VLM annotator, annotated top-down map, query services |
| `x30_modes` | behaviour | mode manager, cmd_vel_mux, safety_guard, velocity_follow, skills |
| `x30_agent` | reasoning | reasoner agent (tool calling), prompts, model client, trace logging |
| `x30_eval` | evaluation | episode formats, runner, metrics, sweeps, data export for fine-tuning |
| `x30_bringup` | integration | launch files and `configs/` for sim, real, replay |

## 6. Repository layout

```
x30_vln/
  README.md  docs/  reference/ (papers text, repos, robot docs; read-only)
  docker/    compose + images + scripts (Windows .ps1 and Linux .sh)
  configs/   robot_sim.yaml robot_real.yaml models.yaml modes/*.yaml
  datasets/  adapters, scene importers, episode files
  ros2_ws/src/<packages above>
  scripts/   helper entry points
  results/   run outputs (gitignored)
```

## 7. Datasets and worlds
Adapters turn a scene (SDF world, or mesh glb/obj/ply) plus an episode file into a runnable Gazebo world and a ground-truth object list. Format of episodes is in INTERFACES.md section 6. Habitat-style datasets (HM3D, MP3D, R2R, ObjectNav) require licence acceptance by the user; adapters are written to load them when the user has downloaded them, and free worlds (AWS RoboMaker small house/hospital/warehouse/bookstore, procedural rooms) are used meanwhile. See `docs/research/papers-digest.md`.

## 8. Fine-tuning before deployment (what it means here)
1. **Parameter tuning** of speed limits, costmap inflation, thresholds, approach standoff, prompts: sweeps in the fast `kinematic` sim with Optuna (`x30_eval`), validated on `legged`.
2. **Model-level**: log every reasoner turn (image, map summary, answer) so a LoRA fine-tune of the VLM on successful trajectories is possible; exporter provided, training itself optional.
3. **Sim-to-real checks**: replay the real bag through the same nodes (lidar odometry, mapping), compare statistics (points per scan, vertical distribution) with the simulated lidar, and keep a measured-parameters file for the real robot.

## 9. Risks (known from earlier work, carry over)
- Gazebo camera rendering on Mesa D3D12 under WSL leaked ~10 MB/s; software rendering (llvmpipe) was stable. Default to software rendering for `sim`, watch memory, restart on threshold.
- WSL is capped at 24 GB, swap off (`~/.wslconfig`). Do not raise without a reason.
- Start Docker only through `docker-clean-start.ps1` (stale socket workaround).
- Real-robot network: ROS1 needs the robot to reach back to the container (host networking). On Windows, use WSL mirrored networking or run the real-robot profile on a Linux workstation.
- Legged physics needs real-time factor close to 1; heavy worlds or 4 software-rendered lidars may break it. That is why the `kinematic` locomotion backend exists.
