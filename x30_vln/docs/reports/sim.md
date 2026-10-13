# Simulator report (x30_sim)

Status 2026-10-09. Package `ros2_ws/src/x30_sim`, image `x30-sim:latest` (`docker/images/sim`), compose `docker/compose.sim.yml` (+ `compose.sim.gpu.yml`),
config `configs/robot_sim.yaml`, scripts `scripts/sim/`, figures `docs/figures/sim/`. How to run: `ros2_ws/src/x30_sim/README.md`.
Every figure below was measured in this session unless it says "not tested". The host is shared with other agents' containers, so RTF numbers carry a few percent noise.

## 1. What exists

| item | state |
|---|---|
| Docker image | Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic 8.15 (via ros_gz), ros2_control, gz_ros2_control, the two pinned upstream repositories at the prework commits, controller patch applied, upstream packages built into `/opt/x30_ext/install` (6.0 GB image). `quadruped_navigation_ros2` is cloned for reference but not built (the framework container does navigation). Build lock taken for every build. Rendering default `RENDER=sw` (Mesa llvmpipe, EGL surfaceless, headless); `RENDER=gpu` supported (section 5). |
| Robot | upstream X30 URDF + 4 Livox `gpu_lidar` at the documented mounts and tilts, ROS IMU (200 Hz, documented position), RGB(-D) camera, optional thermal camera, merged `/lidar_points`. All numbers with sources in `configs/robot_sim.yaml`. |
| Locomotion | `legged` (prework `unitree_guide_controller` + cmd_vel bridge, ported and cleaned, limits 0.4/0.3/0.5) and `kinematic` (legs frozen in the standing pose, gravity-free, Gazebo VelocityControl, limits 0.6/0.3/0.8, 2 % dead zone, 0.5 s timeout, 0.06 s latency, 1 m/s2 and 2 rad/s2 acceleration limits). Both consume only `/cmd_vel`. |
| State outputs | `/odom` (planar, ground truth with optional leg-odometry error, `odom_noise`), TF `odom -> base_link`, `/gt/pose`, `/x30/ready`, `/x30/battery` (fake, 20 %/h), `/x30/health` (rates per subsystem + real-time factor), `/x30/estop` honoured. |
| Ground truth | latched `/gt/objects` (SemanticObjectArray, from the world SDF, label convention in `docs/requests/sim-to-worlds.md`) and `/gt/scene_info`; services `/sim/set_robot_pose`, `/sim/reset_world` (both reset the odometry origin; legged mode parks and restarts the controller; Gazebo and the clock are never restarted). |
| Worlds | by id from `datasets/registry.yaml` (tested: `apt_01`, `lab_01`), plus built-in `default_arena` and `lab_room`. The world SDF is copied, URIs absolutised, missing systems/physics/atmosphere added; the original is untouched. |
| Tools | `scripts/sim/smoke_test.sh`, `smoke_check.py`, `measure_rtf.py`, `experiment.sh`, `grab_frame.py`, `drive_probe.py`, `capture_sim_scans.py`, `compare_lidar_to_bag.py`, `make_figures.py`, `kill_sim.sh`. 14 unit tests (`docker exec x30-sim x30-test`), all passing. |

## 2. Verification (smoke test, both locomotion modes)

`scripts/sim/smoke_test.sh kinematic|legged 120` boots headless, checks and then measures. Result in `default_arena`: **29/29 checks passed in both modes**
(plus `apt_01`, `lab_01` in kinematic and `lab_01` in legged). Checks: every contract topic exists with the right type and a simulated-time rate
(`/lidar_points` 10.0 Hz, `/imu/data` 200 Hz, `/odom` 50 Hz, RGB 10 Hz, depth 10 Hz, thermal 5 Hz, camera infos, battery/health 1 Hz, `/gt/pose` 100 Hz);
merged cloud layout exactly as the contract (fields, offsets, `point_step` 26, `is_dense`, frame `lidar_link`, 79872 points per scan, lines 0..15); RGB `rgb8` in
`camera_optical_frame`, depth `32FC1`, thermal `mono8` + `32FC1` Celsius; TF tree complete (`odom`, `base_link`, `lidar_link`, `imu_link`, both optical frames, four lidar frames);
3 m drive with `/cmd_vel` 0.3 m/s (kinematic: reached 3.01 m, final odom = ground truth 3.08 m; legged: reached 3.01 m, then the controller pulls back, final 2.73 m, odom = GT);
`/sim/set_robot_pose` (error 0.00 m kinematic, 0.09 m legged); e-stop holds the robot (0.000 / 0.005 m moved with a 0.3 m/s command); `/sim/reset_world`; ground-truth objects
(7 in `default_arena`, 62 in `apt_01`, 113 in `lab_01`, matching the registry counts). Images viewed: the RGB frame shows the world (box, walls, the "person" cylinder) and the thermal frame shows only the 309 K object bright against the 22 C ambient
(`docs/figures/sim/rgb_thermal.png`; the radiator peaks at 68.7 C in the temperature image, set 343 K = 69.9 C). Saved to `results/sim_smoke/<mode>/` (rgb.png, thermal.png, lidar.pcd, lidar_bev.png, report.json, rtf.csv).

## 3. Performance (software rendering, 24 logical CPUs shared with other agents)

Defaults: lidar 4 x 19968 rays at 10 Hz, RGB 640x480 + depth at 10 Hz, thermal 384x288 at 5 Hz.

| configuration | RTF | gz sim CPU / RSS | ROS nodes CPU / RSS |
|---|---|---|---|
| kinematic, `default_arena`, 120 s | **0.99** | 120 % / 523 MB | 190 % / 610 MB |
| kinematic, 12 min soak | 0.98 mean (0.95..1.00) | 124 % / 518 -> 516 MB | 194 % / 603 -> 615 MB |
| legged, `default_arena`, 120 s | **0.88** | 185 % / 568 MB | 318 % / 615 MB |
| kinematic, `apt_01` (64 models) | 0.82 | 778 MB | |
| kinematic, `lab_01` (28 x 16 m, 113 objects) | 0.74 | 1017 MB | |
| legged, `lab_01` | 0.62 (smoke test still 29/29, robot stays up, no falls seen) | 1050 MB | |

Memory growth: **none measured in software rendering** (Gazebo RSS 518 -> 516 MB over 12 minutes; ROS nodes +12 MB). `docs/figures/sim/soak_rtf_memory.png`.
Only 12 minutes of kinematic mode were soaked; legged was run for minutes, not 10+.

### What costs real time (this decided the defaults)
1. **The upstream meshes (1.3 M vertices, shank.obj 210 k x 4 legs)** cost about 100 ms per sensor frame under llvmpipe: with tiny sensors RTF was 0.2, independent of ray count or image size.
   They are now decimated at launch by vertex clustering (`robot.visual_mesh_voxel_m: 0.015`: torso 94 k -> 4.1 k vertices, shank 210 k -> 0.3 k; about 2 s once, then cached). RTF with the full sensor suite went from 0.002 (first run) to ~1.0.
2. Remaining cost, kinematic, `default_arena`, steady RTF (one run each, 20 s):

| variant | RTF |
|---|---|
| camera 1280x720 + depth + thermal 384x288 @ 8 Hz | 0.56 |
| camera 640x480 + depth + thermal 384x288 @ 8 Hz | 0.65 |
| camera 1280x720 + depth, no thermal | 0.78 |
| camera 640x480 + depth + thermal 256x192 @ 5 Hz | 0.77 |
| camera 640x480 + depth + thermal 384x288 @ 5 Hz (**default**) | 0.90-0.99 |
| camera 640x480 + depth, no thermal | 0.99 |
| nothing but the 4 lidars | 0.98 |

The thermal camera is the most expensive single sensor (about 0.3 RTF at 8 Hz), the 1280x720 RGB-D camera the second (about 0.2). The four lidars at full density are cheap (the rays are not the cost).
`scripts/sim/experiment.sh` repeats these runs with a modified copy of the config. Options if a world is heavier: lower `thermal.rate_hz`, `camera.depth: false` (launch `depth:=false`), 640x480, or `thermal:=false`.

### Legged mode
The controller needs a real-time factor near 1; measured 0.88 (`default_arena`) and 0.62 (`lab_01`) with every sensor on, with no falls observed in the runs listed above (three smoke runs, a 24 s straight walk of 5.6 m, set_robot_pose and reset_world cycles).
A systematic fall-rate study at low RTF was not done. For parameter sweeps and large worlds use `kinematic`.

## 4. Lidar against the real bag (`scripts/sim/compare_lidar_to_bag.py`, full tables in `docs/reports/sim-lidar-compare.md`, plots `docs/figures/sim/lidar_*.png`)

Facts derived from the bag (60 scans) and used to build the sensor model: the cloud is the concatenation of the four lidars in the order rear-up, rear-down, front-down, front-up (the zero points sit exactly at the documented mounts, so `lidar_link` is the body frame);
`line = 4 * lidar + (0..3)` cycling per point; **47 % of the points are invalid returns kept at the sensor origin with intensity 0**; `timestamp` is in milliseconds over 100 ms; the header stamp equals the first point's timestamp; the
sensor axes of all four lidars were checked against the manual's angles (for each lidar the pitch interval in which every bag point lies inside the [-7, +52] deg band contains the manual value: front-up 24..28 vs 25, front-down -46..-40 vs -46, rear-up -6..0 vs -5, rear-down -58..-52 vs -56).
In the bag the floor is only 0.12-0.2 m below the body frame and the ceiling 2.3 m above: the robot was **lying down**. The standing sim robot is 0.51 m above the floor, so the comparison world `lab_room` raises the floor visually (visual-only box) to give the same spacing.

| | real bag | simulation (`lab_room`) |
|---|---|---|
| points per scan | 80000 +- 783 (77k..82k) | 79872 (fixed) |
| valid fraction | 0.472 | 0.596 |
| valid fraction per lidar (rear-up, rear-down, front-down, front-up) | 0.52, 0.39, 0.41, 0.57 | 0.72, 0.43, 0.54, 0.70 |
| Jensen-Shannon distance of the range / elevation / height histograms | | 0.055 / 0.016 / 0.027 |
| azimuth coverage (36 bins) | 1.00 | 1.00 |
| intensity p5/p25/p50/p75/p95/max | 0/1/4/18/81/255 | 0/1/3/15/106/255 |
| line values, timestamp span | 0..15, 100.3 ms | 0..15, 100.0 ms |

**Residual differences, honestly:**
* The sim has about 12 points per 100 more valid returns. The real robot loses many more rays (lying posture, legs and body in the beams, unknown driver filtering); my body boxes are assumptions. I did not add a random dropout to hide this.
* Real range histogram has a tail beyond 5 m (6 % of points, long corridors/open doors, the lab is larger and cluttered) that the 7 x 6 m sim room does not have; the sim has fewer points between 0.5 and 3 m. Geometry differs by construction.
* Intensity is synthetic (lognormal around the bag's median with range falloff); tags are sampled from the bag's empirical tag distribution. Both match the bag's histograms by construction, not because the simulation models reflectivity or Livox confidence tags.
* The Livox non-repetitive rosette pattern is replaced by a regular 416 x 48 grid per lidar (19968 rays, the bag's count); points are ordered ray by ray and timestamps are linear in the index, with no motion distortion.
* About 0.5 % of the sim points lie below the floor (z down to -2.5 m); not explained yet (ray artefacts at the plane's edge or through the visual floor of the comparison world are suspects).
* Noise: 3 cm Gaussian as specified in the brief (Livox datasheet values not verified).

## 5. Rendering: software vs GPU
* **Software (default)**: llvmpipe over EGL surfaceless (`gz sim -s --headless-rendering`); stable memory; numbers above.
* **GPU** (`RENDER=gpu`, `docker compose -f docker/compose.sim.yml -f docker/compose.sim.gpu.yml up -d sim`, `render:=gpu`): works only through the WSLg X display with `GALLIUM_DRIVER=d3d12` (EGL surfaceless cannot initialise the D3D12 driver; `glxinfo` then reports "D3D12 (NVIDIA GeForce RTX 3090 Ti)").
  Measured over 60 s, kinematic, all sensors: RTF 0.997 (software 0.98), Gazebo CPU 68 % (software 120 %), ROS-side CPU unchanged (190 %, so the speed-up is small), **Gazebo RSS grew 500 -> 574 MB in 60 s (about 1.2 MB/s, 74 MB/min)**. That is below the 10.7 MB/s seen in the drone project with a bigger camera, but it is still growth; a 10-minute GPU soak was not run. Use software rendering for anything long; use the GPU path for short runs or when a heavy world needs it, with the host memory guard of `docs/INFRA.md`.

## 6. Thermal camera
Works with Gazebo's `thermal` sensor under software rendering (no fallback needed). World authors set a temperature in Kelvin with the Thermal plugin **inside the `<visual>`** (verified; a model-level plugin does nothing), everything else is at `thermal.ambient_c` (22 C, uniform: the simulator writes `<atmosphere>` with gradient 0). Output is 16-bit, 0.01 K per count, converted by `thermal_processor` into `/thermal/temperature` (Celsius, float32) and a mono8 display image (auto-gain, minimum span 10 K, or a fixed window). Resolution 384x288, hfov 56 deg, 5 Hz, mounted 8 cm left of the RGB camera: all assumptions (the stock X30 has no thermal camera).

## 7. Known issues and limits
* `kinematic` follows no terrain: constant body height, flat floors only; stairs/ramps need `legged`. The footprint is a 0.98 x 0.30 x 0.20 m bumper (the real body with legs is slimmer in the middle).
* `legged`: after a stop command the controller pulls the robot back about 0.15-0.3 m; the heading after standing up is not exactly the spawn yaw (0.01 to 0.12 rad seen); `set_robot_pose` parks the robot standing, moves it and restarts trotting (a few seconds to some ten seconds wall time, not timed).
* RGB intrinsics/pose, thermal module, IMU noise, odometry error model, battery model, latency and acceleration limits are assumptions (see `configs/robot_sim.yaml`). `odom_noise > 0` is unit-tested but was not run in a live drive.
* TF `odom -> base_link` is published by the simulator (planar). A second publisher in the framework (`/odometry/active`) must be disabled while it runs (`docs/requests/sim-to-interfaces.md`).
* Frame/topic names `lidar_fl/fr/rl/rr` are kept from INTERFACES but mean front-up/front-down/rear-up/rear-down; per-lidar topics `/lidar/<id>/points` are the raw Gazebo clouds (sensor frame, `inf` for no return).
* The GUI (`gui:=true`) was not tested. Fuel (https) model URIs are not fetched. World objects that are meshes without a primitive collision get a 0.5 m placeholder size in `/gt/objects`; moving objects are not tracked (ground truth is read once from the SDF).
* The `gz_frame_id` sensor element is reported as "not defined in SDF" by sdformat at spawn (harmless: the frame ids are applied, verified in the smoke test).
* The Docker image does not contain `graphviz` (used once, installed by hand, to draw the TF tree); `x30-build` and `x30-test` set `-eo pipefail` without `-u` because ROS setup scripts use unset variables.
* `/x30/ready` in kinematic mode becomes true 1 s after the first ground-truth pose; in legged mode it follows the controller (trotting).

## 8. Next steps
1. Run the framework container against the sim (Humble <-> Jazzy over DDS) with distinct `ROS_DOMAIN_ID`s; lidar odometry on `/lidar_points` should drop intensity-0 points at the sensor origins.
2. Calibrate invalid-ray masks and the camera/thermal intrinsics once the real robot gives a calibration and a standing lidar recording.
3. 10-minute GPU soak and a legged soak; a fall-rate study of the legged controller against RTF.
4. Optional: non-repetitive Livox pattern via per-frame rotation of the ray grid; reflectivity from world materials.
