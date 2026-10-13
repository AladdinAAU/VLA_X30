# x30_sim

Gazebo Harmonic simulation of the DEEP Robotics X30 behind the X30 VLN interface contract (`docs/INTERFACES.md`). ROS 2 **Jazzy**,
runs in the `x30-sim` container. It publishes the same robot-facing topics as the real robot: four Livox lidars merged into the
exact `/lidar_points` layout, IMU, odometry, RGB camera, optional thermal camera, battery, health, ready/e-stop, plus sim-only
ground truth. Every number is in `configs/robot_sim.yaml` with its source or the word "assumption".

## Run

```bash
# host (PowerShell / Git Bash), from x30_vln/
docker compose -f docker/compose.sim.yml --profile sim up -d sim          # image x30-sim:latest (docker/images/sim)
docker exec -it x30-sim x30-build                                         # first time and after code changes
docker exec -it -e ROS_DOMAIN_ID=10 x30-sim ros2 launch x30_sim sim.launch.py world:=default_arena locomotion:=kinematic
```

Use a distinct `ROS_DOMAIN_ID` for stand-alone tests (all containers on `x30_net` share domain 0 by default). Inside the container the
project is mounted at `/x30` and the sources at `/ws/src`. Stop with `scripts/sim/kill_sim.sh` (inside the container) or `docker stop x30-sim`.

Launch arguments (`ros2 launch x30_sim sim.launch.py ...`):

| argument | default | meaning |
|---|---|---|
| `world` | `default_arena` | id in `datasets/registry.yaml` (`worlds: {id: {sdf: <path relative to x30_worlds>, spawn: {x,y,z,yaw}}}`) or the built-in arena |
| `locomotion` | `kinematic` | `legged` (unitree_guide_controller, physics at 1 ms, needs real-time factor ~1) or `kinematic` (VelocityControl, 5 ms, flat floors) |
| `gui` | `false` | Gazebo GUI (needs a display; slow with software rendering) |
| `render` | `$RENDER` or `sw` | `sw` Mesa llvmpipe (stable memory), `gpu` WSL2 D3D12 (add `docker/compose.sim.gpu.yml`; leaks with cameras) |
| `thermal` | `true` | optional thermal camera payload (not stock on the X30) |
| `depth` | `true` | sim-only depth image |
| `spawn_x/y/z/yaw` | registry spawn | start pose in the world frame |
| `odom_noise` | `0.0` | 0 = ground-truth `/odom`, 1 = nominal leg-odometry error (`odometry.nominal` in the config), 2 = double, ... |
| `latency` | config (0.06 s) | kinematic backend command latency |
| `rtf` | `1.0` | target real-time factor in the world physics |
| `robot_config` | `$X30_CONFIGS/robot_sim.yaml` | sensor/limit configuration |

## Interfaces

Contract topics: `/lidar_points`, `/imu/data`, `/odom` (+ TF `odom -> base_link`), `/camera/color/{image_raw,camera_info}`,
`/camera/depth/image_raw` (sim only), `/thermal/{image_raw,temperature,camera_info}`, `/x30/{ready,battery,estop,health}`, `/clock`, input `/cmd_vel`.
Extras: per-lidar clouds `/lidar/{fl,fr,rl,rr}/points` (sensor frames, organised, `inf` = no return), `/thermal/raw` (mono16 sensor output),
`/gt/pose`, `/gt/objects`, `/gt/scene_info` (latched), `/gt/odom_raw`.
Services (`x30_msgs/srv/Json`): `/sim/set_robot_pose` `{"x":..,"y":..,"yaw":..,"z":optional}` and `/sim/reset_world` `{}`; both reset the
odometry origin and, in `legged` mode, park and restart the controller; Gazebo is never restarted. Frames: `fl = front-up`, `fr = front-down`,
`rl = rear-up`, `rr = rear-down` (the real robot has up/down pairs, see `docs/requests/sim-to-interfaces.md`).

## Behaviour worth knowing

* **Lidar**: 4 x gz `gpu_lidar` (416 x 48 rays, -7..+52 deg about each sensor's boresight, range 0.1..40 m, 3 cm noise) mounted at the documented
  positions with the documented tilt angles; `lidar_merge` transforms to the body frame (= `lidar_link`), masks rays through the robot body
  (boxes in the config), turns invalid rays into points at the sensor origin with intensity 0 (as the real cloud does), and synthesises intensity,
  tag, line and per-point time. Compare with the real bag: `scripts/sim/compare_lidar_to_bag.py`.
* **Thermal**: world authors set a temperature (Kelvin) with `<plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal"><temperature>310</temperature></plugin>`
  **inside the `<visual>`** of the object; everything else is at `thermal.ambient_c`. `thermal_processor` converts the 16-bit output (0.01 K per count) into
  `/thermal/temperature` (Celsius) and the mono8 display image.
* **Kinematic backend**: the model has its 12 leg joints frozen in the standing pose, no gravity, a body-footprint bumper, and is moved by the Gazebo
  `VelocityControl` plugin. `/cmd_vel` passes a 2 % dead zone, the limits (0.6 / 0.3 / 0.8), a 0.5 s timeout, a latency (0.06 s) and acceleration limits (1 m/s2, 2 rad/s2).
  It does not follow stairs or slopes.
* **Legged backend**: prework controller (`unitree_guide_controller` + patch, built into the image) with `legged_bridge` standing the robot up; limits 0.4 / 0.3 / 0.5.
* **World preparation**: the world SDF is copied to `/tmp/x30_sim` with relative URIs made absolute, missing systems (physics, sensors, IMU, user commands,
  scene broadcaster) added, physics step and real-time factor set, and an ambient temperature; the original is not touched.
* **Ground truth labels**: see `docs/requests/sim-to-worlds.md`.

## Tests

```bash
docker exec x30-sim x30-test                                  # pytest, no ROS nodes needed
MSYS_NO_PATHCONV=1 docker exec -e ROS_DOMAIN_ID=10 x30-sim /x30/scripts/sim/smoke_test.sh kinematic 120   # boots the sim, checks the contract, drives 3 m, measures RTF
```

`WORLD=<registry id>` selects another world. Results land in `results/sim_smoke/<mode>/`. Measured numbers and known issues: `docs/reports/sim.md`.
