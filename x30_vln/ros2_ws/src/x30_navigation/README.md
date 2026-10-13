# x30_navigation

Localization, 2D mapping, Nav2 and frontier exploration for the X30 (ROS 2 Humble, `x30-framework` image). Contract: `docs/INTERFACES.md` sections 1-3. Report with numbers: `docs/reports/navigation.md`. All parameters live in `configs/navigation/` (found through `$X30_CONFIGS`, which the framework image sets to `/x30/configs`; or pass `config_dir:=`).

## Data flow

```
/lidar_points ─┬─ lidar_odometry (KISS-ICP, deskew from per-point timestamp) ─ /odometry/lidar ─┐
/odom (leg/gt) ┼───────────────────────────────────────────────────────────────────────────────┼─ odom_selector ─ /odometry/active + TF odom->base_link, /x30/health
/gt/pose (sim) ┘                                                                               ┘   (odom_source: gt|leg|lidar|fused)
/lidar_points, /imu/data ─ cloud_leveler ─ TF base_link->base_footprint, /nav/cloud_level (z = height above ground, flags ground/obstacle)
      ├─ cloud_to_scan ─ /scan          ├─ obstacle_cloud ─ /costmap_cloud       └─ occupancy_mapper ─ /map + TF map->odom   (slam: occupancy)
      slam_toolbox (reads /scan; publishes /map, map->odom)                                                                   (slam: slam_toolbox)
/map, /scan, /costmap_cloud, /odometry/active ─ Nav2 (MPPI or RPP, Smac2D) ─ /cmd_vel_nav        navigate_to_pose
/map + TF ─ frontier_explorer ─ service /nav/next_frontier, /nav/explore/{start,stop}, /nav/explore_status
```

`base_footprint` is a new frame (level, on the ground below `base_link`, body heading): scan, costmap cloud and the mapper use it so a pitching body does not put the floor into the 2D data. It is published only by `cloud_leveler`.

## Run

```bash
# inside x30-framework (or any Humble container with nav2, slam_toolbox, kiss-icp)
cd /ws && colcon build --symlink-install --packages-up-to x30_navigation && source install/setup.bash
ros2 launch x30_navigation navigation.launch.py profile:=sim            # odom_source leg, Nav2 on, occupancy map
ros2 launch x30_navigation navigation.launch.py profile:=real           # odom_source fused, auto fallback to lidar
ros2 launch x30_navigation navigation.launch.py profile:=replay nav2:=false static_lidar_tf:=true   # bag replay (lidar only)
# options: odom_source:=gt|leg|lidar|fused  slam:=occupancy|slam_toolbox|none  nav2:=true|false  explore:=true
#          limits:=default|legged  controller:=mppi|rpp  use_sim_time:=true  config_dir:=<configs/navigation>
ros2 param set /odom_selector odom_source lidar        # switch at run time (output stays continuous)
ros2 service call /nav/next_frontier x30_msgs/srv/Json '{request: "{\"strategy\": \"nearest\", \"exclude\": []}"}'
```

Sub-launches: `odometry.launch.py`, `mapping.launch.py`, `nav2.launch.py`, `explore.launch.py`.
Nav2 output is `/cmd_vel_nav` only (controller -> `/cmd_vel_nav_raw` -> velocity_smoother -> `/cmd_vel_nav`); `/cmd_vel` is produced by `x30_modes` (mux + safety guard).

## Nodes

| Node | Purpose | Key parameters (configs/navigation/navigation.yaml) |
|---|---|---|
| `lidar_odometry` | KISS-ICP wrapper; gravity-aligned odom frame from the first ground fit; optional `/map_3d` | `max_range 30`, `min_range 0.9`, `voxel_size 0.4`, `min_motion_th 0.05`, `deskew`, `publish_map_3d` |
| `odom_selector` | source choice, single TF publisher, continuity on switch, watchdog (jump, stale, leg/lidar disagreement) on `/x30/health` as `navigation/odometry` | `odom_source`, `fusion.k_pos/k_yaw`, `watchdog.*`, `auto_fallback` |
| `cloud_leveler` | ground plane fit (IMU-seeded RANSAC), `base_footprint` TF, levelled and classified cloud | `tilt_source auto/ground/none`, `band.*` (ground 0.08, obstacle 0.15-0.90 m), `ground.*` |
| `cloud_to_scan` | nearest obstacle per azimuth bin | `num_beams 720`, `range_min/max` |
| `obstacle_cloud` | obstacle points, voxel 5 cm, speckle filter, range limit | `max_range 8`, `voxel_size`, `min_points` |
| `occupancy_mapper` | log-odds grid, ray cast free space, growing map, `map->odom` identity | `grid.*` |
| `frontier_explorer` | frontier service and autonomous loop | `strategy`, `frontier.*`, `goal_timeout` |
| `nav_stub_robot` (tools) | fake robot with lidar/IMU/leg odom/GT in a 2.5D world, for tests without the simulator | `world rooms/hall`, `leg_scale`, `leg_yaw_drift`, `wobble_amp` |
| `replay_report`, `goal_benchmark` (tools) | offline bag report; Nav2 goal-reaching benchmark with ground truth | see the module docstrings |

## Tests

```bash
cd ros2_ws/src/x30_navigation && python3 -m pytest -q -p no:cacheprovider -p no:anyio test      # ROS-free unit tests (ground fit, grid, frontiers, fusion)
python3 scripts/check_stack.py                                                                  # integration check against a running stack (exit code)
```

## Sim notes (2026-10-10)
`odom_tf:=false` for Gazebo (the sim owns odom->base_link); relay /cmd_vel_nav to /cmd_vel with `ros2 run x30_navigation cmd_relay` when x30_modes is absent. Evaluation tools: map_eval, ate_eval --png, goal_benchmark --gt-map. Open problem: intermittent Nav2 costmap stall, see docs/reports/navigation.md section 4.
