# Navigation report (x30_navigation)

Status 2026-10-10. Package `ros2_ws/src/x30_navigation`, configs `configs/navigation/`, figures `docs/figures/navigation/` (captions in CAPTIONS.md), earlier replay evidence `docs/reports/navigation/` (report.json, images). Humble, `x30-framework` image (dev runs used the dev container `x30-nav-fw`).

Summary: odometry, 2D mapping and frontier exploration work and were measured. Nav2 goal reaching works in the stub (17/20) after the costmap-stall fix, but **fails in the Gazebo lab (0/20) because the map inside rooms gets corrupted** (section 4). Numbers below are what was measured; nothing is extrapolated.

## 1. What exists
Nodes: `lidar_odometry` (KISS-ICP, deskew), `odom_selector` (gt|leg|lidar|fused, single `odom->base_link` TF, watchdog, `/x30/health`), `cloud_leveler` (ground fit, `base_footprint`, `/nav/cloud_level` with flags 1 ground / 2 obstacle / 4 body hit), `cloud_to_scan` (`/scan`), `obstacle_cloud` (`/costmap_cloud`, also published when empty), `occupancy_mapper` (`/map`, growing log-odds grid), `frontier_explorer` (service `/nav/next_frontier` in the format of docs/requests/modes-to-navigation.md, `/nav/explore/start|stop`, `/nav/explore_status`). Nav2: MPPI Omni + Smac2D, BT without BackUp, output on `/cmd_vel_nav`.
Tools: `nav_stub_robot` (fake robot and world), `goal_benchmark`, `ate_eval`, `map_eval`, `replay_report`, `cmd_relay` (/cmd_vel_nav -> /cmd_vel for sim runs without x30_modes). 19 ROS-free tests pass (`python3 -m pytest test`).

## 2. How to run
- Stub (no simulator): `nav_stub_robot --ros-args -p cmd_topic:=/cmd_vel_nav -p world:=rooms -p start:="[1.2,3.5,0.0]"` and `ros2 launch x30_navigation navigation.launch.py profile:=sim`.
- Gazebo sim (Jazzy `x30-sim`, same `ROS_DOMAIN_ID` in both containers, both on `x30_net`): `ros2 launch x30_navigation navigation.launch.py profile:=sim use_sim_time:=true odom_tf:=false odom_source:=leg` plus `ros2 run x30_navigation cmd_relay`, then `/nav/explore/start`. `odom_tf:=false` because the sim publishes `odom->base_link`; the selector then only publishes `/odometry/active`. Reset the robot with `/sim/set_robot_pose` before a run.
- Real: `profile:=real` (odom_source fused, fallback to lidar). Replay: `profile:=replay nav2:=false static_lidar_tf:=true`.
- Evaluation: `ate_eval --out ate.json --png ate.png`, `map_eval --gt-map <world map.yaml> --png map.png`, `goal_benchmark --gt-map <map.yaml> --goals 20 --explore-first`.
- A Humble container attached to two docker networks must pin the DDS interface (Cyclone `<Interfaces><NetworkInterface name="eth1"/>`), otherwise it does not see the sim.

## 3. Parameters
All in `configs/navigation/*.yaml`, documented inline. Changed in this round: `cloud_leveler.band.body_half_x/y` 0.70/0.40 (the sim legs return points at x +-0.49..0.63, y +-0.32, inside the footprint inflation), local costmap `raytrace_min_range` 0.7 (obstacles closer than the body box are invisible, so they must not be cleared), `frontier_explorer.max_consecutive_empty` 8. Launch argument `odom_tf` added.

## 4. Measurements
Stub world `rooms` (16 x 10 m, 1 m doors, robot 1.0 x 0.46 m), explore first, then goals:
- Exploration reached 99.8 % (run t3, before the BLAS fix) and 100 % (run t7, after) of the reachable area in 170-180 s, 42-45 m path, 0 failed goals. In 4 other runs exploration stalled (coverage 20-68 %, 4-11 failed goals): the stub loop is **flaky**.
- Goal phase, run t7 (aborted by me): 2 of 12 goals succeeded (efficiency 0.96 and 0.89, 0 collisions); goal 3 timed out with 2 contacts in a 1 m door; afterwards every goal aborted. Success rate 2/12 = 17 %. No complete 20-goal run exists.
- Lidar odometry vs ground truth (t7, 105 m): ATE RMSE 0.13 m (max 4.1 m, one outlier), yaw RMSE 3.3 deg, drift 0.17 % of path.
- Map quality after a scripted drive without Nav2: precision 0.69, recall 0.77 at 0.15 m (`stub_rooms_map_vs_gt.png`).

Gazebo `lab_01` (28 x 14 m, 9 rooms, kinematic locomotion, RTF 0.7-0.8):
- 2D map vs ground truth after 190 s of exploration: occupied precision 0.94, recall 0.79, free-cell error 4 % (`sim_lab01_map_vs_gt_exploring.png`).
- Frontier exploration (run s5, all fixes): 85 % of the reachable area in 360 s, 35 m path, 15 goals sent, 1 failed. Run s1 (before the fixes): 66 % in 190 s.
- Goal reaching (run s5): of the first 10 goals 1 succeeded (5.8 m, efficiency 0.985, no collision) and 9 aborted at once. This is **not a usable success rate** because of the stall below; the run was cut off and has no complete summary.
- Lidar-odometry ATE in the sim: **not measured**. Several recorder runs were lost (recorder killed before writing, then the sim container was taken over by another agent running a different world, so the last attempt got no `/gt/pose`). The node runs at 10 Hz, 4-18 ms per scan.

### Fixed: stale costmap robot pose
Symptom: both Nav2 costmaps stopped updating their robot pose (footprint stamp frozen from the first seconds), plans started at an old position, every goal aborted. Bisect (fp2 probe of `published_footprint` age): not DDS (also with FastDDS, WhcHigh 8 MB did not help alone), not TF publishers (fresh stamps), not clock. With `observation_sources: costmap_cloud` only, both costmaps stayed fresh in 5 of 5 runs (2 Cyclone, 3 FastDDS); with `scan` in the list they froze in 4 of 5. Cause: the LaserScan observation source of the obstacle layers (laser_geometry TF lookups inside the costmap callbacks block the costmap thread). Fix: `/scan` removed from both costmaps (`configs/navigation/nav2_params.yaml`); `/costmap_cloud` carries the obstacles; `/scan` stays for the mapper and slam_toolbox.
Re-measured after the fix, stub `rooms`: exploration 100 % in 170 s (49 m, 0 failed); **20-goal benchmark 17/20 = 85 % success**, path efficiency 0.98, 2 collisions, 29.6 s mean per goal.
Sim `lab_01` (own sim container, domain 21): exploration no longer stalls (12 goals, 0 failed, 34 m) but stops at 67 % of the reachable area ("complete" with unreachable frontiers), and the 20-goal benchmark gave **0/20**: the `/map` inside lab 5 becomes corrupted (occupied precision falls to 0.54, scribbled walls), so the planner finds no path. A scripted in-place spin at 0.5 rad/s does not corrupt the map (precision 0.94 before and 0.92 after), so rotation timing is not the cause; suspects are collisions that tilt the gravity-free kinematic robot and furniture clutter in rooms. Not resolved. Sim ATE still not measured (recorder file not written in that run).
Other causes found and fixed earlier: BLAS threads (cloud_leveler at 1000 % CPU), robot wedged by legs in the costmap, free-space carving through walls nearer than the body box (BODY flag, unit-tested), frontier "complete" after 3 empty answers.

## 5. Limits
- A 1.0 m robot in 1.0 m doors collides when it turns inside the door frame; benchmark goals keep 0.6 m clearance.
- Obstacles closer than 0.4 m laterally or 0.7 m longitudinally are invisible (body box); they are only remembered from earlier views.
- The real robot has no lidar returns below about -10.5 deg elevation and 47 % of the real cloud are no-return points at the origin; they are dropped (intensity 0) before use. Real leg geometry in the cloud is unknown; the 0.70 x 0.40 box was tuned on the sim.
- `/map` is a growing 0.05 m grid (600 x 600 to 1003 x 600 cells in the lab); large messages stress DDS (request: docs/requests/navigation-to-infra.md).
- Real-robot navigation was not run; only bag replay (lidar odometry and map, see navigation/report.json).

## 6. Next steps
Find the in-room map corruption in the sim (check robot tilt on contact, odometry vs scan alignment in rooms); rerun the 20-goal benchmark in stub and lab_01 and the sim ATE with the sim to ourselves; narrow-door handling (heading-aligned paths); avoid frontier goals on wall gaps.


## Waypoint navigation
Service `/nav/follow_waypoints` (x30_msgs/Json) of node `waypoint_runner`: request `{"waypoints": [[x,y,yaw?],...], "radius": 0.4, "timeout_s": 120, "stop_on_failure": false}` in the `map` frame; response per waypoint `{reached, status, final_error_m, duration_s}`; progress JSON on `/nav/waypoint_progress`. Both backends serve the same `navigate_to_pose` action, so the runner (and the explorer) do not care which one runs. Backend: launch argument `planner_backend:=nav2|direct` (default **nav2**).
- `direct` (`direct_nav.py`, `direct_navigator.py`): Dijkstra on /map with clearance 0.42 m (fallback 0.3 m, then unknown cells as free, then a straight line), pure pursuit with heading control, turn-in-place above 0.6 rad, slowdown inside 1.2 m of the goal, replanning every 2 s, wedge recovery (no motion for 3 s: reverse 0.4 m), output `/cmd_vel_nav`. No obstacle avoidance of its own; the safety guard (x30_modes) or `cmd_relay` (test) owns obstacle stopping.
- Run: `docker exec x30-wp-fw bash /x30/scripts/sim/wp_run.sh <nav2|direct> <explore_s> <n_goals>` (launch + cmd_relay + explore + `waypoint_bench`), sim: `ros2 launch x30_sim sim.launch.py world:=replica_apartment_1 locomotion:=kinematic`, domain 20, `GZ_PARTITION=wp`. Unit tests (no ROS): `python -m pytest test/test_direct_nav.py` (3 pass; 16 pass together with grid and odom tests).
- Fixed on the way: `navigation.launch.py` called `perform()` without a context and lacked imports (rewritten with an OpaqueFunction); the frame of the benchmark goals (map origin = robot pose when the odometry is reset, `--map-origin`).

Measured in Gazebo `replica_apartment_1`, kinematic locomotion, RTF about 0.3-0.4, leg odometry, 120 s of frontier exploration first, then random waypoints (>= 3 m path distance, tolerance 0.4 m, 120 s timeout), ground truth from `/gt/pose` and the world raster (`waypoint_bench`):

| backend | waypoints | reached | mean path efficiency (reached) | collisions (GT clearance < 0.15 m) |
|---|---|---|---|---|
| direct | 12 | 1 (8 %) | 0.71 | 4 |
| nav2 | 10 | 2 (20 %) | 1.0 (capped) | 7 |

An earlier direct run without recovery and clearance 0.35 reached 0 of 12. **Neither backend is reliable in this world.** Observed: the 1.0 x 0.46 m kinematic body wedges against the scan mesh (rotation blocked while translation is free; reversing frees it), the robot gets stuck inside furniture-dense rooms, and exploration ends early (coverage 0.43). The direct tracker also ran with its heading estimate disagreeing with ground truth by up to 1.5 rad in one probe (leg odometry yaw drift or a stale TF; not isolated), which alone explains many aborted goals. Nav2 in the same world mostly timed out or aborted at once (costmaps over the corrupted rooms of the map). The `lab_01` map-corruption cause was not investigated in this round.
**Default: nav2**, because it avoids obstacles itself and had the better rate here; `direct` is the fallback for open space or when Nav2 stalls, and it needs the safety guard. Not measured: the direct backend on the stub world or on `lab_01`, Nav2 with the x30_modes safety guard instead of `cmd_relay`. The results are in `results/waypoints/bench_*.json`; figure `waypoints_replica_apartment_1.png`.
