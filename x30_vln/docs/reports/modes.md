# Report: x30_modes (behaviour layer)

Package `ros2_ws/src/x30_modes` (ROS 2 Humble, Python). Configs: `configs/skills.yaml`, `configs/modes/{sim,real}.yaml` and per-mode files. How to run: package `README.md`.
All numbers below were measured in this work; where something was not tested it says so.

## 1. What exists

| Part | Files | Status |
|---|---|---|
| Skill registry (10 skills: goto_pose, goto_object, approach, explore, search, turn_to, move_relative, scan, verify, stop) | `configs/skills.yaml` | in use by x30_agent (loader in `x30_agent/core/skills.py`); validated by `registry.py` |
| `cmd_vel_mux` | `cmd_vel_mux.py`, `mux_logic.py` | teleop > nav, per-source timeouts, teleop lockout, mode gating |
| `safety_guard` | `safety_guard.py`, `guard_logic.py`, `geometry.py` | swept-footprint clearance + stopping distance, e-stop/tilt/stale/no-lidar hazards, accel limits, `/x30/safety` |
| `mode_manager` | `mode_manager.py`, `mode_logic.py` | `/x30/navigate`, `/x30/set_mode`, `/x30/mode`; delegation to `/x30/agent/{follow,run}`; timeout, watchdog, safety/teleop/e-stop aborts, path length |
| `skill_server` | `skill_server.py`, `robot_api.py`, `skills/*` | actions `/x30/skill/<name>`, service `/x30/primitive` |
| Launch + params | `launch/modes.launch.py`, `configs/modes/*.yaml` | sim and real profiles (real: slower, larger margins; **real values are estimates**) |
| Tests | `test/*.py` (80 pytest), `test/ros_integration/*` (fake robot, stub agent, fake VLM, 79 checks; simulator check, 14 checks) | see section 2 |

Design points that matter: the guard is the only producer of `/cmd_vel`; the stopping bound is `v*latency + v^2/(2*decel) + margin <= clearance` with the same decel enforced
by the slew limiter; clearance is exact slab geometry for translations and a swept-pose test for rotations; points inside the robot rectangle (+ `self_margin`) and optional leg zones are ignored,
lone points (fewer than 3 in a 10 cm cell) and zero-intensity points (Livox invalid returns at the sensor origin) are dropped. Skill results always end with a "Next:" hint on failure.

## 2. Verification

### 2.1 Without ROS (pytest, host, 80 tests, 1 s)
geometry (clearance, grazing walls, rotation sweep, standoff poses, grids), guard (ramp, limits, stale, e-stop, tilt, no-lidar, stopping margin in closed loop at 0.2/0.4/0.6 m/s, speckle), mux (priority, timeouts, lockout, gating),
primitives (parser, translate/turn controllers on a lagged robot model: < 3 cm / < 3 deg), mode state machine (switch cancels the task, preemption, unresponsive task), watchdog, path length, registry validation, semantic selection, cloud parsing.

### 2.2 Fake robot in the `x30-framework` image (ROS_DOMAIN_ID 30): 79/79 checks
The fake robot integrates `/cmd_vel` with 50 ms delay and a 0.15 s velocity lag, publishes odom/TF (map and odom deliberately offset), cloud/scan/IMU/camera/map, a fake Nav2 (A* + pure pursuit on `/cmd_vel_nav`, so it goes through mux and guard), a semantic map, a frontier service; plus a stub agent and a fake VLM. Log: `results/modes/last_run.log`, data: `results/modes/integration_all.json`.

| Check | Result |
|---|---|
| Priority switching | nav passes; teleop wins within one tick; nav resumes 0.96 s after release (lockout 0.5 s); idle blocks all; teleop mode blocks nav |
| Guard stop, forward 0.2 / 0.4 / 0.6 m/s | standstill gap 0.204 / 0.194 / 0.194 m (margin 0.20), 0 collisions |
| Guard stop backward 0.3, sideways 0.3, 0.6 m/s past a chair | 0.197, 0.238, 0.19 m; 0 collisions |
| Box appearing 1.0 / 0.6 / 0.35 m ahead of the nose at 0.4 m/s | stop gap 0.193 / 0.195 / 0.182 m; 0 collisions |
| E-stop to zero `/cmd_vel` | 30-51 ms; tilt 50 deg stops; stale command: zero 0.82-0.86 s after the last command (0.5 s timeout + decel); max measured acceleration 0.84 m/s2 (limit 0.8) |
| Primitives (move 100 x3, 50 back, 30 left/right, 20; turn +-90, 45, 30, 180, 360) | max error 2.2 cm and 2.2 deg; mean 1.3 cm and 1.8 deg; `/x30/primitive` text and action forms work; bad text rejected with a message |
| Primitive into the pillar | stops after 80 of 300 cm with "stopped by the safety guard: obstacle 20 cm ahead. Next: ..." |
| goto_pose around the pillar | arrived 0.13 m from target, path 7.5 m, 0 collisions; obstacle/unreachable/invalid-argument cases return helpful messages |
| search -> goto_object -> approach -> verify | found chair id 1 after exploring one frontier; approach to 0.8 m gave 0.87 m measured nose gap, 0.1 deg bearing error (0.82 m by the skill's own estimate); verify yes/no/unsure through the fake VLM |
| rejected_ids, exclusivity, stop, action cancel | all as specified; second motion skill answers "Busy"; stop cancels goto_pose and halts |
| goto_pose blocked by an unmapped wall | aborted after 6 s with the guard message, 0 collisions |
| Mode manager | goal_pose result path length 7.66 m vs 7.63 m truth (mean error 0.4 m allowed); explore until exhausted; delegation to the stub agent relays feedback and turns; switching to teleop mid-task: task aborted ("mode switched ..."), velocity zero, 1.1 s; new goal preempts the old; timeout, no_progress, teleop takeover, e-stop and a pinned robot (safety) give their stop reasons; non-task mode returns error |

### 2.3 Real simulator, `kinematic` locomotion, ROS_DOMAIN_ID 30 (image `x30-sim`, kinematic): 14/14 checks
Run: `test/ros_integration/run_sim_check.sh` (framework container on `x30_net`); the guard reads the raw `/lidar_points` because Nav2/`/costmap_cloud` was not running (navigation was still in progress). Data: `results/modes/sim_check.json`, trace `approach_trace_0.6.json`, figure `docs/figures/modes/m3_sim_wall_approach.png`.

* Primitives against ground truth (`/gt/pose`): move 100 cm: 0.0 cm error, back 50: -0.7 cm, left 30: -0.8 cm; turn +90: -0.6 deg, -90: -0.8 deg, +45: +0.3 deg (profile `sim`, `coast_s` 0.12; with the first guess 0.25 the undershoot was 4 cm / 3 deg, fixed by calibration).
* Guard against the east wall (inner face x = 5.9 m from the SDF), command 0.2 / 0.4 / 0.6 m/s: robot stopped at x = 5.10 / 5.13 / 5.20, i.e. **nose-to-wall gap 0.25 / 0.22 / 0.15 m** (nominal margin 0.20 m: **0.05 m short at 0.6 m/s**, no contact). Cause: the raw 80k-point cloud arrives with more latency than the configured value; `latency_s` was raised to 0.45 for the sim profile but the margin is still not met at 0.6 m/s. With the navigation agent's `/costmap_cloud` the latency should be lower; re-measure then. `move_relative` 500 cm into the wall stops after 313-319 cm with the guard message; a 360 deg scan works.
* **goto_pose / approach in the simulator: sim verification pending** (needs Nav2 and `/map` from x30_navigation; not running during this work). They are verified only against the fake Nav2.

## 3. Findings from the simulator that changed the code
1. The merged cloud contains the robot's own legs beside the body (points at |y| 0.31-0.38 m, |x| about 0.88 m) and body points up to x about 0.6 m, as the real bag does (53k of 80k points within 0.8 m). Guard now has `self_margin`, a `legs_x/legs_y_min/legs_y_max` zone, and `min_cell_points` (lone returns and body speckle made the robot stop and go).
2. Livox invalid returns (intensity 0 at the sensor origin, per `docs/requests/sim-to-interfaces.md`) are dropped (`drop_zero_intensity`).
3. Primitive lag compensation (`coast_s`) must be calibrated per backend: sim 0.12, fake robot 0.2 (default), real 0.4 (guess).

## 4. Known issues and limits
* Real-robot parameters (`configs/modes/real.yaml`) are estimates from `docs/robot/x30-interfaces.md` section 5.1; stopping distance, latency and footprint must be measured on the robot before raising speeds. Nothing was run on the real robot.
* Guard margin shortfall in the simulator at 0.6 m/s (section 2.3). A blind zone exists: obstacles closer than `self_margin` (0.05 m) to the footprint are treated as body; leg-zone obstacles beside the body (|y| 0.28-0.45) are invisible to the guard when `legs_*` are enabled.
* goto_object/approach use the static `/map` for standoff poses and the lidar for the last metre; the object position error of the semantic map limits the approach bearing (not tested with the real semantic map).
* `explore`/`search` depend on assumed formats of `/nav/next_frontier` and `/semantic_map/query` (documented in `docs/requests/modes-to-navigation.md`, `modes-to-semantic_map.md`); only the fake services were exercised.
* `verify` was tested against a fake OpenAI server (parsing, yes/no/unsure paths); **a call to the real Qwen3-VL server was not made in this work**. `second look` for a single "no" is only suggested in the message, not executed.
* The soft e-stop topic is subscribed twice (volatile and latched) so a late start still sees a latched true; this prints an incompatible-QoS warning at start.
* `velocity_follow` / `object_nav` were tested against a stub; the real x30_agent servers were not run with this manager.

## 5. Requests filed
`docs/requests/modes-to-agent.md` (registry notes, delegation contract, `/x30/primitive`), `modes-to-navigation.md` (frontier service, `/cmd_vel_nav` remap, `/costmap_cloud` rate and self-filter), `modes-to-semantic_map.md` (query format).

## 6. Next steps
Run goto_pose/approach/explore in the simulator once Nav2 and the frontier service run; re-measure guard margins on `/costmap_cloud`; measure real stopping distance and latency and fill `real.yaml`; wire a real VLM check for `verify`; add the second-look behaviour.
