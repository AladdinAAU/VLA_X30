# x30_modes

Behaviour layer of the X30 VLN framework (ROS 2 Humble, Python): velocity mux, safety guard, mode manager, skill server.
Contract: `docs/INTERFACES.md` sections 1, 2, 5, 6. Report with measurements: `docs/reports/modes.md`.

## Nodes

| Node | In | Out | Notes |
|---|---|---|---|
| `cmd_vel_mux` | `/cmd_vel_teleop`, `/cmd_vel_nav`, `/x30/mode` | `/cmd_vel_raw` (20 Hz), `/x30/cmd_source` | teleop > nav, per-source timeout, teleop lockout, mode gating (idle blocks all, teleop mode only teleop) |
| `safety_guard` | `/cmd_vel_raw`, `/costmap_cloud` and/or `/scan`, `/x30/estop`, `/imu/data`, TF | `/cmd_vel` (20 Hz), `/x30/safety` (SafetyStatus) | slows/stops from clearance of the swept footprint along the commanded direction and the stopping distance; zero on stale command, stale lidar, e-stop, tilt; acceleration and hard speed limits |
| `mode_manager` | `/x30/navigate` (action), `/x30/set_mode`, `/odom`, `/x30/safety`, `/x30/cmd_source` | `/x30/mode` (latched) | modes idle, teleop, velocity_follow, object_nav, goal_pose, explore; timeout, progress watchdog, e-stop/safety/teleop aborts, path length from odometry |
| `skill_server` | TF, `/costmap_cloud`, `/map`, `/semantic_map/*`, camera | `/x30/skill/<name>` actions, `/x30/primitive` service, `/cmd_vel_nav` | skills of `configs/skills.yaml` |

Delegation: `velocity_follow` -> action `/x30/agent/follow`, `object_nav` -> `/x30/agent/run` (x30_agent, `Navigate` type).
External services used by skills (assumptions, see `docs/requests/modes-to-*.md`): `/nav/next_frontier`, `/semantic_map/query`, Nav2 `navigate_to_pose`.

## Run

```
ros2 launch x30_modes modes.launch.py profile:=sim            # or real; parameters in configs/modes/<profile>.yaml
ros2 action send_goal /x30/navigate x30_msgs/action/Navigate "{mode: goal_pose, goal: {header: {frame_id: map}, pose: {position: {x: 2.0}}}}"
ros2 action send_goal /x30/skill/move_relative x30_msgs/action/Skill "{name: move_relative, args_json: '{\"distance_cm\": 100}'}"
ros2 service call /x30/set_mode x30_msgs/srv/SetMode "{mode: teleop}"
```
Per-mode settings (timeouts, watchdog window, strategy): `configs/modes/{goal_pose,explore,velocity_follow,object_nav}.yaml`
(`sim:` / `real:` sections override the top level); task `params_json` overrides them again.

## Tests

* Without ROS (host or container): `cd ros2_ws/src/x30_modes && python -m pytest` (geometry, guard, mux, primitives, mode machine, registry, semantic).
* With ROS, fake robot, in the `x30-framework` image (domain 30): `test/ros_integration/start_stack.sh` then
  `test/ros_integration/run_in_container.sh [mux guard hazards primitives skills modes]`. The fake robot (`fake_robot.py`) integrates `/cmd_vel`
  with delay and lag, publishes odom/TF/cloud/scan/IMU/camera/map, a fake Nav2 (A* + pure pursuit), a semantic map and frontier service;
  `stub_agent.py` stands in for x30_agent; `fake_vlm.py` for the reasoner VLM. Results: `results/modes/*.json`.

## Notes

* The guard drops obstacle points inside the robot rectangle (own body, and the sensor-origin "invalid return" points of the lidar).
* Guard margins, speed limits and primitive speeds differ between sim and real (`configs/modes/sim.yaml`, `real.yaml`); the real values are estimates until
  the stopping distance is measured on the robot.
* `verify` uses `x30_common.vlm_client` and `configs/models.yaml` section `reasoner_vlm`.
