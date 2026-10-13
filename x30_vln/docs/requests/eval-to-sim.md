# Requests from EVAL to SIM (and WORLDS)

1. `/sim/set_robot_pose` answers `{"success": bool,...}` and returns only when the robot is ready again: the runner relies on this (it also checks `/x30/ready`). Currently matches sim_services.py.
2. **Contacts**: please publish collisions as `/sim/contacts` (`std_msgs/String`, one message per contact event, content free text/JSON). The runner counts events (merged within 1 s) as the
   `collisions` metric (`collision_basis: contacts`); without it collisions are derived from the GT grid footprint (`collision_basis: grid`), which needs the occupancy grid below.
3. **Occupancy grid per scene** (WORLDS): `ros2_ws/src/x30_worlds/worlds/<scene>/occupancy.yaml` in ROS `map_server` format (yaml + pgm/png), **world frame** (origin = world coordinates of the lower-left pixel),
   resolution <= 0.1 m, obstacles at robot height only (furniture the robot cannot pass counts as occupied). It gives the optimal path for SPL and the collision check. Without it SPL uses a straight-line optimum (pessimistic) and is flagged.
4. `/gt/objects` must keep `caption` = instance name (as now) and a footprint `size` (x, y, z, axis-aligned in the world frame); the evaluator measures object distance to that footprint box.
5. `/gt/pose`: also fill `pose.position.z` (used only for projecting GT boxes into camera frames).
6. Real time factor: one episode of 300 s sim time needs a usable RTF. The evaluator enforces the time limit in sim time (needs `/clock`) with a wall guard of 3x + 180 s.
