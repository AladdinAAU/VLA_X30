# sim -> interfaces (INTERFACES.md), for the main session

1. **Lidar frame names.** Section 1 names `lidar_fl/fr/rl/rr_link` (front-left/right, rear-left/right). The real layout (manual p.67, bag) is two sensors on the
   centre line, one up and one down, at each end. The simulator keeps the names and maps `fl = front-up`, `fr = front-down`, `rl = rear-up`, `rr = rear-down`.
   Suggest renaming to `lidar_front_up_link`, `lidar_front_down_link`, `lidar_rear_up_link`, `lidar_rear_down_link` (and the topics `/lidar/<id>/points`).
2. **Zero points in `/lidar_points`.** The real cloud (bag) keeps invalid returns as points AT THE SENSOR ORIGIN with intensity 0 (about 53 % of each scan):
   the lidar-odometry and mapping code must drop points with intensity 0 whose position equals a sensor mount, or range < 0.1 m. The simulator reproduces this
   (config `lidar.invalid_as_origin`). `timestamp` is milliseconds (bag: wall-clock epoch ms; sim: sim-time ms) and spans the 100 ms scan.
3. **TF `odom -> base_link`.** The simulator publishes it (planar, from `/odom`). Do not start a second publisher (`/odometry/active`) while `x30_sim` runs unless
   the simulator's TF is disabled; a launch argument for that can be added if the framework wants the lidar odometry to own the TF.
4. **IMU, odom rates.** Sim `/imu/data` 200 Hz (real 200 Hz), `/odom` 50 Hz (real leg odometry 200 Hz).
5. **Thermal default rate** is 5 Hz at 384x288 (INTERFACES 5-10 Hz): every thermal frame costs simulation real time.
