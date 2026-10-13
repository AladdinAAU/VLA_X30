# Simulator figures (x30_sim)

All made by `scripts/sim/make_figures.py` from sim outputs (`scripts/sim/grab_frame.py`, `capture_sim_scans.py`, `measure_rtf.py`).

- `rgb_thermal.png`: left, the simulated RGB camera (640x480) in `default_arena` looking at the warm "person" cylinder; right, the thermal camera (384x288, mono8 display, auto-gain, hot = bright) of the same moment: only the 309 K object stands out from the 22 C ambient.
- `lidar_real_vs_sim.png`: top-down view of one merged `/lidar_points` scan, coloured by height. Left: the real bag (lab, robot lying down); middle: simulation in `lab_room` (the world tuned for the comparison); right: simulation in the 12 x 10 m `default_arena`. The gap beside the robot (red triangle) is the body/leg shadow of the four tilted sensors.
- `tf_tree.png`: TF tree of the running simulator (`odom -> base_link -> base -> TORSO -> sensors and legs`), from `ros2 run tf2_tools view_frames`.
- `soak_rtf_memory.png`: 12-minute kinematic run with every sensor on: real-time factor about 0.98, Gazebo resident memory flat at about 517 MB, ROS nodes about 605 MB, topic rates (wall clock).
- `lidar_histograms.png`: range, elevation and height histograms of the real bag against the simulation in `lab_room` (from `scripts/sim/compare_lidar_to_bag.py`).
