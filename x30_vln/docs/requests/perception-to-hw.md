# perception -> hw / sim (sensor providers)

What the lidar-to-image lifting needs from the robot-facing side (INTERFACES.md section 2 already says most of it):

1. **TF**: static `base_link -> camera_optical_frame` and `base_link -> lidar_link` (calibrated; an error of 2 degrees in pointing moves lifted
   points by 0.1 m at 3 m and drops 15-20 % of the points in a mask, 4 degrees is the limit for thin objects, beyond about 8 degrees thin objects in
   front of a far wall are lifted onto the wall; see docs/reports/perception.md), plus `odom -> base_link` with a smooth, low-latency pose (>= 30 Hz).
   Without the odometry TF the node falls back to one uncompensated scan. Optional `map -> odom`; without it detections are published in `odom`.
2. **Camera calibration** of the wide-angle USB camera (unknown today): `camera_info` with `plumb_bob` or `equidistant` distortion, or a calibration file
   (ROS camera_calibration YAML) passed as parameter `calibration_file`.
3. **Lidar timestamps**: the node relies on the header stamp being the *start* of the scan (verified on the recorded bag: header stamp equals the
   smallest per-point `timestamp`, which is float64 milliseconds). Per-point `timestamp` enables deskewing; if the sim does not provide it the node uses one pose per scan.
   Keep camera and lidar stamps on the same clock (the node compares them).
4. **Sim**: `/camera/depth/image_raw` (32FC1, metres, aligned with the RGB image) is only read with `depth_source: sim_depth`. The lidar needs returns
   at least 10 degrees below the horizontal if floor-standing objects should be lifted (the real merged cloud has none below about -10.5 degrees).
