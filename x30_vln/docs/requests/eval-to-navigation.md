# Requests from EVAL to NAVIGATION

Between episodes the robot is teleported by the simulator. Please provide Json services (request `{}`), each returning `ok=true` when done:
- `/mapping/reset`: clear the 2D occupancy map / local costmaps / accumulated 3D map.
- `/slam/reset`: restart lidar odometry and SLAM so that the `map` frame origin is the robot pose at this moment (TF `map -> base_link` identity-ish at the start). The evaluator measures the map-to-world
  transform from TF and GT at the start of every episode and uses it for goal conversion and for the ATE/semantic-map metrics, so any fixed `map` frame works, but a stale map from the previous episode does not.
If a single service does both, expose it as `/mapping/reset` and tell me; the list lives in `configs/eval/eval.yaml: reset.services` (optional entries are skipped with a note when missing).
