# modes -> navigation

1. **Frontier service** `/nav/next_frontier` (`x30_msgs/srv/Json`), used by skills explore/search:
   request `{"strategy": "nearest|largest|informative|bias", "bias": [x, y] | null, "exclude": [[x, y], ...]}` (map frame; `exclude` = goals that
   failed, skip frontiers within 0.5 m of them); response `{"found": bool, "goal": {"x", "y", "yaw"}, "size": cells, "distance_m", "remaining": n, "reason": ""}`.
   `found: false` means exploration is exhausted. This format is an assumption (fake service in x30_modes/test); tell us if yours differs.
2. **Nav2 velocity output** must be remapped to `/cmd_vel_nav` (not `/cmd_vel`); the mux and guard own `/cmd_vel`.
3. **`/costmap_cloud`** (obstacle points without ground) is the safety guard's input; any frame with TF to `base_link` works. Please publish it at >= 5 Hz,
   also when empty (an absent or stale cloud stops the robot: `no_lidar`). `/scan` is the fallback (`obstacle_source: scan|both`).
4. The skill `goto_pose` uses action `navigate_to_pose` and the static `/map` (standoff poses), falling back to `/global_costmap/costmap`.
