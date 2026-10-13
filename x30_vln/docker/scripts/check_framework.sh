#!/bin/bash
# Run inside the framework container: python deps, kiss_icp pipeline, ROS packages, rosdep.   (x30.ps1 exec framework /x30/docker/scripts/check_framework.sh)
. /usr/local/bin/x30-env
python3 - <<'PY'
import importlib
for m in ("numpy", "scipy", "cv2", "requests", "yaml", "pytest", "optuna", "trimesh", "open3d", "PIL", "httpx", "openai", "kiss_icp", "cv_bridge", "tf_transformations"):
    mod = importlib.import_module(m); print(f"  {m:20s} {getattr(mod, '__version__', 'ok')}")
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
cfg = KISSConfig(); cfg.mapping.voxel_size = 1.0
k = KissICP(cfg)
rng = np.random.default_rng(0)
def room(n=20000, shift=0.0):   # floor + 4 walls of a 10 x 8 m room seen from inside
    x = rng.uniform(-5, 5, n); y = rng.uniform(-4, 4, n); k = rng.integers(0, 5, n)
    p = np.stack([x, y, np.zeros(n)], 1)
    p[k == 1] = np.stack([np.full((k == 1).sum(), 5.0), y[k == 1], rng.uniform(0, 2.5, (k == 1).sum())], 1)
    p[k == 2] = np.stack([np.full((k == 2).sum(), -5.0), y[k == 2], rng.uniform(0, 2.5, (k == 2).sum())], 1)
    p[k == 3] = np.stack([x[k == 3], np.full((k == 3).sum(), 4.0), rng.uniform(0, 2.5, (k == 3).sum())], 1)
    p[k == 4] = np.stack([x[k == 4], np.full((k == 4).sum(), -4.0), rng.uniform(0, 2.5, (k == 4).sum())], 1)
    p[:, 0] -= shift; p[:, 2] += 0.5
    return p
for i in range(4):
    k.register_frame(room(shift=0.1 * i), np.linspace(0, 1, 20000))
print("kiss_icp odometry x after 4 frames moving 0.1 m each: %.2f m" % k.last_pose[0, 3])
print("kiss_icp register_frame ok")
PY
for p in nav2_bringup slam_toolbox pcl_ros robot_localization pointcloud_to_laserscan foxglove_bridge rosbag2_storage_mcap rmw_cyclonedds_cpp rviz2 tf2_tools; do
    ros2 pkg prefix $p >/dev/null 2>&1 && echo "  ros pkg $p ok" || echo "  ros pkg $p MISSING"
done
x30-rosdep --check 2>&1 | tail -n 3
