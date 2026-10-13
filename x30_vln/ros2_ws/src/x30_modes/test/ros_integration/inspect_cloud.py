"""Print statistics of one /lidar_points scan in the body frame (diagnostic for the guard height band and self filter)."""
import numpy as np
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from x30_modes.cloud import cloud_xyz

rclpy.init()
n = rclpy.create_node("inspect_cloud")
box = {}
n.create_subscription(PointCloud2, "/lidar_points", lambda m: box.setdefault("m", m),
                      QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
while "m" not in box:
    rclpy.spin_once(n, timeout_sec=0.5)
m = box["m"]
xyz = cloud_xyz(m, True)
r = np.hypot(xyz[:, 0], xyz[:, 1])
print("frame", m.header.frame_id, "points", xyz.shape[0], "range<0.1:", int((r < 0.1).sum()))
z = xyz[r > 0.3, 2]
print("z percentiles (r>0.3):", np.percentile(z, [0, 1, 5, 25, 50, 75, 95, 99, 100]).round(2))
for lo, hi in ((0.3, 0.6), (0.6, 1.0), (1.0, 2.0)):
    s = xyz[(r > lo) & (r <= hi)]
    if len(s):
        print(f"range {lo}-{hi}: {len(s)} pts, z median {np.median(s[:, 2]):.2f}, z min {s[:, 2].min():.2f}, z max {s[:, 2].max():.2f}")
side = xyz[(np.abs(xyz[:, 1]) < 0.5) & (xyz[:, 0] > -0.8) & (xyz[:, 0] < 0.8) & (r > 0.3)]
print("near-body box pts:", len(side), "x range", side[:, 0].min().round(2) if len(side) else None, side[:, 0].max().round(2) if len(side) else None,
      "y abs min", np.abs(side[:, 1]).min().round(2) if len(side) else None)
b = xyz[(r > 0.1) & (r < 1.2) & (xyz[:, 2] > -0.25) & (xyz[:, 2] < 0.7)]
print("band pts r<1.2:", len(b))
H, xe, ye = np.histogram2d(b[:, 0], b[:, 1], bins=[np.arange(-1.2, 1.21, 0.2), np.arange(-0.8, 0.81, 0.2)])
print("x rows -1.2..1.2 step .2, y cols -0.8..0.8 step .2")
print(H.astype(int))
inside = (np.abs(b[:, 0]) < 0.5) & (np.abs(b[:, 1]) < 0.32)
rest = b[~inside]
dx = np.maximum(np.abs(rest[:, 0]) - 0.5, 0)
dy = np.maximum(np.abs(rest[:, 1]) - 0.32, 0)
g = np.hypot(dx, dy)
print("after self filter:", len(rest), "points; nearest to the footprint:")
for i in np.argsort(g)[:8]:
    print(rest[i].round(2), "gap", g[i].round(3))
print("points with gap<0.1:", int((g < 0.1).sum()))
fw = b[(b[:, 0] > 0.56) & (np.abs(b[:, 1]) < 0.4)]
print("ahead band pts (x>0.56, |y|<0.4):", len(fw))
for i in np.argsort(fw[:, 0])[:6]:
    print(fw[i].round(2))
for name, sel, col in (("front x", (np.abs(b[:, 1]) < 0.4) & (b[:, 0] > 0.3), 0), ("rear -x", (np.abs(b[:, 1]) < 0.4) & (b[:, 0] < -0.3), 0),
                       ("left y", (np.abs(b[:, 0]) < 0.4) & (b[:, 1] > 0.2), 1), ("right -y", (np.abs(b[:, 0]) < 0.4) & (b[:, 1] < -0.2), 1)):
    v = np.abs(b[sel, col])
    h, e = np.histogram(v, bins=np.arange(0.3, 1.01, 0.05))
    print(name, "counts per 5 cm from 0.30:", h.tolist())
rclpy.shutdown()
