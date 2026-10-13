"""ate_eval: passive recorder that scores odometry/map poses against /gt/pose (ATE after rigid SE(2) alignment).

  ros2 run x30_navigation ate_eval --duration 300 --out ate.json [--sources /odometry/lidar /odometry/active]

Records /gt/pose (world frame) and every Odometry topic given plus the TF pose map->base_link. Samples are matched by
nearest time (<= 50 ms). For each source: ATE RMSE and max after the best 2D rigid alignment (Umeyama, no scale) over the
whole run, final-position error and relative drift (final error / path length) after aligning only the first 20 samples.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data


def align2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation (2x2) and translation minimising |R src + t - dst|."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    h = (src - ms).T @ (dst - md)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T
    return r, md - r @ ms


def score(est: np.ndarray, gt: np.ndarray) -> dict:
    """est, gt: (N, 3) x, y, yaw at matched times."""
    r, t = align2d(est[:, :2], gt[:, :2])
    e = est[:, :2] @ r.T + t
    err = np.linalg.norm(e - gt[:, :2], axis=1)
    dyaw = np.angle(np.exp(1j * (est[:, 2] + math.atan2(r[1, 0], r[0, 0]) - gt[:, 2])))
    k = min(20, len(est))
    r0, t0 = align2d(est[:k, :2], gt[:k, :2])
    e0 = est[:, :2] @ r0.T + t0
    path = float(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1).sum())
    final = float(np.linalg.norm(e0[-1] - gt[-1, :2]))
    return {"samples": len(est), "path_m": round(path, 2), "ate_rmse_m": round(float(np.sqrt((err ** 2).mean())), 4),
            "ate_max_m": round(float(err.max()), 4), "yaw_rmse_deg": round(float(np.degrees(np.sqrt((dyaw ** 2).mean()))), 3),
            "final_error_m": round(final, 4), "drift_percent": round(100.0 * final / max(path, 1e-6), 2)}


class Recorder(Node):
    def __init__(self, sources):
        super().__init__("ate_eval")
        self.gt: list[tuple[float, float, float, float]] = []
        self.est: dict[str, list] = {s: [] for s in sources}
        self.create_subscription(PoseStamped, "/gt/pose", self.on_gt, 50)
        for s in sources:
            if s != "map_tf":
                self.create_subscription(Odometry, s, lambda m, s=s: self.on_odom(s, m), qos_profile_sensor_data)
        self.tf_buffer = Buffer()
        self.tfl = TransformListener(self.tf_buffer, self)
        if "map_tf" in sources:
            self.create_timer(0.1, self.on_tf)

    @staticmethod
    def _yaw(q) -> float:
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def on_gt(self, m: PoseStamped) -> None:
        self.gt.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.pose.position.x, m.pose.position.y, self._yaw(m.pose.orientation)))

    def on_odom(self, s: str, m: Odometry) -> None:
        p = m.pose.pose
        self.est[s].append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.position.x, p.position.y, self._yaw(p.orientation)))

    def on_tf(self) -> None:
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return
        t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        v = tf.transform.translation
        self.est["map_tf"].append((t, v.x, v.y, self._yaw(tf.transform.rotation)))


def match(est: list, gt: list, tol: float = 0.05):
    g = np.array(gt)
    out_e, out_g = [], []
    for t, x, y, yaw in est:
        i = int(np.argmin(np.abs(g[:, 0] - t)))
        if abs(g[i, 0] - t) <= tol:
            out_e.append((x, y, yaw))
            out_g.append(g[i, 1:])
    return np.array(out_e), np.array(out_g)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--out", default="ate.json")
    ap.add_argument("--png", default="", help="also draw ground truth and the rigidly aligned estimates")
    ap.add_argument("--sources", nargs="+", default=["/odometry/lidar", "/odometry/active", "/odom", "map_tf"])
    args = ap.parse_args(argv)
    rclpy.init()
    node = Recorder(args.sources)
    t_end = time.monotonic() + args.duration
    try:
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    result = {}
    aligned = {}
    for s, samples in node.est.items():
        e, g = match(samples, node.gt)
        if len(e) > 30:
            result[s] = score(e, g)
            r, t = align2d(e[:, :2], g[:, :2])
            aligned[s] = (e[:, :2] @ r.T + t, g)
        else:
            result[s] = {"samples": int(len(e)), "error": "not enough matched samples"}
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    if args.png and aligned:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gt = np.array(node.gt)
        fig, ax = plt.subplots(figsize=(9, 6), dpi=80)
        ax.plot(gt[:, 1], gt[:, 2], "k-", lw=2.5, label="ground truth (/gt/pose)")
        for name, (e, g) in aligned.items():
            ax.plot(e[:, 0], e[:, 1], lw=1.0, label=f"{name}: ATE {result[name]['ate_rmse_m']:.2f} m, drift {result[name]['drift_percent']:.1f} %")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_title("Trajectories after rigid SE(2) alignment to the ground truth")
        fig.tight_layout()
        fig.savefig(args.png)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
