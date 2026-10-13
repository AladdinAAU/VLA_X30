"""map_eval: score the 2D `/map` against the ground-truth occupancy raster of the world, and draw the comparison.

  ros2 run x30_navigation map_eval --world rooms --out map_eval.json --png map_eval.png          # stub world
  ros2 run x30_navigation map_eval --gt-map <x30_worlds/worlds/lab_01/map.yaml> --out ... --png ...  # simulator world

The transform world <- map is taken from one simultaneous sample of /gt/pose (world frame) and TF map -> base_link, so it needs no
assumption about where the map frame started. Keep the robot still while the sample is taken. Metrics (tolerance `--tol`, default 0.15 m):
  occ_precision  occupied map cells that lie within tol of a ground-truth occupied cell
  occ_recall     ground-truth occupied cells (inside the explored area) that have an occupied map cell within tol
  free_error     free map cells that are ground-truth occupied (a wall drawn as free space)
  coverage       ground-truth free cells reachable from the robot that the map marks free
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy import ndimage
from tf2_ros import Buffer, TransformListener

from .goal_benchmark import load_gt_map
from .stub_world import make_world


def map_to_world_cells(m: OccupancyGrid, T_wm: np.ndarray, kind: int) -> tuple[np.ndarray, np.ndarray]:
    """World xy of the map cells with value `kind` (100 occupied, 0 free) and the grid as array."""
    data = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
    iy, ix = np.nonzero(data == kind) if kind == 0 else np.nonzero(data >= 65)
    r = m.info.resolution
    mx, my = m.info.origin.position.x + (ix + 0.5) * r, m.info.origin.position.y + (iy + 0.5) * r
    return np.column_stack((T_wm[0, 0] * mx + T_wm[0, 1] * my + T_wm[0, 2], T_wm[1, 0] * mx + T_wm[1, 1] * my + T_wm[1, 2])), data


def score(m: OccupancyGrid, T_wm: np.ndarray, blocked: np.ndarray, origin: np.ndarray, res: float, robot_xy: np.ndarray, tol: float) -> dict:
    dist_gt = ndimage.distance_transform_edt(~blocked) * res          # distance to the nearest GT occupied cell
    occ_xy, data = map_to_world_cells(m, T_wm, 100)
    free_xy, _ = map_to_world_cells(m, T_wm, 0)

    def lookup(xy: np.ndarray, field: np.ndarray, outside: float) -> np.ndarray:
        ij = np.floor((xy - origin) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < field.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < field.shape[0])
        out = np.full(len(xy), outside)
        out[ok] = field[ij[ok, 1], ij[ok, 0]]
        return out

    prec = float((lookup(occ_xy, dist_gt, 99.0) <= tol).mean()) if len(occ_xy) else 0.0
    free_err = float((lookup(free_xy, dist_gt, 99.0) < 0.5 * res).mean()) if len(free_xy) else 0.0
    # recall: GT occupied cells that are inside the part of the world the map has seen (known cells within 1 m of any known cell)
    map_occ = np.zeros_like(blocked)
    if len(occ_xy):
        ij = np.floor((occ_xy - origin) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < blocked.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < blocked.shape[0])
        map_occ[ij[ok, 1], ij[ok, 0]] = True
    dist_map = ndimage.distance_transform_edt(~map_occ) * res
    seen = np.zeros_like(blocked)
    ij = np.floor((np.vstack((occ_xy, free_xy)) - origin) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < blocked.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < blocked.shape[0])
    seen[ij[ok, 1], ij[ok, 0]] = True
    seen = ndimage.binary_dilation(seen, iterations=int(round(0.3 / res)))
    gt_seen = blocked & seen
    rec = float((dist_map[gt_seen] <= tol).mean()) if gt_seen.any() else 0.0
    # coverage of the reachable free space
    free_gt = dist_gt > 0.3
    lab, _ = ndimage.label(free_gt)
    ij0 = np.floor((robot_xy - origin) / res).astype(int)
    reach = lab == lab[ij0[1], ij0[0]] if lab[ij0[1], ij0[0]] > 0 else np.zeros_like(free_gt)
    mfree = np.zeros_like(blocked)
    if len(free_xy):
        ij = np.floor((free_xy - origin) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < blocked.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < blocked.shape[0])
        mfree[ij[ok, 1], ij[ok, 0]] = True
    cov = float(mfree[reach].mean()) if reach.any() else 0.0
    return {"tol_m": tol, "occ_precision": round(prec, 4), "occ_recall": round(rec, 4), "free_error": round(free_err, 4), "coverage": round(cov, 4),
            "map_occupied_cells": int(len(occ_xy)), "map_free_cells": int(len(free_xy))}


class Grab(Node):
    def __init__(self) -> None:
        super().__init__("map_eval")
        self.map: OccupancyGrid | None = None
        self.gt: PoseStamped | None = None
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/map", lambda m: setattr(self, "map", m), qos)
        self.create_subscription(PoseStamped, "/gt/pose", lambda m: setattr(self, "gt", m), 10)
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="rooms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-map", default="")
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--out", default="map_eval.json")
    ap.add_argument("--png", default="")
    args = ap.parse_args(argv)
    res = 0.05
    if args.gt_map:
        blocked, origin = load_gt_map(args.gt_map, res)
    else:
        blocked, origin = make_world(args.world, args.seed).raster(res)
    rclpy.init()
    node = Grab()
    t_end = time.monotonic() + 20.0
    tf = None
    while time.monotonic() < t_end and (node.map is None or node.gt is None or tf is None):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.gt is not None:
            try:
                tf = node.buf.lookup_transform("map", "base_link", rclpy.time.Time())
            except Exception:
                tf = None
    if node.map is None or node.gt is None or tf is None:
        raise SystemExit("no /map, /gt/pose or TF map->base_link")
    g, p = node.gt.pose, tf.transform
    yaw_g = math.atan2(2 * (g.orientation.w * g.orientation.z + g.orientation.x * g.orientation.y), 1 - 2 * (g.orientation.y ** 2 + g.orientation.z ** 2))
    yaw_m = math.atan2(2 * (p.rotation.w * p.rotation.z + p.rotation.x * p.rotation.y), 1 - 2 * (p.rotation.y ** 2 + p.rotation.z ** 2))
    dyaw = yaw_g - yaw_m
    c, s = math.cos(dyaw), math.sin(dyaw)
    t = np.array([g.position.x, g.position.y]) - np.array([c * p.translation.x - s * p.translation.y, s * p.translation.x + c * p.translation.y])
    T_wm = np.array([[c, -s, t[0]], [s, c, t[1]], [0, 0, 1]])
    result = score(node.map, T_wm, blocked, origin, res, np.array([g.position.x, g.position.y]), args.tol)
    result["world_from_map"] = {"x": round(float(t[0]), 3), "y": round(float(t[1]), 3), "yaw_deg": round(math.degrees(dyaw), 2)}
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        occ_xy, _ = map_to_world_cells(node.map, T_wm, 100)
        free_xy, _ = map_to_world_cells(node.map, T_wm, 0)
        h, w = blocked.shape
        fig, ax = plt.subplots(figsize=(11, 11 * h / w + 0.6), dpi=80)
        ax.imshow(np.where(blocked, 0.25, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1,
                  extent=[origin[0], origin[0] + w * res, origin[1], origin[1] + h * res])
        if len(free_xy):
            ax.scatter(free_xy[:, 0], free_xy[:, 1], s=1.8, c="#9ecae1", marker="s", linewidths=0)
        if len(occ_xy):
            ax.scatter(occ_xy[:, 0], occ_xy[:, 1], s=3.0, c="#d62728", marker="s", linewidths=0)
        ax.plot([g.position.x], [g.position.y], "g^", ms=9)
        ax.set_title(f"/map (red = occupied, blue = free) over ground truth (dark); precision {result['occ_precision']:.2f}, "
                     f"recall {result['occ_recall']:.2f}, coverage {result['coverage']:.2f}", fontsize=9)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(args.png)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
