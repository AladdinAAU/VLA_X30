"""replay_report: run the navigation pipeline libraries over a rosbag2 of /lidar_points and write numbers and figures.

  ros2 run x30_navigation replay_report --bag <dir> --out <dir> [--max-range 30 --voxel 0.4 --min-range 0.9]

Offline (no ROS graph, one process, single thread) so that timings are not disturbed by DDS or other nodes.
Outputs: report.json (odometry, timing, map statistics), trajectory.png, accumulated_map.png (top-down, all scans placed
with the odometry), scan_slice.png (levelled cloud, height classes, /scan polygon), occupancy.png.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..cloud_ops import HeightBand, classify, scan_from_points
from ..geometry import invert
from ..grid import GridParams, LogOddsGrid
from ..ground import GroundEstimator, GroundParams, footprint_transform, to_level_frame
from ..pointcloud import cloud_to_struct, voxel_downsample, xyz_of


def read_bag(uri: str, topic: str = "/lidar_points"):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name == topic:
            yield deserialize_message(data, PointCloud2)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-range", type=float, default=30.0)
    ap.add_argument("--min-range", type=float, default=0.9)
    ap.add_argument("--voxel", type=float, default=0.4)
    ap.add_argument("--no-deskew", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = args.max_range, args.min_range, not args.no_deskew
    cfg.mapping.voxel_size = args.voxel
    cfg.adaptive_threshold.min_motion_th, cfg.adaptive_threshold.initial_threshold = 0.05, 1.0
    cfg.registration.max_num_iterations = 100
    kiss = KissICP(cfg)
    est = GroundEstimator(GroundParams())
    grid = LogOddsGrid(GridParams())
    band = HeightBand()

    t_stage = {"kiss_icp": [], "ground_fit": [], "level_classify": [], "scan": [], "grid": []}
    poses, world_pts, n_points = [], [], []
    first_level = None
    height = []
    for msg in read_bag(args.bag):
        pts = cloud_to_struct(msg)
        xyz = xyz_of(pts)
        ok = np.isfinite(xyz).all(axis=1)
        xyz = xyz[ok]
        n_points.append(len(xyz))
        s = pts["timestamp"][ok].astype(np.float64)
        tn = (s - s.min()) / max(s.max() - s.min(), 1e-9)
        t0 = time.perf_counter()
        frame, _ = kiss.register_frame(xyz, tn)
        t_stage["kiss_icp"].append(time.perf_counter() - t0)
        pose = kiss.last_pose.copy()
        poses.append(pose)

        t0 = time.perf_counter()
        plane = est.fit(xyz)
        t_stage["ground_fit"].append(time.perf_counter() - t0)
        height.append(plane.height_above_ground)
        t0 = time.perf_counter()
        lev = to_level_frame(xyz, plane)
        ground, obstacle = classify(lev, band)
        t_stage["level_classify"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ranges = scan_from_points(lev, band, 720)
        t_stage["scan"].append(time.perf_counter() - t0)
        if first_level is None:
            first_level = (lev, ground, obstacle, ranges)
        t0 = time.perf_counter()
        ft = footprint_transform(plane)
        pose_map = pose @ ft  # sensor frame == base frame here; level frame pose in the odometry frame
        rot, trans = pose_map[:2, :2], pose_map[:2, 3]
        grid.integrate(trans.copy(), lev[obstacle][:, :2] @ rot.T + trans, lev[ground][:, :2] @ rot.T + trans)
        t_stage["grid"].append(time.perf_counter() - t0)
        world = frame @ pose[:3, :3].T + pose[:3, 3]
        world_pts.append(voxel_downsample(world, 0.1)[0].astype(np.float32))

    p = np.array(poses)[:, :3, 3]
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    rel = np.array([np.degrees(math.atan2(q[1, 0], q[0, 0])) for q in poses])
    m = grid.to_int8()
    report = {
        "scans": len(poses), "points_per_scan_mean": float(np.mean(n_points)),
        "final_position_m": p[-1].round(4).tolist(), "final_yaw_deg": float(rel[-1]),
        "path_length_m": float(steps.sum()), "max_step_m": float(steps.max()),
        "position_std_mm": (p.std(axis=0) * 1e3).round(2).tolist(),
        "max_displacement_from_start_mm": float(np.linalg.norm(p - p[0], axis=1).max() * 1e3),
        "ground_height_above_ground_m_mean_std": [float(np.mean(height)), float(np.std(height))],
        "ms_per_scan_mean": {k: round(float(np.mean(v)) * 1e3, 2) for k, v in t_stage.items()},
        "ms_per_scan_max": {k: round(float(np.max(v)) * 1e3, 2) for k, v in t_stage.items()},
        "ms_per_scan_total_mean": round(float(sum(np.mean(v) for v in t_stage.values())) * 1e3, 2),
        "map": {"shape": list(m.shape), "free_cells": int((m == 0).sum()), "occupied_cells": int((m == 100).sum()),
                "unknown_cells": int((m == -1).sum()), "free_area_m2": round(float((m == 0).sum()) * 0.0025, 1)},
        "kiss_icp": {"max_range": args.max_range, "min_range": args.min_range, "voxel": args.voxel, "deskew": not args.no_deskew},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    _figures(out, p, np.vstack(world_pts), first_level, m, grid)


def _figures(out: Path, p: np.ndarray, world: np.ndarray, first_level, m: np.ndarray, grid: LogOddsGrid) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    t = np.arange(len(p)) * 0.1
    for i, lab in enumerate("xyz"):
        ax[0].plot(t, (p[:, i] - p[0, i]) * 1e3, label=lab)
    ax[0].set(xlabel="s", ylabel="mm from start", title="stationary robot: odometry drift"); ax[0].legend()
    ax[1].plot((p[:, 0] - p[0, 0]) * 1e3, (p[:, 1] - p[0, 1]) * 1e3, ".-", lw=0.5)
    ax[1].set(xlabel="x mm", ylabel="y mm", title="xy track"); ax[1].set_aspect("equal")
    fig.tight_layout(); fig.savefig(out / "trajectory.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    w = world[(np.abs(world[:, 0]) < 30) & (np.abs(world[:, 1]) < 30) & (world[:, 2] > -3) & (world[:, 2] < 8)]
    ax.scatter(w[:, 0], w[:, 1], c=w[:, 2], s=0.15, cmap="viridis", rasterized=True)
    ax.plot(p[:, 0], p[:, 1], "r-", lw=1); ax.plot(p[0, 0], p[0, 1], "r*", ms=12)
    ax.set_aspect("equal"); ax.set_title(f"accumulated deskewed cloud, {len(p)} scans placed by KISS-ICP (colour = z, red = robot)")
    fig.tight_layout(); fig.savefig(out / "accumulated_map.png", dpi=80); plt.close(fig)

    lev, ground, obstacle, ranges = first_level
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    near = (np.hypot(lev[:, 0], lev[:, 1]) < 12)
    other = near & ~ground & ~obstacle
    ax[0].scatter(lev[other, 0], lev[other, 1], s=0.2, c="lightgrey", label="ignored (body, above band)")
    ax[0].scatter(lev[ground & near, 0], lev[ground & near, 1], s=0.2, c="tab:green", label="ground")
    ax[0].scatter(lev[obstacle & near, 0], lev[obstacle & near, 1], s=0.6, c="tab:red", label="obstacle band")
    ax[0].set_aspect("equal"); ax[0].legend(markerscale=10); ax[0].set_title("scan 0 in the levelled frame (z = height above ground)")
    az = -math.pi + (np.arange(len(ranges)) + 0.5) * 2 * math.pi / len(ranges)
    ok = np.isfinite(ranges)
    ax[1].plot(ranges[ok] * np.cos(az[ok]), ranges[ok] * np.sin(az[ok]), "k.", ms=2)
    ax[1].plot(0, 0, "r^"); ax[1].set_aspect("equal"); ax[1].set_title(f"/scan from cloud_to_scan (720 beams, {ok.sum()} with a return)")
    fig.tight_layout(); fig.savefig(out / "scan_slice.png", dpi=80); plt.close(fig)

    img = np.full(m.shape, 0.5)
    img[m == 0] = 1.0
    img[m == 100] = 0.0
    ys, xs = np.nonzero(m != -1)
    pad = 20
    fig, ax = plt.subplots(figsize=(12, 7))
    ext = [grid.origin[0], grid.origin[0] + m.shape[1] * grid.p.resolution, grid.origin[1], grid.origin[1] + m.shape[0] * grid.p.resolution]
    ax.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    ax.set_xlim(grid.origin[0] + (xs.min() - pad) * grid.p.resolution, grid.origin[0] + (xs.max() + pad) * grid.p.resolution)
    ax.set_ylim(grid.origin[1] + (ys.min() - pad) * grid.p.resolution, grid.origin[1] + (ys.max() + pad) * grid.p.resolution)
    ax.set_title("occupancy map (white free, black occupied, grey unknown)")
    fig.tight_layout(); fig.savefig(out / "occupancy.png", dpi=80); plt.close(fig)


if __name__ == "__main__":
    main()
