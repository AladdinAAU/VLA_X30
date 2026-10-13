#!/usr/bin/env python3
"""Lidar point density in a virtual camera image, measured on a recorded bag (/lidar_points only).

Question answered: how many lidar points fall on an object of a given size and distance, i.e. what is the smallest mask that
gives a usable 3D estimate? The bag has no images, so a virtual pinhole camera is placed at the lidar_link origin (or at an
offset) and every scan is projected into it. Angular density is what matters: a lidar samples directions, so the number of
points per image pixel does not depend on range; the surface density at distance d is points_per_pixel * (f / d)^2.

Needs `rosbags` (pip install rosbags). Example:
  python lidar_coverage.py ../../../../datasets/real/lab_2026-10-06 --hfov 90 --width 640 --height 480 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x30_perception.camera_model import CameraModel  # noqa: E402
from x30_perception.pointcloud import decode_xyz  # noqa: E402


def read_scans(bag: Path, topic: str):
    from rosbags.highlevel import AnyReader

    with AnyReader([bag]) as r:
        conns = [c for c in r.connections if c.topic == topic]
        for conn, _, raw in r.messages(conns):
            m = r.deserialize(raw, conn.msgtype)
            xyz, _ = decode_xyz(m.data, m.fields, m.point_step, m.is_bigendian)
            yield m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, xyz


def project(cam, xyz, R_cam_lidar, t_cam_in_lidar, min_range):
    p = (xyz - t_cam_in_lidar) @ R_cam_lidar.T
    rng = np.linalg.norm(xyz, axis=1)
    uv, valid = cam.project(p)
    ok = valid & cam.in_bounds(uv) & (rng >= min_range)
    return uv[ok], np.linalg.norm(p[ok], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag', type=Path)
    ap.add_argument('--topic', default='/lidar_points')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--hfov', type=float, default=90.0, help='horizontal field of view, degrees')
    ap.add_argument('--yaw', type=float, default=0.0, help='camera heading in the lidar frame, degrees (0 = lidar +x)')
    ap.add_argument('--pitch', type=float, default=0.0, help='camera pitch down, degrees')
    ap.add_argument('--offset', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='camera position in lidar_link, m')
    ap.add_argument('--min-range', type=float, default=0.6, help='drop own-body returns closer than this to the lidar, m')
    ap.add_argument('--accumulate', type=int, default=3, help='scans merged without motion compensation (0.1 s each)')
    ap.add_argument('--json', type=Path)
    a = ap.parse_args()

    cam = CameraModel.from_fov(a.width, a.height, np.radians(a.hfov))
    yaw, pit = np.radians(a.yaw), np.radians(a.pitch)
    fwd = np.array([np.cos(yaw) * np.cos(pit), np.sin(yaw) * np.cos(pit), -np.sin(pit)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.cross(fwd, right)
    R_cam_lidar = np.stack([right, down, fwd])  # rows: optical x, y, z axes expressed in the lidar frame
    off = np.array(a.offset)
    scans = [xyz for _, xyz in read_scans(a.bag, a.topic)]

    def density(n_acc):
        grid = np.zeros((4, 4))
        covered, total_pts, n_img = 0.0, 0, 0
        depth_all = []
        for i in range(0, len(scans) - n_acc + 1, n_acc):
            xyz = np.concatenate(scans[i : i + n_acc])
            uv, d = project(cam, xyz, R_cam_lidar, off, a.min_range)
            n_img += 1
            total_pts += len(uv)
            depth_all.append(d)
            cell = (np.clip((uv[:, 1] / a.height * 4).astype(int), 0, 3), np.clip((uv[:, 0] / a.width * 4).astype(int), 0, 3))
            np.add.at(grid, cell, 1)
            pix = np.unique(uv[:, 1].astype(int) * a.width + uv[:, 0].astype(int))
            covered += pix.size / (a.width * a.height)
        px_per_cell = a.width * a.height / 16
        return {
            'points_per_image': total_pts / n_img,
            'pixel_coverage': covered / n_img,
            'grid_points_per_px': (grid / n_img / px_per_cell).tolist(),
            'depth': np.concatenate(depth_all),
        }

    res = {
        'camera': {
            'width': a.width,
            'height': a.height,
            'hfov_deg': a.hfov,
            'fx': cam.fx,
            'yaw_deg': a.yaw,
            'pitch_deg': a.pitch,
            'offset_m': a.offset,
        },
        'scans': len(scans),
        'accumulate': a.accumulate,
    }
    for n in (1, a.accumulate):
        d = density(n)
        depth = d.pop('depth')
        res[f'acc{n}'] = d
        res[f'acc{n}']['depth_hist'] = {
            f'{lo}-{hi}m': int(((depth >= lo) & (depth < hi)).sum() / (len(scans) // n))
            for lo, hi in ((0.6, 1), (1, 2), (2, 4), (4, 7), (7, 12), (12, 40))
        }

    # surface density and the smallest usable object at several distances, from the central-region angular density
    n = a.accumulate
    grid = np.array(res[f'acc{n}']['grid_points_per_px'])
    rho_mid = float(grid[1:3, 1:3].mean())  # central 2x2 cells
    rho_low = float(grid[3].mean())  # bottom row of cells
    rho_min = max(float(grid[:3].min()), 1e-9)  # sparsest cell in the upper three rows
    tab = []
    for dist in (1.0, 3.0, 5.0, 10.0):
        ppm2 = rho_mid * (cam.fx / dist) ** 2
        tab.append(
            {
                'distance_m': dist,
                'pixel_footprint_mm': 1000 * dist / cam.fx,
                'points_per_m2': ppm2,
                'area_for_5pts_m2': 5 / ppm2,
                'area_for_20pts_m2': 20 / ppm2,
                'side_for_5pts_cm': 100 * np.sqrt(5 / ppm2),
                'side_for_20pts_cm': 100 * np.sqrt(20 / ppm2),
                'mask_side_px_for_5pts': float(np.sqrt(5 / rho_mid)),
                'mask_side_px_for_20pts': float(np.sqrt(20 / rho_mid)),
                'mask_side_px_for_5pts_sparsest_cell': float(np.sqrt(5 / rho_min)),
                'mask_side_px_for_20pts_sparsest_cell': float(np.sqrt(20 / rho_min)),
            }
        )
    res['central_points_per_px'] = rho_mid
    res['bottom_row_points_per_px'] = rho_low
    res['sparsest_upper_cell_points_per_px'] = rho_min
    res['table'] = tab

    xyz = np.concatenate(scans[:20])
    far = xyz[np.linalg.norm(xyz, axis=1) >= a.min_range]
    el = np.degrees(np.arcsin(far[:, 2] / np.linalg.norm(far, axis=1)))
    res['elevation_deg_percentiles_0_1_5_50_95_99_100'] = np.percentile(el, [0, 1, 5, 50, 95, 99, 100]).round(1).tolist()
    res['own_body_fraction_below_min_range'] = float((np.linalg.norm(xyz, axis=1) < a.min_range).mean())
    row_min_el = float(np.percentile(el, 1))
    res['image_row_of_lowest_elevation'] = float(cam.cy - cam.fy * np.tan(np.radians(row_min_el))) if a.pitch == 0 else None

    print(json.dumps(res, indent=1))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
