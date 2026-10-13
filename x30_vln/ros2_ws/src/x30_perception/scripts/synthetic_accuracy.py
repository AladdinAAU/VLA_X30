#!/usr/bin/env python3
"""Accuracy of the lidar lifting on the synthetic room (pure numpy, no ROS, no GPU).

Sweeps: object distance, object size, extrinsic yaw error, robot motion with/without motion compensation, and reports
point counts, position error and the bearing_only rate. Ground-truth masks come from the renderer, so this isolates the
lifting from the segmenter. Targets are flat (billboards): the error is the error against the visible plane; a real object
with depth has its centroid on the visible surface, about half of its depth in front of its geometric centre.

  python synthetic_accuracy.py --json ../../../results/perception/synthetic_accuracy.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x30_perception import synthetic as syn  # noqa: E402
from x30_perception.camera_model import CameraModel  # noqa: E402
from x30_perception.geometry import apply, euler_to_matrix, invert, make_transform  # noqa: E402
from x30_perception.lidar_projector import ProjectorParams, lift_mask, merge_scans, project_points, select_scans  # noqa: E402

CAM = CameraModel.from_fov(640, 480, 1.5184)
T_LIDAR_OPT = syn.optical_pose((0.5, 0.0, -0.1))  # camera optical frame in the lidar frame
P = ProjectorParams()


def _yaw(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def run(scene, lidar_fn, t_img=0.15, compensate=True, ext_err_deg=0.0, n_scans=3, seed=1, pts=80000):
    rng = np.random.default_rng(seed)
    scans = [syn.simulate_scan(scene, lidar_fn, t0, pts, rng=rng, el_range_deg=(-7, 52)) for t0 in (0.0, 0.1, 0.2)]
    scans = select_scans(scans, t_img, n_scans)
    pose_fn = lidar_fn if compensate else (lambda t: lidar_fn(t_img))
    pw, _ = merge_scans(scans, pose_fn, 3, P.min_range_m)
    T_cam = lidar_fn(t_img) @ T_LIDAR_OPT
    T_est = T_cam.copy()  # the calibration the lifting believes in
    if ext_err_deg:
        T_est[:3, :3] = T_cam[:3, :3] @ euler_to_matrix(
            0.0, np.radians(ext_err_deg), 0.0
        )  # about the optical y axis = horizontal pointing error
    t = time.perf_counter()
    proj = project_points(CAM, apply(invert(T_est), pw), P)
    t_proj = time.perf_counter() - t
    _, _, labels = syn.render(scene, CAM, T_cam, with_texture=False)
    out, t_lift = [], 0.0
    for i, b in enumerate(scene.billboards):
        mask = labels == i
        if mask.sum() < 10:
            continue
        t = time.perf_counter()
        lift = lift_mask(proj, mask, CAM, P, T_out_cam=T_est)
        t_lift += time.perf_counter() - t
        c = np.array(b.center)
        out.append(
            {
                'label': b.label,
                'mask_px': int(mask.sum()),
                'in_mask': lift.n_points,
                'used': lift.n_used,
                'bearing_only': lift.bearing_only,
                'plane_err_m': None if lift.bearing_only else abs(float(lift.centroid[0] - c[0])),
                'pos_err_m': None if lift.bearing_only else float(np.linalg.norm(lift.centroid - c)),
            }
        )
    return out, {'project_ms': 1000 * t_proj, 'lift_ms_per_object': 1000 * t_lift / max(len(out), 1)}


def static_lidar(t):
    return make_transform(np.eye(3), (0.0, 0.0, 0.7))


def summarize(rows):
    ok = [r for r in rows if not r['bearing_only']]
    return {
        'n': len(rows),
        'bearing_only': len(rows) - len(ok),
        'points_median': float(np.median([r['used'] for r in ok])) if ok else 0,
        'plane_err_median_m': float(np.median([r['plane_err_m'] for r in ok])) if ok else None,
        'pos_err_median_m': float(np.median([r['pos_err_m'] for r in ok])) if ok else None,
        'pos_err_max_m': float(np.max([r['pos_err_m'] for r in ok])) if ok else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=Path)
    a = ap.parse_args()
    res = {}
    # 1. distance x size (square object whose centre is at lidar height, so it is inside the lidar's vertical FOV)
    table = []
    for size in (0.2, 0.4, 0.8):
        for d in (1.5, 3.0, 5.0, 8.0, 12.0):
            sc = syn.Scene([syn.Billboard('obj', (d, 0.0, 0.7), size, size)], room=(-6, 20, -8, 8, 3.0))
            rows, tm = run(sc, static_lidar)
            table.append({'size_m': size, 'distance_m': d, **(rows[0] if rows else {'mask_px': 0, 'bearing_only': True}), **tm})
    res['distance_size'] = table
    # 2. extrinsic yaw error, 4 objects at 3-8 m
    scene = syn.Scene(
        [
            syn.Billboard('a', (3.0, 0.8, 0.7), 0.6, 0.8),
            syn.Billboard('b', (5.0, -1.2, 0.8), 0.8, 1.0),
            syn.Billboard('c', (8.0, 1.5, 0.9), 1.0, 1.2),
            syn.Billboard('d', (4.0, -0.2, 0.7), 0.5, 0.6),
        ]
    )
    res['extrinsic_yaw_error'] = {str(e): summarize(run(scene, static_lidar, ext_err_deg=e)[0]) for e in (0.0, 0.5, 1.0, 2.0)}

    # 3. motion: 0.6 m/s forward, 0.5 rad/s turn, with and without compensation
    def moving(t):
        yaw = 0.5 * t
        return make_transform(_yaw(yaw), (0.6 * t * np.cos(yaw), 0.6 * t * np.sin(yaw), 0.7))

    res['motion'] = {
        name: summarize(run(scene, moving, compensate=c)[0]) for name, c in (('compensated', True), ('uncompensated', False))
    }
    # 3b. thin object in front of a far wall: extrinsic error pulls wall points into the mask edge
    post = syn.Scene([syn.Billboard('post', (3.0, 0.0, 0.7), 0.3, 1.2)], room=(-6, 14, -8, 8, 3.0))
    res['thin_post_extrinsic_yaw_error'] = {
        str(e): summarize(run(post, static_lidar, ext_err_deg=e)[0]) for e in (0.0, 1.0, 2.0, 4.0, 8.0)
    }
    # 4. latency of the numpy part for a typical frame (4 objects)
    _, tm = run(scene, static_lidar)
    res['timing_host_cpu'] = tm
    print(json.dumps(res, indent=1))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
