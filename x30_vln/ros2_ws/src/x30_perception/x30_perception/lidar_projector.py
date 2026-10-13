"""Lift 2D instance masks to 3D by projecting lidar points into the camera image (pure numpy, no ROS imports).

The X30 has no depth camera. Pipeline implemented here:

1. `select_scans` / `merge_scans`: take the N lidar scans nearest to the image time and express them in a common, smoothly
   moving frame (odom) using the pose of the sensor at the time of each scan (or of each time slice of a scan: motion
   compensation). The caller supplies `pose_fn(t) -> 4x4` (normally a TF lookup).
2. The merged cloud is moved into the camera frame at the *image* time (one more rigid transform), so scan-to-image time
   offsets and robot motion are compensated as far as the odometry is good.
3. `project_points`: pinhole (+distortion) projection, image bounds, range limits and an occlusion test: a point that
   lies noticeably behind another point within a small pixel window is dropped (lidar points seen through gaps, e.g.
   wall points between chair legs, must not land on the chair).
4. `lift_mask`: points whose pixel is inside the (slightly eroded) mask are split into depth clusters, the foreground
   cluster with enough support is kept, stray points are removed by Euclidean clustering, and a centroid (median) and an
   extent (percentile range) are computed in the requested output frame. With fewer than `min_points` points the result is
   `bearing_only`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .camera_model import CameraModel
from .geometry import apply


@dataclass
class ProjectorParams:
    min_range_m: float = 0.6  # drops the robot's own body (57 % of the recorded cloud lies within 0.5 m)
    max_range_m: float = 30.0
    occlusion_window_px: int = 5  # odd; <= 1 disables the occlusion test. In working-resolution pixels
    occlusion_abs_tol_m: float = 0.10
    occlusion_rel_tol: float = 0.04  # tolerance = abs + rel * depth
    mask_erode_px: int = 2  # shrink masks by this much before collecting points (edge bleed)
    erode_min_area_px: int = 400  # do not erode tiny masks
    min_points: int = 5  # fewer lidar points in the mask -> bearing_only
    depth_gap_abs_m: float = 0.25  # split the in-mask points where the sorted depth jumps by abs + rel * depth
    depth_gap_rel: float = 0.04
    cluster_keep_frac: float = 0.4  # a nearer cluster wins over the biggest if it has >= this fraction of its points
    euclid_eps_m: float = 0.30  # 3D connectivity radius = eps + eps_rel * median depth
    euclid_eps_rel: float = 0.03
    extent_percentiles: tuple = (2.0, 98.0)
    max_points_per_mask: int = 3000  # random subsampling cap (speed)


@dataclass
class Scan:
    stamp: float  # seconds, start of the scan
    xyz: np.ndarray  # (N,3) float32, sensor frame
    t_rel: np.ndarray | None = None  # (N,) seconds since `stamp` for every point, if the sensor provides it
    duration: float = 0.1


@dataclass
class ProjectedPoints:
    u: np.ndarray  # (N,) int pixel column in the working image
    v: np.ndarray  # (N,) int pixel row
    depth: np.ndarray  # (N,) float, z in the camera frame
    xyz: np.ndarray  # (N,3) float, camera frame
    stats: dict = field(default_factory=dict)


@dataclass
class Lift:
    bearing_only: bool
    n_points: int  # points inside the mask before clustering
    n_used: int  # points of the kept cluster
    median_depth: float  # NaN if bearing_only
    centroid: np.ndarray | None  # (3,) in the output frame
    extent: np.ndarray | None  # (3,) axis-aligned size in the output frame
    bearing_cam: np.ndarray  # (3,) unit viewing ray through the mask centre, camera frame
    points_cam: np.ndarray  # (n_used,3) kept points, camera frame (for debugging / thermal)
    reason: str = ''


# ------------------------------------------------------------------ scans
def select_scans(scans: list[Scan], t_image: float, n: int) -> list[Scan]:
    """The `n` scans whose mid-time is closest to the image time, oldest first."""
    if not scans or n <= 0:
        return []
    best = sorted(scans, key=lambda s: abs(s.stamp + 0.5 * s.duration - t_image))[:n]
    return sorted(best, key=lambda s: s.stamp)


def merge_scans(
    scans: list[Scan],
    pose_fn: Callable[[float], np.ndarray | None],
    time_bins: int = 1,
    min_range_m: float = 0.0,
    max_range_m: float = np.inf,
) -> tuple[np.ndarray, dict]:
    """Transform scans into the frame `pose_fn` maps to (T_fixed_sensor at time t) and concatenate them.

    With `time_bins` > 1 and per-point times, each scan is split into time slices that use the pose at their own mid-time
    (deskewing while the robot moves). Points outside [min_range, max_range] of the sensor are dropped first.
    Returns (points (M,3) float32, {'scans_used', 'scans_skipped', 'points'})."""
    out, used, skipped = [], 0, 0
    for s in scans:
        xyz = s.xyz
        r = np.linalg.norm(xyz, axis=1)
        keep = (r >= min_range_m) & (r <= max_range_m)
        xyz = xyz[keep]
        trel = s.t_rel[keep] if s.t_rel is not None else None
        if xyz.shape[0] == 0:
            skipped += 1
            continue
        if time_bins > 1 and trel is not None and np.ptp(trel) > 1e-4:
            edges = np.linspace(trel.min(), trel.max() + 1e-9, time_bins + 1)
            which = np.clip(np.digitize(trel, edges) - 1, 0, time_bins - 1)
            groups = [(s.stamp + 0.5 * (edges[b] + edges[b + 1]), xyz[which == b]) for b in range(time_bins)]
        else:
            groups = [(s.stamp + 0.5 * s.duration, xyz)]
        ok = True
        parts = []
        for t, pts in groups:
            if pts.shape[0] == 0:
                continue
            T = pose_fn(t)
            if T is None:
                ok = False
                break
            parts.append(apply(T, pts.astype(np.float64)).astype(np.float32))
        if ok and parts:
            out.extend(parts)
            used += 1
        else:
            skipped += 1
    pts = np.concatenate(out, axis=0) if out else np.zeros((0, 3), np.float32)
    return pts, {'scans_used': used, 'scans_skipped': skipped, 'points': int(pts.shape[0])}


# ------------------------------------------------------------------ projection
def _shift_reduce(img: np.ndarray, r: int, op, pad) -> np.ndarray:
    """Separable (2r+1)x(2r+1) window reduction with `op` (np.minimum / np.maximum), borders padded with `pad`."""

    def one_axis(a, axis):
        out = a.copy()
        for d in range(1, r + 1):
            sl_a, sl_b = [slice(None)] * 2, [slice(None)] * 2
            sl_a[axis], sl_b[axis] = slice(d, None), slice(None, -d)
            out[tuple(sl_a)] = op(out[tuple(sl_a)], a[tuple(sl_b)])
            out[tuple(sl_b)] = op(out[tuple(sl_b)], a[tuple(sl_a)])
            edge_lo, edge_hi = [slice(None)] * 2, [slice(None)] * 2
            edge_lo[axis], edge_hi[axis] = slice(0, d), slice(-d, None)
            out[tuple(edge_lo)] = op(out[tuple(edge_lo)], pad)
            out[tuple(edge_hi)] = op(out[tuple(edge_hi)], pad)
        return out

    return one_axis(one_axis(img, 0), 1)


def erode_mask(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion with a square window; image borders do not erode (objects cut by the frame keep their edge)."""
    if r <= 0:
        return mask
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask
    y0, y1 = max(ys.min() - r, 0), min(ys.max() + r + 1, mask.shape[0])
    x0, x1 = max(xs.min() - r, 0), min(xs.max() + r + 1, mask.shape[1])
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    # padding with 1 only where the crop touches the image border, 0 elsewhere (the crop includes an r margin of zeros)
    er = _shift_reduce(sub, r, np.minimum, 1)
    out = np.zeros_like(mask, dtype=bool)
    out[y0:y1, x0:x1] = er.astype(bool)
    return out


def project_points(cam: CameraModel, pts_cam: np.ndarray, p: ProjectorParams) -> ProjectedPoints:
    """Project camera-frame points, keep those inside the image and range limits, drop occluded ones."""
    stats = {'input': int(pts_cam.shape[0])}
    if pts_cam.shape[0] == 0:
        return ProjectedPoints(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0), np.zeros((0, 3)), stats)
    pts = np.asarray(pts_cam, np.float64)
    uv, valid = cam.project(pts)
    inb = valid & cam.in_bounds(uv)
    z = pts[:, 2]
    rng = np.linalg.norm(pts, axis=1)
    inb &= (rng >= p.min_range_m) & (rng <= p.max_range_m)
    pts, uv, z = pts[inb], uv[inb], z[inb]
    u = np.clip(uv[:, 0].astype(np.int32), 0, cam.width - 1)
    v = np.clip(uv[:, 1].astype(np.int32), 0, cam.height - 1)
    stats['in_image'] = int(z.size)
    if p.occlusion_window_px > 1 and z.size:
        key = v.astype(np.int64) * cam.width + u
        order = np.lexsort((z, key))
        ks = key[order]
        first = np.ones(ks.size, bool)
        first[1:] = ks[1:] != ks[:-1]
        zbuf = np.full(cam.height * cam.width, np.inf)
        zbuf[ks[first]] = z[order][first]
        zmin = _shift_reduce(zbuf.reshape(cam.height, cam.width), p.occlusion_window_px // 2, np.minimum, np.inf)
        visible = z <= zmin[v, u] + p.occlusion_abs_tol_m + p.occlusion_rel_tol * z
        stats['occluded'] = int((~visible).sum())
        pts, z, u, v = pts[visible], z[visible], u[visible], v[visible]
    stats['kept'] = int(z.size)
    return ProjectedPoints(u, v, z, pts, stats)


# ------------------------------------------------------------------ per-mask estimate
def _depth_cluster(depth: np.ndarray, p: ProjectorParams) -> np.ndarray:
    """Boolean selection of the foreground depth cluster among points sorted arbitrarily."""
    order = np.argsort(depth)
    d = depth[order]
    brk = np.nonzero(np.diff(d) > p.depth_gap_abs_m + p.depth_gap_rel * d[:-1])[0] + 1
    bounds = np.concatenate([[0], brk, [d.size]])
    counts = np.diff(bounds)
    ok = np.nonzero(counts >= max(p.cluster_keep_frac * counts.max(), p.min_points))[0]
    if ok.size == 0:
        return np.zeros(depth.size, bool)
    c = ok[0]  # clusters are in increasing depth: the nearest supported one
    sel = np.zeros(depth.size, bool)
    sel[order[bounds[c] : bounds[c + 1]]] = True
    return sel


def _largest_component(pts: np.ndarray, eps: float) -> np.ndarray:
    """Boolean selection of the largest group of points connected through voxels of size eps (26-neighbourhood)."""
    if pts.shape[0] < 2:
        return np.ones(pts.shape[0], bool)
    vox = np.floor(pts / eps).astype(np.int64)
    uniq, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    index = {tuple(k): i for i, k in enumerate(uniq)}
    label = -np.ones(len(uniq), int)
    sizes = []
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    for s in range(len(uniq)):
        if label[s] >= 0:
            continue
        lab = len(sizes)
        label[s] = lab
        stack, total = [s], 0
        while stack:
            i = stack.pop()
            total += counts[i]
            k = uniq[i]
            for o in offsets:
                j = index.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]))
                if j is not None and label[j] < 0:
                    label[j] = lab
                    stack.append(j)
        sizes.append(total)
    return label[inv] == int(np.argmax(sizes))


def estimate_object(pts_cam: np.ndarray, p: ProjectorParams, T_out_cam: np.ndarray | None = None) -> tuple[bool, dict]:
    """Robust 3D estimate from candidate camera-frame points. Returns (ok, dict with points/centroid/extent/median_depth)."""
    if pts_cam.shape[0] < p.min_points:
        return False, {'reason': 'too_few_points'}
    sel = _depth_cluster(pts_cam[:, 2], p)
    if sel.sum() < p.min_points:
        return False, {'reason': 'no_supported_depth_cluster'}
    pts = pts_cam[sel]
    md = float(np.median(pts[:, 2]))
    comp = _largest_component(pts, p.euclid_eps_m + p.euclid_eps_rel * md)
    if comp.sum() < p.min_points:
        return False, {'reason': 'fragmented'}
    pts = pts[comp]
    out = apply(T_out_cam, pts) if T_out_cam is not None else pts
    lo, hi = np.percentile(out, p.extent_percentiles, axis=0)
    return True, {
        'points': pts,
        'centroid': np.median(out, axis=0),
        'extent': hi - lo,
        'median_depth': float(np.median(pts[:, 2])),
    }


def lift_mask(
    proj: ProjectedPoints,
    mask: np.ndarray,
    cam: CameraModel,
    p: ProjectorParams,
    T_out_cam: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> Lift:
    """Estimate the 3D position of one mask (bool array at the working resolution of `cam`)."""
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    if area == 0:
        raise ValueError('empty mask')
    bearing = cam.rays(np.array([[xs.mean() + 0.5, ys.mean() + 0.5]]))[0]
    m = erode_mask(mask, p.mask_erode_px) if area >= p.erode_min_area_px else mask
    if not m.any():
        m = mask
    inside = m[proj.v, proj.u] if proj.u.size else np.zeros(0, bool)
    pts = proj.xyz[inside]
    n_in = int(pts.shape[0])
    if n_in > p.max_points_per_mask:
        rng = rng or np.random.default_rng(0)
        pts = pts[rng.choice(n_in, p.max_points_per_mask, replace=False)]
    ok, est = estimate_object(pts, p, T_out_cam)
    if not ok:
        return Lift(True, n_in, 0, float('nan'), None, None, bearing, np.zeros((0, 3)), est['reason'])
    return Lift(
        False, n_in, int(est['points'].shape[0]), est['median_depth'], est['centroid'], est['extent'], bearing, est['points']
    )


def depth_image_points(depth: np.ndarray, mask: np.ndarray, cam: CameraModel, p: ProjectorParams) -> np.ndarray:
    """Camera-frame points from a metric depth image (sim ground truth) for the pixels of `mask` (working resolution)."""
    m = erode_mask(mask, p.mask_erode_px) if mask.sum() >= p.erode_min_area_px else mask
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return np.zeros((0, 3))
    if ys.size > p.max_points_per_mask:
        idx = np.linspace(0, ys.size - 1, p.max_points_per_mask).astype(int)
        ys, xs = ys[idx], xs[idx]
    z = depth[ys, xs].astype(np.float64)
    ok = np.isfinite(z) & (z > p.min_range_m) & (z < p.max_range_m)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    rays = cam.rays(np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64))
    return rays * (z / rays[:, 2])[:, None]


def lift_mask_from_depth(
    depth: np.ndarray, mask: np.ndarray, cam: CameraModel, p: ProjectorParams, T_out_cam: np.ndarray | None = None
) -> Lift:
    """Same output as `lift_mask` but with points taken from a depth image (`depth_source: sim_depth`)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError('empty mask')
    bearing = cam.rays(np.array([[xs.mean() + 0.5, ys.mean() + 0.5]]))[0]
    pts = depth_image_points(depth, mask, cam, p)
    ok, est = estimate_object(pts, p, T_out_cam)
    if not ok:
        return Lift(True, int(pts.shape[0]), 0, float('nan'), None, None, bearing, np.zeros((0, 3)), est['reason'])
    return Lift(
        False,
        int(pts.shape[0]),
        int(est['points'].shape[0]),
        est['median_depth'],
        est['centroid'],
        est['extent'],
        bearing,
        est['points'],
    )
