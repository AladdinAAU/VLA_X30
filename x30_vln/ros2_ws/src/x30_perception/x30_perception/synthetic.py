"""Synthetic test world: a room with upright rectangles ("billboards") that can carry a real photo as texture, a pinhole
camera renderer with exact ground-truth masks and depth, and a virtual lidar that moves with the robot.

Used by the unit tests (accuracy of the lidar lifting against known geometry) and by scripts/synthetic_stream.py, which
publishes the same data as ROS topics so the whole client chain can be exercised without the simulator.
World frame: x forward / y left / z up, floor at z = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera_model import CameraModel
from .geometry import apply, invert, make_transform
from .lidar_projector import Scan

FLOOR, CEILING, WALL = -1, -2, -3


@dataclass
class Billboard:
    label: str
    center: tuple  # world (x, y, z) of the rectangle centre
    width: float
    height: float
    facing_yaw: float = 3.14159265  # direction of the surface normal in the world (pi: faces back towards -x)
    texture: np.ndarray | None = None  # BGR image, shown on the rectangle (None: flat colour)
    color: tuple = (60, 60, 200)

    @property
    def normal(self) -> np.ndarray:
        return np.array([np.cos(self.facing_yaw), np.sin(self.facing_yaw), 0.0])

    @property
    def tangent(self) -> np.ndarray:
        return np.array([-np.sin(self.facing_yaw), np.cos(self.facing_yaw), 0.0])

    def corners(self) -> np.ndarray:
        """Top-left, top-right, bottom-right, bottom-left as seen from the front (normal side)."""
        c, t, hw, hh = np.array(self.center, float), self.tangent, self.width / 2, self.height / 2
        up = np.array([0, 0, 1.0])
        # seen from the front (looking against the normal) the tangent points to the viewer's right
        return np.stack([c - t * hw + up * hh, c + t * hw + up * hh, c + t * hw - up * hh, c - t * hw - up * hh])


@dataclass
class Scene:
    billboards: list = field(default_factory=list)
    room: tuple = (-6.0, 14.0, -6.0, 6.0, 2.8)  # x0, x1, y0, y1, ceiling height

    def cast(self, origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest hit along rays: (distance (N,), object id (N,)); billboards >= 0, FLOOR/CEILING/WALL negative."""
        n = dirs.shape[0]
        origins = np.broadcast_to(origins, (n, 3))
        best = np.full(n, np.inf)
        ids = np.full(n, WALL)
        x0, x1, y0, y1, zc = self.room
        planes = [
            (np.array([0, 0, 1.0]), 0.0, FLOOR),
            (np.array([0, 0, -1.0]), -zc, CEILING),
            (np.array([1.0, 0, 0]), x0, WALL),
            (np.array([-1.0, 0, 0]), -x1, WALL),
            (np.array([0, 1.0, 0]), y0, WALL),
            (np.array([0, -1.0, 0]), -y1, WALL),
        ]
        for nrm, off, oid in planes:  # inward normals: points with nrm.p >= off are inside
            dn = dirs @ nrm
            with np.errstate(divide='ignore', invalid='ignore'):
                s = (off - origins @ nrm) / dn
            ok = (dn < -1e-9) & (s > 0) & (s < best)
            best = np.where(ok, s, best)
            ids = np.where(ok, oid, ids)
        for i, b in enumerate(self.billboards):
            c, nrm, tg = np.array(b.center, float), b.normal, b.tangent
            dn = dirs @ nrm
            with np.errstate(divide='ignore', invalid='ignore'):
                s = ((c - origins) @ nrm) / dn
            p = origins + dirs * s[:, None]
            ok = (
                (np.abs(dn) > 1e-9)
                & (s > 0)
                & (s < best)
                & (np.abs((p - c) @ tg) <= b.width / 2)
                & (np.abs(p[:, 2] - c[2]) <= b.height / 2)
            )
            best = np.where(ok, s, best)
            ids = np.where(ok, i, ids)
        return best, ids


def optical_pose(position, yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """T_world_optical for a camera at `position` looking along heading `yaw` (z-up world), pitched down by `pitch`."""
    f = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.cross(f, right)
    return make_transform(np.stack([right, down, f], axis=1), position)


def render(scene: Scene, cam: CameraModel, T_world_cam: np.ndarray, with_texture: bool = True):
    """Return (rgb uint8 HxWx3 BGR, depth float32 HxW (z in camera frame), labels int16 HxW with the object id)."""
    import cv2

    h, w = cam.height, cam.width
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays_cam = cam.rays(np.stack([uu.ravel(), vv.ravel()], axis=1))
    R = T_world_cam[:3, :3]
    dirs = rays_cam @ R.T
    dist, ids = scene.cast(T_world_cam[:3, 3], dirs)
    depth = (dist * rays_cam[:, 2]).reshape(h, w).astype(np.float32)
    labels = ids.reshape(h, w).astype(np.int16)
    # background: shaded floor / wall / ceiling
    img = np.zeros((h, w, 3), np.float32)
    shade = np.clip(1.0 - dist / 25.0, 0.3, 1.0).reshape(h, w)
    for oid, col in ((FLOOR, (120, 125, 130)), (WALL, (170, 175, 165)), (CEILING, (210, 210, 210))):
        m = labels == oid
        img[m] = np.array(col, np.float32) * shade[m][:, None]
    img = img.astype(np.uint8)
    T_cam_world = invert(T_world_cam)
    for i, b in enumerate(scene.billboards):
        vis = labels == i
        if not vis.any():
            continue
        if with_texture and b.texture is not None:
            th, tw = b.texture.shape[:2]
            src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
            pc = apply(T_cam_world, b.corners())
            uv, valid = cam.project(pc)
            if valid.all():
                M = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
                warped = cv2.warpPerspective(b.texture, M, (w, h), flags=cv2.INTER_LINEAR)
                img[vis] = warped[vis]
                continue
        img[vis] = np.array(b.color, np.uint8)
    return img, depth, labels


def simulate_scan(
    scene: Scene,
    T_world_lidar_fn,
    t0: float,
    n_points: int = 80000,
    duration: float = 0.1,
    rng: np.random.Generator | None = None,
    el_range_deg=(-10.0, 60.0),
    az_range_deg=(-180.0, 180.0),
    noise_m: float = 0.02,
) -> Scan:
    """Virtual Livox-like scan. `T_world_lidar_fn(t)` is the lidar pose at time t, so a moving robot gives a skewed scan
    (every point is expressed in the sensor frame of its own time, as a real driver delivers it)."""
    rng = rng or np.random.default_rng(0)
    t_rel = np.sort(rng.uniform(0, duration, n_points))
    az = np.radians(rng.uniform(*az_range_deg, n_points))
    el = np.radians(rng.uniform(*el_range_deg, n_points))
    d_l = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    # group points by pose (poses change continuously; 20 slices are enough for the test speeds)
    xyz = np.zeros((n_points, 3), np.float32)
    hit = np.zeros(n_points, bool)
    for sl in np.array_split(np.arange(n_points), 20):
        T = T_world_lidar_fn(t0 + float(t_rel[sl].mean()))
        dw = d_l[sl] @ T[:3, :3].T
        dist, _ = scene.cast(T[:3, 3], dw)
        dist = dist + rng.normal(0, noise_m, dist.shape)
        ok = np.isfinite(dist) & (dist > 0.1) & (dist < 40)
        xyz[sl] = (d_l[sl] * dist[:, None]).astype(np.float32)
        hit[sl] = ok
    return Scan(t0, xyz[hit], t_rel[hit], duration)
