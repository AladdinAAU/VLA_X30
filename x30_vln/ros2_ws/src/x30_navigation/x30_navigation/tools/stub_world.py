"""Tiny 2.5D world for navigation tests without the simulator: extruded wall segments, floor and ceiling,
a vectorised 3D lidar ray caster and a ground-truth occupancy raster. ROS-free.

This is test tooling for x30_navigation (the real simulator is x30_sim). Coordinates: world frame, z = 0 is the floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CEILING = 2.6


@dataclass
class World:
    segments: np.ndarray                      # (S, 5): x1, y1, x2, y2, height
    bounds: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    free_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # sample start/goal candidates
    name: str = "world"

    def raycast(self, origin: np.ndarray, dirs: np.ndarray, max_range: float = 30.0) -> np.ndarray:
        """Range along each unit direction (N, 3) from origin (3,) to floor / ceiling / wall; inf beyond max_range."""
        n = len(dirs)
        t = np.full(n, np.inf)
        dz = dirs[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_floor = np.where(dz < -1e-6, -origin[2] / dz, np.inf)
            t_ceil = np.where(dz > 1e-6, (CEILING - origin[2]) / dz, np.inf)
        t = np.minimum(t_floor, t_ceil)
        d = dirs[:, :2]
        seg = self.segments
        p1 = seg[:, 0:2]
        e = seg[:, 2:4] - p1
        w = p1[None, :, :] - origin[None, None, :2]                      # (1, S, 2)
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]   # (N, S)
        num_t = w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]
        num_s = w[..., 0] * d[:, None, 1] - w[..., 1] * d[:, None, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = num_t / denom
            ss = num_s / denom
        with np.errstate(invalid="ignore"):
            z = origin[2] + tt * dz[:, None]
        ok = (np.abs(denom) > 1e-9) & (tt > 0) & (ss >= 0) & (ss <= 1) & (z >= 0) & (z <= seg[None, :, 4])
        tt = np.where(ok, tt, np.inf)
        t_wall = tt.min(axis=1)
        t = np.minimum(t, t_wall)
        t[t > max_range] = np.inf
        return t

    def collides(self, x: float, y: float, yaw: float, length: float, width: float, min_height: float = 0.1) -> bool:
        """Whether the footprint rectangle touches any wall/obstacle segment taller than min_height."""
        c, s = math.cos(yaw), math.sin(yaw)
        hl, hw = length / 2.0, width / 2.0
        corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        poly = corners @ np.array([[c, s], [-s, c]]) + np.array([x, y])
        for x1, y1, x2, y2, h in self.segments:
            if h < min_height:
                continue
            if _segment_hits_polygon(np.array([x1, y1]), np.array([x2, y2]), poly):
                return True
        return False

    def raster(self, resolution: float = 0.05, min_height: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Ground-truth occupancy raster (True = blocked) and its origin."""
        xmin, xmax, ymin, ymax = self.bounds
        w, h = int((xmax - xmin) / resolution) + 1, int((ymax - ymin) / resolution) + 1
        img = np.zeros((h, w), dtype=bool)
        for x1, y1, x2, y2, hh in self.segments:
            if hh < min_height:
                continue
            n = max(2, int(math.hypot(x2 - x1, y2 - y1) / (0.5 * resolution)))
            xs = np.linspace(x1, x2, n)
            ys = np.linspace(y1, y2, n)
            cx = np.clip(((xs - xmin) / resolution).astype(int), 0, w - 1)
            cy = np.clip(((ys - ymin) / resolution).astype(int), 0, h - 1)
            img[cy, cx] = True
        return img, np.array([xmin, ymin])


def _segment_hits_polygon(a: np.ndarray, b: np.ndarray, poly: np.ndarray) -> bool:
    def inside(p):
        sign = None
        for i in range(4):
            q1, q2 = poly[i], poly[(i + 1) % 4]
            c = (q2[0] - q1[0]) * (p[1] - q1[1]) - (q2[1] - q1[1]) * (p[0] - q1[0])
            if abs(c) < 1e-12:
                continue
            if sign is None:
                sign = c > 0
            elif (c > 0) != sign:
                return False
        return True

    if inside(a) or inside(b):
        return True
    for i in range(4):
        if _segments_intersect(a, b, poly[i], poly[(i + 1) % 4]):
            return True
    return False


def _segments_intersect(p1, p2, p3, p4) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _box(x: float, y: float, sx: float, sy: float, h: float):
    x1, x2, y1, y2 = x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2
    return [(x1, y1, x2, y1, h), (x2, y1, x2, y2, h), (x2, y2, x1, y2, h), (x1, y2, x1, y1, h)]


def _wall_with_doors(x1, y1, x2, y2, doors, door_w=1.0, h=CEILING):
    """Wall from (x1,y1) to (x2,y2) with door gaps centred at the given positions (distance along the wall)."""
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    cuts = sorted(doors)
    out, pos = [], 0.0
    for c in cuts:
        a, b = c - door_w / 2, c + door_w / 2
        if a > pos:
            out.append((x1 + ux * pos, y1 + uy * pos, x1 + ux * a, y1 + uy * a, h))
        pos = b
    if pos < length:
        out.append((x1 + ux * pos, y1 + uy * pos, x2, y2, h))
    return out


def make_world(name: str = "rooms", seed: int = 0) -> World:
    """Presets: 'hall' (one big room with obstacles), 'rooms' (3 x 2 rooms with 1 m doors and obstacles)."""
    rng = np.random.default_rng(seed)
    seg: list[tuple] = []
    if name == "hall":
        w, h = 14.0, 9.0
        seg += _wall_with_doors(0, 0, w, 0, []) + _wall_with_doors(w, 0, w, h, []) \
            + _wall_with_doors(w, h, 0, h, []) + _wall_with_doors(0, h, 0, 0, [])
        for p in [(4, 3, 1.0, 0.6, 1.2), (8, 6, 0.8, 0.8, 0.5), (10.5, 2.5, 0.6, 1.4, 1.0), (6, 7.5, 1.2, 0.5, 1.4), (3, 6.5, 0.5, 0.5, 0.3)]:
            seg += _box(*p)
        bounds = (0.0, w, 0.0, h)
    else:
        w, h = 16.0, 10.0
        seg += _wall_with_doors(0, 0, w, 0, []) + _wall_with_doors(w, 0, w, h, []) \
            + _wall_with_doors(w, h, 0, h, []) + _wall_with_doors(0, h, 0, 0, [])
        # two vertical partitions at x = 5.3 and 10.7, one horizontal at y = 5 (doors 1.0 m wide)
        seg += _wall_with_doors(5.3, 0, 5.3, h, [2.0, 7.5])
        seg += _wall_with_doors(10.7, 0, 10.7, h, [3.5, 8.0])
        seg += _wall_with_doors(0, 5.0, 5.3, 5.0, [2.6])
        seg += _wall_with_doors(5.3, 5.0, 10.7, 5.0, [])
        seg += _wall_with_doors(10.7, 5.0, w, 5.0, [13.4])
        for p in [(2.5, 2.2, 0.8, 0.8, 1.1), (3.8, 7.5, 1.0, 0.5, 0.6), (8.0, 2.5, 0.6, 0.6, 1.4), (7.9, 7.6, 1.4, 0.5, 1.0),
                  (13.5, 2.0, 1.0, 1.0, 0.8), (13.8, 7.0, 0.6, 1.6, 1.3), (1.6, 8.3, 0.5, 0.5, 0.4)]:
            seg += _box(*p)
        bounds = (0.0, w, 0.0, h)
    segments = np.array(seg, dtype=np.float64)
    world = World(segments=segments, bounds=bounds, name=name)
    img, origin = world.raster(0.1)
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(~img) * 0.1
    ys, xs = np.nonzero(dist > 0.6)
    pts = np.column_stack((xs * 0.1 + origin[0], ys * 0.1 + origin[1]))
    world.free_points = pts[rng.permutation(len(pts))]
    return world
