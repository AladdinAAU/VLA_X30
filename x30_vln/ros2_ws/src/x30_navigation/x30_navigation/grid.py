"""Growing 2D log-odds occupancy grid with azimuth-binned ray casting. ROS-free.

One scan is reduced to: obstacle points (hits) and ground points (free-space evidence), both in the map frame,
plus the sensor origin. Free space is carved along one ray per azimuth bin up to the nearest obstacle (or the
farthest ground return when the bin has no obstacle). Cells hit in the same scan are never freed in that scan, so
a table top (hit) and the wall seen under it (ray through the table's cell) do not cancel out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridParams:
    resolution: float = 0.05
    initial_size: float = 30.0       # metres, square, centred on the first sensor position
    grow_pad: float = 10.0           # metres added when a scan leaves the grid
    max_size: float = 400.0          # metres per side; scans beyond are clipped
    log_hit: float = 0.7
    log_free: float = 0.35
    log_min: float = -3.0
    log_max: float = 3.5
    occupied_thr: float = 1.0        # log-odds at or above: occupied (needs two hits)
    free_thr: float = -0.3           # log-odds at or below: free
    num_bins: int = 1024             # azimuth bins for ray casting
    max_free_range: float = 12.0


class LogOddsGrid:
    def __init__(self, params: GridParams | None = None):
        self.p = params or GridParams()
        self.log: np.ndarray | None = None
        self.origin = np.zeros(2)  # world position of the lower-left corner of cell (0, 0)

    @property
    def shape(self) -> tuple[int, int]:
        return (0, 0) if self.log is None else self.log.shape

    def _init_around(self, xy: np.ndarray) -> None:
        n = int(round(self.p.initial_size / self.p.resolution))
        self.log = np.zeros((n, n), dtype=np.float32)
        self.origin = xy - 0.5 * n * self.p.resolution

    def ensure_contains(self, lo: np.ndarray, hi: np.ndarray) -> None:
        """Grow the grid (in whole cells) so that it covers the world rectangle [lo, hi]."""
        res = self.p.resolution
        h, w = self.log.shape
        x0, y0 = self.origin
        pad = self.p.grow_pad
        add_l = int(np.ceil((x0 - lo[0] + pad) / res)) if lo[0] < x0 else 0
        add_b = int(np.ceil((y0 - lo[1] + pad) / res)) if lo[1] < y0 else 0
        add_r = int(np.ceil((hi[0] + pad - (x0 + w * res)) / res)) if hi[0] >= x0 + w * res else 0
        add_t = int(np.ceil((hi[1] + pad - (y0 + h * res)) / res)) if hi[1] >= y0 + h * res else 0
        if not (add_l or add_b or add_r or add_t):
            return
        new = np.zeros((h + add_b + add_t, w + add_l + add_r), dtype=np.float32)
        new[add_b:add_b + h, add_l:add_l + w] = self.log
        self.log = new
        self.origin = np.array([x0 - add_l * res, y0 - add_b * res])

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((xy - self.origin) / self.p.resolution).astype(np.int64)

    def integrate(self, origin_xy: np.ndarray, hits_xy: np.ndarray, ground_xy: np.ndarray, cap_xy: np.ndarray | None = None) -> None:
        """Integrate one scan (all coordinates in the map frame, metres). `cap_xy`: hits that only stop free-space carving."""
        p = self.p
        if self.log is None:
            self._init_around(origin_xy)
        pts = [origin_xy[None, :]]
        if len(hits_xy):
            pts.append(hits_xy)
        if len(ground_xy):
            pts.append(ground_xy)
        allp = np.vstack(pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        if (hi - lo).max() < p.max_size:
            self.ensure_contains(lo, hi)
        h, w = self.log.shape

        hit_cells = self._cells_in_grid(hits_xy)
        hit_key = np.unique(hit_cells[:, 1] * w + hit_cells[:, 0]) if len(hit_cells) else np.zeros(0, np.int64)

        free_key = self._free_cells(origin_xy, hits_xy, ground_xy, w, h, cap_xy)
        if len(free_key):
            free_key = np.setdiff1d(free_key, hit_key, assume_unique=True)
            flat = self.log.reshape(-1)
            flat[free_key] = np.maximum(flat[free_key] - p.log_free, p.log_min)
        if len(hit_key):
            flat = self.log.reshape(-1)
            flat[hit_key] = np.minimum(flat[hit_key] + p.log_hit, p.log_max)

    def _cells_in_grid(self, xy: np.ndarray) -> np.ndarray:
        if len(xy) == 0:
            return np.zeros((0, 2), np.int64)
        c = self.world_to_cell(xy)
        h, w = self.log.shape
        ok = (c[:, 0] >= 0) & (c[:, 0] < w) & (c[:, 1] >= 0) & (c[:, 1] < h)
        return c[ok]

    def _free_cells(self, origin_xy, hits_xy, ground_xy, w: int, h: int, cap_xy=None) -> np.ndarray:
        p = self.p
        nb = p.num_bins

        def bins_of(xy):
            d = xy - origin_xy
            az = np.arctan2(d[:, 1], d[:, 0])
            return np.minimum(((az + np.pi) / (2.0 * np.pi) * nb).astype(np.int64), nb - 1), np.hypot(d[:, 0], d[:, 1])

        r_obs = np.full(nb, np.inf)
        if len(hits_xy):
            b, r = bins_of(hits_xy)
            np.minimum.at(r_obs, b, r)
        r_gnd = np.full(nb, -1.0)
        if len(ground_xy):
            b, r = bins_of(ground_xy)
            np.maximum.at(r_gnd, b, np.minimum(r, p.max_free_range))
        r_free = np.where(np.isfinite(r_obs), np.minimum(r_obs, p.max_free_range), r_gnd)
        if cap_xy is not None and len(cap_xy):
            b, r = bins_of(cap_xy)
            r_cap = np.full(nb, np.inf)
            np.minimum.at(r_cap, b, r)
            r_free = np.where(np.isfinite(r_cap), np.minimum(np.where(r_free > 0, r_free, np.inf), r_cap), r_free)
        valid = r_free > 2.0 * p.resolution
        if not valid.any():
            return np.zeros(0, np.int64)
        centers = (np.nonzero(valid)[0] + 0.5) / nb * 2.0 * np.pi - np.pi
        lengths = r_free[valid] - p.resolution          # stop one cell short of the obstacle
        steps = np.maximum((lengths / (0.7 * p.resolution)).astype(np.int64), 1)
        ray = np.repeat(np.arange(len(centers)), steps)
        k = np.arange(len(ray)) - np.repeat(np.cumsum(steps) - steps, steps)
        t = (k + 1) * (lengths[ray] / steps[ray])
        xy = origin_xy + np.column_stack((np.cos(centers[ray]) * t, np.sin(centers[ray]) * t))
        c = self.world_to_cell(xy)
        ok = (c[:, 0] >= 0) & (c[:, 0] < w) & (c[:, 1] >= 0) & (c[:, 1] < h)
        c = c[ok]
        return np.unique(c[:, 1] * w + c[:, 0])

    def to_int8(self) -> np.ndarray:
        """Occupancy values for nav_msgs/OccupancyGrid: -1 unknown, 0 free, 100 occupied."""
        out = np.full(self.log.shape, -1, dtype=np.int8)
        out[self.log <= self.p.free_thr] = 0
        out[self.log >= self.p.occupied_thr] = 100
        return out
