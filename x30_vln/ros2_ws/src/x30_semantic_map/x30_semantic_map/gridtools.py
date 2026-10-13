"""Occupancy grid helpers: coordinates, line of sight, frontiers (no ROS imports).

The grid follows nav_msgs/OccupancyGrid: data[y * width + x], value -1 unknown, 0..100 occupancy, `origin` is the world
position of the corner of cell (0, 0). A rotated grid origin (yaw != 0) is not supported (slam_toolbox and the
occupancy mapper publish yaw 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import FrontierCfg


@dataclass
class GridView:
    data: np.ndarray                  # (H, W) int8 / int16, -1 unknown
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """(column, row) of a world point (not clipped)."""
        return int(math.floor((x - self.origin_x) / self.resolution)), int(math.floor((y - self.origin_y) / self.resolution))

    def cell_to_world(self, col: float, row: float) -> tuple[float, float]:
        """World coordinates of the centre of cell (col, row)."""
        return self.origin_x + (col + 0.5) * self.resolution, self.origin_y + (row + 0.5) * self.resolution

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def free_mask(self, free_th: int = 25) -> np.ndarray:
        return (self.data >= 0) & (self.data < free_th)

    def occupied_mask(self, occ_th: int = 65) -> np.ndarray:
        return self.data >= occ_th

    def unknown_mask(self) -> np.ndarray:
        return self.data < 0

    def known_bounds(self) -> tuple[float, float, float, float] | None:
        """(xmin, ymin, xmax, ymax) of all known cells, or None for an empty map."""
        rows, cols = np.nonzero(self.data >= 0)
        if rows.size == 0:
            return None
        x0, y0 = self.origin_x + cols.min() * self.resolution, self.origin_y + rows.min() * self.resolution
        x1, y1 = self.origin_x + (cols.max() + 1) * self.resolution, self.origin_y + (rows.max() + 1) * self.resolution
        return x0, y0, x1, y1

    def line_of_sight(self, p0, p1, occ_th: int = 65, ignore_end_m: float = 0.3) -> bool:
        """True if no occupied cell lies between the points (the last `ignore_end_m` are ignored: objects sit on walls)."""
        p0, p1 = np.asarray(p0, float)[:2], np.asarray(p1, float)[:2]
        dist = float(np.linalg.norm(p1 - p0))
        if dist < 1e-6:
            return True
        n = max(2, int(dist / (self.resolution * 0.5)))
        for k in range(1, n):
            s = k / n
            if (1 - s) * dist < ignore_end_m:
                break
            p = p0 + s * (p1 - p0)
            c, r = self.world_to_cell(p[0], p[1])
            if self.in_bounds(c, r) and self.data[r, c] >= occ_th:
                return False
        return True


@dataclass
class Frontier:
    center: tuple[float, float]      # world position of a frontier cell near the cluster centre
    length_m: float
    size_cells: int
    reachable: bool = True


def reachable_mask(grid: GridView, start_xy: tuple[float, float], free_th: int = 25, search_cells: int = 6) -> np.ndarray:
    """Free cells connected to the cell nearest to `start_xy` (unknown cells block)."""
    free = grid.free_mask(free_th).astype(np.uint8)
    n, labels = cv2.connectedComponents(free, connectivity=4)
    c, r = grid.world_to_cell(*start_xy)
    best = 0
    for rad in range(0, search_cells + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if grid.in_bounds(c + dc, r + dr) and labels[r + dr, c + dc] > 0:
                    best = int(labels[r + dr, c + dc])
                    break
            if best:
                break
        if best:
            break
    return labels == best if best else np.zeros_like(free, dtype=bool)


def find_frontiers(grid: GridView, robot_xy: tuple[float, float] | None, cfg: FrontierCfg, free_th: int = 25,
                   ) -> list[Frontier]:
    """Free cells that touch large unknown regions, clustered; sorted nearest first when the robot position is known."""
    free = grid.free_mask(free_th)
    unknown = grid.unknown_mask()
    # unknown pockets smaller than min_unknown_cells (speckle inside walls) are not unexplored space
    n, lab, stats, _ = cv2.connectedComponentsWithStats(unknown.astype(np.uint8), connectivity=4)
    big = np.zeros_like(unknown)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_unknown_cells:
            big |= lab == i
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    near_unknown = cv2.dilate(big.astype(np.uint8), kernel).astype(bool)
    edge = free & near_unknown
    if not edge.any():
        return []
    spread = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(spread, connectivity=8)
    reach = reachable_mask(grid, robot_xy, free_th) if robot_xy is not None else None
    out: list[Frontier] = []
    for i in range(1, n):
        cells = np.argwhere((lab == i) & edge)
        if len(cells) == 0:
            continue
        length = len(cells) * grid.resolution
        if length < cfg.min_length_m:
            continue
        centre = cells.mean(axis=0)
        k = int(np.argmin(((cells - centre) ** 2).sum(axis=1)))
        r, c = cells[k]
        x, y = grid.cell_to_world(c, r)
        out.append(Frontier((x, y), float(length), int(len(cells)),
                            bool(reach[r, c]) if reach is not None else True))
    if robot_xy is not None:
        out.sort(key=lambda f: math.hypot(f.center[0] - robot_xy[0], f.center[1] - robot_xy[1]))
    return out
