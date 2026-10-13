"""Direct waypoint tracking without Nav2 (ROS-free): grid path planning and a pure-pursuit velocity law.

plan_path: Dijkstra on the free cells of an occupancy grid (-1 unknown, 0 free, 100 occupied), farther than `radius` from obstacles.
track: velocity command toward the next path point with heading control; slows near the goal. Obstacle stops are left to the safety guard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra

from .frontiers import _graph
from .geometry import wrap_angle


@dataclass
class TrackParams:
    v_max: float = 0.4          # m/s forward
    vy_max: float = 0.15        # m/s sideways (small corrections only)
    wz_max: float = 0.5         # rad/s
    k_yaw: float = 1.2          # wz = k_yaw * heading error
    lookahead: float = 0.7      # m
    slow_radius: float = 1.2    # m: speed ramps down inside this distance of the goal
    v_min: float = 0.08         # m/s creeping speed near the goal
    turn_in_place: float = 0.6  # rad: above this heading error only rotate


def plan_path(grid: np.ndarray, resolution: float, origin: np.ndarray, start_xy, goal_xy, radius: float = 0.3,
              unknown_free: bool = False) -> np.ndarray | None:
    """World-frame polyline (N, 2) from start to goal, or None. Start and goal snap to the nearest traversable cell within 0.6 m."""
    occ = grid == 100
    dist = ndimage.distance_transform_edt(~occ) * resolution
    free = (grid == 0) | ((grid == -1) & unknown_free)
    trav = free & (dist >= radius)
    h, w = grid.shape

    def cell(xy):
        c = np.floor((np.asarray(xy) - origin) / resolution).astype(int)
        return int(np.clip(c[0], 0, w - 1)), int(np.clip(c[1], 0, h - 1))

    def snap(c):
        if trav[c[1], c[0]]:
            return c
        r = int(math.ceil(0.6 / resolution))
        y0, y1, x0, x1 = max(0, c[1] - r), min(h, c[1] + r + 1), max(0, c[0] - r), min(w, c[0] + r + 1)
        ys, xs = np.nonzero(trav[y0:y1, x0:x1])
        if len(ys) == 0:
            return None
        k = int(np.argmin((ys + y0 - c[1]) ** 2 + (xs + x0 - c[0]) ** 2))
        return int(xs[k] + x0), int(ys[k] + y0)

    s, g = snap(cell(start_xy)), snap(cell(goal_xy))
    if s is None or g is None:
        return None
    mat, idx = _graph(trav)
    d, pred = dijkstra(mat, directed=False, indices=int(idx[s[1], s[0]]), return_predecessors=True)
    gi = int(idx[g[1], g[0]])
    if not np.isfinite(d[gi]):
        return None
    inv = np.column_stack(np.nonzero(trav))                       # node index -> (row, col), same order as _graph
    nodes = [gi]
    while nodes[-1] != int(idx[s[1], s[0]]):
        nodes.append(int(pred[nodes[-1]]))
    cells = inv[nodes[::-1]]
    xy = origin + (cells[:, ::-1] + 0.5) * resolution
    xy = xy[:: max(1, int(0.25 / resolution))]                    # one point per ~0.25 m
    return np.vstack([xy, np.asarray(goal_xy, dtype=float)[None, :]])


def straight_line(start_xy, goal_xy, step: float = 0.25) -> np.ndarray:
    a, b = np.asarray(start_xy, float), np.asarray(goal_xy, float)
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    return np.linspace(a, b, n)


def track(pose_xyyaw, path: np.ndarray, p: TrackParams) -> tuple[float, float, float, float]:
    """Velocity command (vx, vy, wz) in the body frame and the remaining distance to the path end."""
    x, y, yaw = pose_xyyaw
    pos = np.array([x, y])
    goal = path[-1]
    dist_goal = float(np.linalg.norm(goal - pos))
    near = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
    seg = np.linalg.norm(np.diff(path[near:], axis=0), axis=1) if near < len(path) - 1 else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    k = int(np.searchsorted(cum, p.lookahead)) if len(cum) else 0
    target = path[min(near + k, len(path) - 1)]
    d = target - pos
    err = wrap_angle(math.atan2(d[1], d[0]) - yaw) if np.linalg.norm(d) > 1e-3 else 0.0
    wz = float(np.clip(p.k_yaw * err, -p.wz_max, p.wz_max))
    if abs(err) > p.turn_in_place:
        return 0.0, 0.0, wz, dist_goal
    v = p.v_max * max(0.0, math.cos(err)) ** 2
    if dist_goal < p.slow_radius:
        v = min(v, max(p.v_min, p.v_max * dist_goal / p.slow_radius))
    return float(v), 0.0, wz, dist_goal
