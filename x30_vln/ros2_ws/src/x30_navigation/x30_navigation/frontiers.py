"""Frontier detection and selection on an occupancy grid (numpy / scipy, ROS-free).

A frontier is a traversable, reachable free cell that touches unknown space. Reachability and distance come
from Dijkstra over the free cells that are farther than the robot radius from any obstacle, so the returned
goals can be driven to, and frontiers behind walls or in closed pockets are never offered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

UNKNOWN, FREE, OCCUPIED = -1, 0, 100


@dataclass
class FrontierParams:
    downsample: int = 2                 # planning resolution = map resolution * downsample
    robot_radius: float = 0.35          # cells closer than this to an obstacle are not traversable
    min_frontier_length: float = 0.5    # m, shorter frontiers are noise
    gain_radius: float = 2.5            # m, unknown area counted around a goal for info_gain
    exclude_radius: float = 0.8         # m, frontiers within this of an excluded point are skipped
    assumed_depth: float = 2.0          # m, unexplored depth assumed behind a frontier for explored_fraction
    min_goal_distance: float = 0.4      # m, ignore frontiers closer than this to the robot (already there)
    goal_clearance: float = 0.6         # m; prefer goal cells at least this far from obstacles (room to turn the 1 m body)


@dataclass
class Frontier:
    goal: tuple[float, float, float]    # x, y, yaw (towards the unknown space)
    size: int                           # frontier cells (planning resolution)
    length_m: float
    distance_m: float                   # path distance from the robot
    info_gain: float                    # unknown area (m^2) within gain_radius of the goal
    centroid: tuple[float, float]


@dataclass
class FrontierResult:
    frontiers: list[Frontier] = field(default_factory=list)
    explored_fraction: float = 0.0
    known_free_area: float = 0.0


def _downsample(grid: np.ndarray, f: int) -> np.ndarray:
    """Block reduce: occupied if any cell is occupied, else free if any is free, else unknown."""
    if f <= 1:
        return grid
    h, w = grid.shape
    ph, pw = (-h) % f, (-w) % f
    g = np.pad(grid, ((0, ph), (0, pw)), constant_values=UNKNOWN)
    blocks = g.reshape(g.shape[0] // f, f, g.shape[1] // f, f)
    occ = (blocks == OCCUPIED).any(axis=(1, 3))
    free = (blocks == FREE).any(axis=(1, 3))
    out = np.full(occ.shape, UNKNOWN, dtype=np.int8)
    out[free] = FREE
    out[occ] = OCCUPIED
    return out


def _graph(trav: np.ndarray):
    """Sparse 8-connected graph over traversable cells; returns (csr matrix, index map)."""
    h, w = trav.shape
    idx = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.nonzero(trav)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, wts = [], [], []
    for dy, dx, c in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
        a = idx[0:h - dy, max(0, -dx):w - max(0, dx)]
        b = idx[dy:h, max(0, dx):w - max(0, -dx)]
        ok = (a >= 0) & (b >= 0)
        rows.append(a[ok])
        cols.append(b[ok])
        wts.append(np.full(ok.sum(), c))
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    n = len(ys)
    mat = coo_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n)).tocsr()
    return mat, idx


def find_frontiers(grid: np.ndarray, resolution: float, origin: np.ndarray, robot_xy: np.ndarray,
                   params: FrontierParams | None = None) -> FrontierResult:
    """Detect reachable frontiers on an int8 grid (-1 unknown, 0 free, 100 occupied)."""
    p = params or FrontierParams()
    f = max(1, p.downsample)
    g = _downsample(grid, f)
    res = resolution * f
    h, w = g.shape
    free = g == FREE
    occ = g == OCCUPIED
    unk = g == UNKNOWN
    known_free_area = float(free.sum()) * res * res
    result = FrontierResult(known_free_area=known_free_area)
    if known_free_area == 0.0:
        return result

    dist_obs = ndimage.distance_transform_edt(~occ) * res
    trav = free & (dist_obs >= p.robot_radius)
    rc = np.floor((robot_xy - origin) / res).astype(np.int64)
    rx, ry = int(np.clip(rc[0], 0, w - 1)), int(np.clip(rc[1], 0, h - 1))
    if not trav[ry, rx]:  # robot slightly inside the inflated zone: allow its surroundings
        r_cells = int(math.ceil(0.5 / res))
        y0, y1, x0, x1 = max(0, ry - r_cells), min(h, ry + r_cells + 1), max(0, rx - r_cells), min(w, rx + r_cells + 1)
        trav[y0:y1, x0:x1] |= free[y0:y1, x0:x1]
        if not trav[ry, rx]:
            return result
    labels, _ = ndimage.label(trav, structure=np.ones((3, 3)))
    trav &= labels == labels[ry, rx]

    unk_near = ndimage.binary_dilation(unk, structure=ndimage.generate_binary_structure(2, 1))
    front = trav & unk_near
    if not front.any():
        result.explored_fraction = 1.0
        return result
    fl, nf = ndimage.label(front, structure=np.ones((3, 3)))

    mat, idx = _graph(trav)
    d = dijkstra(mat, directed=False, indices=int(idx[ry, rx])) * res

    min_cells = max(1, int(round(p.min_frontier_length / res)))
    gain_cells = int(round(p.gain_radius / res))
    out: list[Frontier] = []
    for lab in range(1, nf + 1):
        ys, xs = np.nonzero(fl == lab)
        if len(ys) < min_cells:
            continue
        cx, cy = xs.mean(), ys.mean()
        dcent = (xs - cx) ** 2 + (ys - cy) ** 2
        roomy = dist_obs[ys, xs] >= p.goal_clearance
        k = int(np.argmin(np.where(roomy, dcent, np.inf))) if roomy.any() else int(np.argmin(dcent))
        gx, gy = int(xs[k]), int(ys[k])
        dist = float(d[idx[gy, gx]])
        if not np.isfinite(dist) or dist < p.min_goal_distance:
            continue
        y0, y1, x0, x1 = max(0, gy - gain_cells), min(h, gy + gain_cells + 1), max(0, gx - gain_cells), min(w, gx + gain_cells + 1)
        win = unk[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (yy - gy) ** 2 + (xx - gx) ** 2 <= gain_cells ** 2
        gain_mask = win & disk
        gain = float(gain_mask.sum()) * res * res
        if gain_mask.any():
            ux, uy = xx[gain_mask].mean() - gx, yy[gain_mask].mean() - gy
            yaw = math.atan2(uy, ux)
        else:
            yaw = 0.0
        wx, wy = origin + (np.array([gx, gy]) + 0.5) * res
        out.append(Frontier(goal=(float(wx), float(wy), yaw), size=int(len(ys)), length_m=len(ys) * res,
                            distance_m=dist, info_gain=gain,
                            centroid=tuple(origin + (np.array([cx, cy]) + 0.5) * res)))
    result.frontiers = out
    potential = sum(fr.length_m for fr in out) * p.assumed_depth
    result.explored_fraction = known_free_area / (known_free_area + potential) if (known_free_area + potential) > 0 else 0.0
    return result


def select(frontiers: list[Frontier], strategy: str = "nearest", bias_xy=None, exclude=None,
           exclude_radius: float = 0.8) -> Frontier | None:
    """Pick one frontier. Strategies: nearest, largest, informative, biased (towards `bias_xy`)."""
    exclude = exclude or []
    cand = [fr for fr in frontiers
            if all(math.hypot(fr.goal[0] - ex[0], fr.goal[1] - ex[1]) > exclude_radius for ex in exclude)]
    if not cand:
        return None
    if strategy == "largest":
        return max(cand, key=lambda fr: (fr.length_m, -fr.distance_m))
    if strategy == "informative":
        return max(cand, key=lambda fr: fr.info_gain / (fr.distance_m + 2.0))
    if strategy in ("biased", "bias") and bias_xy is not None:
        return min(cand, key=lambda fr: math.hypot(fr.goal[0] - bias_xy[0], fr.goal[1] - bias_xy[1]) + 0.25 * fr.distance_m)
    return min(cand, key=lambda fr: fr.distance_m)
