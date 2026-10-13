"""known_map (ROS-free): ground-truth occupancy grid of a world, footprint-aware A* and the velocity law of `known_map_nav`.

Frames: everything here is in the WORLD frame (the frame of the registry, the SDF and `/gt/pose`). The node converts to the odometry frame.
The grid is a bool array `blocked[row, col]` (row 0 = lowest y) with a lower-left `origin`, resampled to `res` (any-occupied rule).
"""
from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .geometry import wrap_angle

SQ2 = math.sqrt(2.0)
NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]


def load_world_grid(map_yaml: str, res: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(blocked, origin_xy) of a map_server yaml, resampled to `res` (a coarse cell is blocked when any fine cell is)."""
    import yaml
    from PIL import Image
    meta = yaml.safe_load(open(map_yaml))
    img = np.array(Image.open(os.path.join(os.path.dirname(map_yaml), meta["image"])).convert("L"), dtype=float) / 255.0
    occ = img if meta.get("negate", 0) else 1.0 - img
    blocked = occ[::-1] >= meta.get("occupied_thresh", 0.65)
    k = max(1, int(round(res / meta["resolution"])))
    h, w = -(-blocked.shape[0] // k) * k, -(-blocked.shape[1] // k) * k
    pad = np.zeros((h, w), bool)
    pad[: blocked.shape[0], : blocked.shape[1]] = blocked
    return pad.reshape(h // k, k, w // k, k).any(axis=(1, 3)), np.array(meta["origin"][:2], dtype=float)


def resolve_map(world: str, registry: str, worlds_root: str) -> tuple[str, dict]:
    """(map yaml path, spawn dict) of a registry world id."""
    import yaml
    w = yaml.safe_load(open(registry))["worlds"][world]
    return os.path.join(worlds_root, w["map_file"]), w.get("spawn", {})


@dataclass
class PlanParams:
    radii: tuple = (0.55, 0.42, 0.32)   # hard clearance tiers [m] (body half-length 0.5 + margin first; the narrow tiers need holonomic motion)
    soft: float = 0.9                    # cost is raised for cells closer than this to an obstacle
    soft_gain: float = 3.0
    snap: float = 0.8                    # start / goal snap radius [m]


class KnownMap:
    def __init__(self, blocked: np.ndarray, origin: np.ndarray, res: float, pp: PlanParams | None = None):
        self.blocked, self.origin, self.res = blocked, np.asarray(origin, float), res
        self.pp = pp or PlanParams()
        self.dist = ndimage.distance_transform_edt(~blocked) * res          # clearance to the nearest occupied cell [m]
        self.extra = np.zeros_like(blocked)                                  # temporary obstacles (blocked-path replanning)
        self.h, self.w = blocked.shape

    def cell(self, xy) -> tuple[int, int]:
        c = np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(int)
        return int(np.clip(c[0], 0, self.w - 1)), int(np.clip(c[1], 0, self.h - 1))

    def xy(self, c) -> np.ndarray:
        return self.origin + (np.asarray(c, float) + 0.5) * self.res

    def clearance(self, xy) -> float:
        c = self.cell(xy)
        return float(self.dist[c[1], c[0]])

    def add_blocker(self, xy, radius: float) -> None:
        yy, xx = np.ogrid[: self.h, : self.w]
        cx, cy = self.cell(xy)
        self.extra |= (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius / self.res) ** 2

    def clear_blockers(self) -> None:
        self.extra[:] = False

    def _trav(self, radius: float) -> np.ndarray:
        d = self.dist if not self.extra.any() else ndimage.distance_transform_edt(~(self.blocked | self.extra)) * self.res
        return (d >= radius), d

    def _snap(self, trav: np.ndarray, c):
        if trav[c[1], c[0]]:
            return c
        r = int(math.ceil(self.pp.snap / self.res))
        y0, y1, x0, x1 = max(0, c[1] - r), min(self.h, c[1] + r + 1), max(0, c[0] - r), min(self.w, c[0] + r + 1)
        ys, xs = np.nonzero(trav[y0:y1, x0:x1])
        if len(ys) == 0:
            return None
        k = int(np.argmin((ys + y0 - c[1]) ** 2 + (xs + x0 - c[0]) ** 2))
        return int(xs[k] + x0), int(ys[k] + y0)

    def astar(self, start_xy, goal_xy, radius: float) -> np.ndarray | None:
        """8-connected A* on cells with clearance >= radius; corner cutting forbidden; cost = length * (1 + gain * closeness)."""
        trav, d = self._trav(radius)
        s, g = self._snap(trav, self.cell(start_xy)), self._snap(trav, self.cell(goal_xy))
        if s is None or g is None:
            return None
        soft, gain = max(self.pp.soft, radius + 1e-3), self.pp.soft_gain
        pen = 1.0 + gain * np.clip((soft - d) / (soft - radius + 1e-6), 0.0, 1.0) ** 2
        gx, gy = g
        openq = [(0.0, 0.0, s)]
        best = {s: 0.0}
        parent: dict = {}
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur == g:
                break
            if gc > best.get(cur, 1e18):
                continue
            cx, cy = cur
            for dx, dy, l in NB:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.w and 0 <= ny < self.h) or not trav[ny, nx]:
                    continue
                if dx and dy and not (trav[cy, nx] and trav[ny, cx]):
                    continue
                ng = gc + l * 0.5 * (pen[ny, nx] + pen[cy, cx])
                if ng < best.get((nx, ny), 1e18):
                    best[(nx, ny)] = ng
                    parent[(nx, ny)] = cur
                    hh = math.hypot(nx - gx, ny - gy)
                    heapq.heappush(openq, (ng + hh, ng, (nx, ny)))
        if g not in parent and g != s:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        pts = np.array([self.xy(c) for c in cells[::-1]])
        pts[0] = np.asarray(start_xy, float)
        gc = self.cell(goal_xy)
        if trav[gc[1], gc[0]]:
            pts[-1] = np.asarray(goal_xy, float)
        return pts

    def line_free(self, a, b, trav: np.ndarray) -> bool:
        n = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / (self.res * 0.5)))
        for t in np.linspace(0, 1, n):
            c = self.cell(np.asarray(a) * (1 - t) + np.asarray(b) * t)
            if not trav[c[1], c[0]]:
                return False
        return True

    def smooth(self, path: np.ndarray, radius: float, step: float = 0.25) -> np.ndarray:
        """Greedy line-of-sight shortcutting on the clearance mask, then resampling every `step` m."""
        trav, _ = self._trav(radius)
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.line_free(path[i], path[j], trav):
                j -= 1
            out.append(path[j])
            i = j
        out = np.array(out)
        res = [out[0]]
        for a, b in zip(out[:-1], out[1:]):
            n = max(1, int(np.linalg.norm(b - a) / step))
            res.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
        return np.array(res)

    def plan(self, start_xy, goal_xy) -> tuple[np.ndarray, float] | None:
        """(smoothed path, radius used) with the widest clearance tier that connects start and goal."""
        for r in self.pp.radii:
            p = self.astar(start_xy, goal_xy, r)
            if p is not None:
                return self.smooth(p, r), r
        return None

    def reachable_from(self, xy, radius: float) -> np.ndarray:
        trav, _ = self._trav(radius)
        lab, _ = ndimage.label(trav, structure=np.ones((3, 3)))
        s = self._snap(trav, self.cell(xy))
        return (lab == lab[s[1], s[0]]) & trav if s else np.zeros_like(trav)

    def nearest_reachable(self, start_xy, goal_xy, max_dist: float = 3.0):
        """Cell reachable from start (widest clearance tier first) that is nearest to goal and within max_dist of it: (xy, radius) or None.
        Used when the goal itself is inside an obstacle, in an enclosed pocket, or otherwise unreachable (e.g. a room centre under a desk)."""
        for r in self.pp.radii:
            reach = self.reachable_from(start_xy, r)
            ys, xs = np.nonzero(reach)
            if len(xs) == 0:
                continue
            pts = self.origin + (np.stack([xs, ys], axis=1) + 0.5) * self.res
            d = np.hypot(pts[:, 0] - goal_xy[0], pts[:, 1] - goal_xy[1])
            k = int(d.argmin())
            if d[k] <= max_dist:
                return pts[k], r
        return None


def path_length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def footprint_contact(blocked: np.ndarray, origin, res, pose, length=1.0, width=0.46, shrink=0.0) -> bool:
    """True when the oriented body rectangle at pose (x, y, yaw) overlaps an occupied cell (ground truth collision metric)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    gu = np.arange(-length / 2 + shrink, length / 2 - shrink + 1e-9, res / 2)
    gv = np.arange(-width / 2 + shrink, width / 2 - shrink + 1e-9, res / 2)
    U, V = np.meshgrid(gu, gv)
    px, py = x + c * U - s * V, y + s * U + c * V
    ix, iy = np.floor((px - origin[0]) / res).astype(int), np.floor((py - origin[1]) / res).astype(int)
    ok = (ix >= 0) & (ix < blocked.shape[1]) & (iy >= 0) & (iy < blocked.shape[0])
    return bool(blocked[iy[ok], ix[ok]].any())


@dataclass
class CtrlParams:
    v_max: float = 0.45
    vy_max: float = 0.25
    wz_max: float = 0.6
    k_yaw: float = 1.2
    lookahead: float = 0.6
    slow_radius: float = 1.0       # slow down near the goal
    v_min: float = 0.08
    turn_in_place: float = 0.7     # rad
    rot_clearance: float = 0.5     # m: in-place rotation only when the clearance is at least this (body half-length)
    slow_clear_lo: float = 0.3     # clearance at which speed is at its floor
    slow_clear_hi: float = 0.9     # clearance above which speed is unrestricted
    slow_floor: float = 0.35


def control(pose, path: np.ndarray, clear: float, p: CtrlParams) -> tuple[float, float, float, float]:
    """(vx, vy, wz, distance to the path end). Rotation in place only in free space; in tight places the body translates holonomically."""
    x, y, yaw = pose
    pos = np.array([x, y])
    dist_goal = float(np.linalg.norm(path[-1] - pos))
    near = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
    seg = np.linalg.norm(np.diff(path[near:], axis=0), axis=1) if near < len(path) - 1 else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    k = int(np.searchsorted(cum, p.lookahead)) if len(cum) else 0
    target = path[min(near + k, len(path) - 1)]
    d = target - pos
    if np.linalg.norm(d) < 1e-3:
        return 0.0, 0.0, 0.0, dist_goal
    err = wrap_angle(math.atan2(d[1], d[0]) - yaw)
    wz = float(np.clip(p.k_yaw * err, -p.wz_max, p.wz_max))
    spd = p.v_max * float(np.clip((clear - p.slow_clear_lo) / (p.slow_clear_hi - p.slow_clear_lo), p.slow_floor, 1.0))
    if dist_goal < p.slow_radius:
        spd = min(spd, max(p.v_min, p.v_max * dist_goal / p.slow_radius))
    if clear >= p.rot_clearance:
        if abs(err) > p.turn_in_place:
            return 0.0, 0.0, wz, dist_goal
        return float(spd * max(0.0, math.cos(err)) ** 2), 0.0, wz, dist_goal
    # tight space: no rotation in place; slide toward the target in the body frame, turn gently while moving
    sp = min(spd, 0.25)
    vx, vy = sp * math.cos(err), sp * math.sin(err)
    return float(vx), float(np.clip(vy, -p.vy_max, p.vy_max)), float(np.clip(wz, -0.25, 0.25)), dist_goal
