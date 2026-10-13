"""Ground-truth occupancy grid in the world frame: shortest paths (for SPL) and collision checks.

The grid is a boolean array ``occ[row=y, col=x]`` with ``origin`` = world coordinates of the lower-left corner of
cell (0, 0). Maps are read from the ROS ``map_server`` YAML + PGM/PNG format or built from arrays (tests, fake world).
Unknown cells count as FREE here: this is a ground-truth map of the simulated world, not an exploration map.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import yaml

_NEIGH = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


@dataclass
class OccGrid:
    occ: np.ndarray               # bool (H, W), True = occupied
    resolution: float             # m per cell
    origin: tuple[float, float]   # world xy of the lower-left corner of cell (0, 0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.occ.shape

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        """(row, col) of a world point; may lie outside the grid."""
        return (int(math.floor((y - self.origin[1]) / self.resolution)),
                int(math.floor((x - self.origin[0]) / self.resolution)))

    def cell_center(self, r: int, c: int) -> tuple[float, float]:
        return (self.origin[0] + (c + 0.5) * self.resolution, self.origin[1] + (r + 0.5) * self.resolution)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.occ.shape[0] and 0 <= c < self.occ.shape[1]

    def occupied_at(self, x: float, y: float) -> bool:
        r, c = self.to_cell(x, y)
        return (not self.in_bounds(r, c)) or bool(self.occ[r, c])

    def inflated(self, radius_m: float) -> np.ndarray:
        """Occupancy dilated by a disk of ``radius_m`` (numpy only)."""
        k = int(math.ceil(radius_m / self.resolution))
        if k <= 0:
            return self.occ.copy()
        h, w = self.occ.shape
        pad = np.pad(self.occ, k, constant_values=False)
        out = np.zeros_like(self.occ)
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                if dr * dr + dc * dc <= k * k:
                    out |= pad[k + dr:k + dr + h, k + dc:k + dc + w]
        return out

    def min_dist_to_occupied(self, x: float, y: float, search_m: float = 1.0) -> float:
        """Distance from a point to the nearest occupied cell edge within ``search_m`` (inf if none)."""
        r0, c0 = self.to_cell(x, y)
        k = int(math.ceil(search_m / self.resolution))
        rs = slice(max(0, r0 - k), min(self.occ.shape[0], r0 + k + 1))
        cs = slice(max(0, c0 - k), min(self.occ.shape[1], c0 + k + 1))
        rr, cc = np.nonzero(self.occ[rs, cs])
        if len(rr) == 0:
            return math.inf
        cx = self.origin[0] + (cc + cs.start + 0.5) * self.resolution
        cy = self.origin[1] + (rr + rs.start + 0.5) * self.resolution
        d = np.hypot(cx - x, cy - y) - self.resolution / 2.0
        return float(max(d.min(), 0.0))


def _read_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] not in (b"\n", b""):
                pos += 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    magic, w, h, maxv = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    pos += 1
    if magic != b"P5":
        raise ValueError(f"{path}: only binary PGM (P5) supported")
    dtype = np.uint8 if maxv < 256 else ">u2"
    return np.frombuffer(data, dtype=dtype, count=w * h, offset=pos).reshape(h, w).astype(float) / maxv


def load_map_yaml(path: str | Path) -> OccGrid:
    """Load a ROS map_server map (yaml + pgm/png). Image row 0 is the top (max y)."""
    p = Path(path)
    meta = yaml.safe_load(p.read_text(encoding="utf-8"))
    img_path = (p.parent / meta["image"]).resolve()
    if img_path.suffix.lower() == ".pgm":
        img = _read_pgm(img_path)
    else:
        import matplotlib.image as mpimg  # lazy: only for png maps
        img = mpimg.imread(str(img_path))
        img = img[..., :3].mean(axis=2) if img.ndim == 3 else img
        img = img.astype(float) if img.max() <= 1.0 else img / 255.0
    negate = int(meta.get("negate", 0))
    occ_prob = img if negate else 1.0 - img
    occ = occ_prob > float(meta.get("occupied_thresh", 0.65))
    org = meta.get("origin", [0.0, 0.0, 0.0])
    return OccGrid(occ=np.flipud(occ), resolution=float(meta["resolution"]), origin=(float(org[0]), float(org[1])))


def grid_from_rects(width_m: float, height_m: float, resolution: float, rects: Sequence[Sequence[float]],
                    origin: tuple[float, float] = (0.0, 0.0), border: bool = True) -> OccGrid:
    """Build a grid from axis-aligned rectangles [x0, y0, x1, y1] (world metres); optional outer wall."""
    w, h = int(round(width_m / resolution)), int(round(height_m / resolution))
    occ = np.zeros((h, w), bool)
    g = OccGrid(occ, resolution, origin)
    for x0, y0, x1, y1 in rects:
        r0, c0 = g.to_cell(min(x0, x1), min(y0, y1))
        r1, c1 = g.to_cell(max(x0, x1), max(y0, y1))
        occ[max(r0, 0):max(r1, 0) + 1, max(c0, 0):max(c1, 0) + 1] = True
    if border:
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return g


def _line_free(free: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Bresenham line-of-sight between two cells over a boolean free mask."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = 1 if r1 > r0 else -1, 1 if c1 > c0 else -1
    err, r, c = dc - dr, r0, c0
    while True:
        if not free[r, c]:
            return False
        if (r, c) == (r1, c1):
            return True
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def shortest_path(grid: OccGrid, start_xy: Sequence[float], target: Callable[[float, float], bool] | Sequence[float],
                  inflate_m: float = 0.3) -> tuple[float, list[tuple[float, float]]] | None:
    """Shortest any-angle path length (Dijkstra on 8-neighbours, then line-of-sight shortcutting).

    ``target`` is either a goal point (reached when within one cell) or a vectorisable predicate over world xy
    arrays (e.g. "within 1.5 m of any chair"). Returns (length_m, waypoints) or None if unreachable.
    The start cell is forced free so a start next to a wall still works."""
    free = ~grid.inflated(inflate_m)
    h, w = free.shape
    s = grid.to_cell(*start_xy[:2])
    if not grid.in_bounds(*s):
        return None
    free[s] = True
    if callable(target):
        rows, cols = np.mgrid[0:h, 0:w]
        xs = grid.origin[0] + (cols + 0.5) * grid.resolution
        ys = grid.origin[1] + (rows + 0.5) * grid.resolution
        tmask = np.asarray(target(xs, ys), bool) & free
    else:
        t = grid.to_cell(target[0], target[1])
        if not grid.in_bounds(*t):
            return None
        tmask = np.zeros((h, w), bool)
        tmask[max(t[0] - 1, 0):t[0] + 2, max(t[1] - 1, 0):t[1] + 2] = True
        tmask &= free
    if not tmask.any():
        return None
    dist = np.full((h, w), np.inf)
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    dist[s] = 0.0
    pq = [(0.0, s)]
    goal = None
    while pq:
        d, (r, c) = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        if tmask[r, c]:
            goal = (r, c)
            break
        for dr, dc, cost in _NEIGH:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc]:
                continue
            if dr and dc and not (free[r + dr, c] and free[r, c + dc]):
                continue                                      # no corner cutting
            nd = d + cost * grid.resolution
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (nd, (nr, nc)))
    if goal is None:
        return None
    cells = [goal]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    cells.reverse()
    keep = [cells[0]]                                         # greedy string pulling
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_free(free, cells[i], cells[j]):
            j -= 1
        keep.append(cells[j])
        i = j
    pts = [(float(start_xy[0]), float(start_xy[1]))] + [grid.cell_center(r, c) for r, c in keep[1:]]
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    return length, pts
