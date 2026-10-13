"""2D occupancy grid, inflation, A*, geodesic distance fields, Nav2-style map files."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

SQRT2 = math.sqrt(2.0)
_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class GridMap:
    """Boolean occupancy grid. ``occ[iy, ix]``; cell centre = origin + (i + 0.5) * resolution."""

    occ: np.ndarray
    resolution: float
    origin: tuple  # (x0, y0) of the lower-left corner of cell (0, 0)

    @property
    def shape(self) -> tuple:
        return self.occ.shape

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Returns (iy, ix)."""
        return (
            int(math.floor((y - self.origin[1]) / self.resolution)),
            int(math.floor((x - self.origin[0]) / self.resolution)),
        )

    def cell_to_world(self, iy: int, ix: int) -> tuple[float, float]:
        return (
            self.origin[0] + (ix + 0.5) * self.resolution,
            self.origin[1] + (iy + 0.5) * self.resolution,
        )

    def in_bounds(self, iy: int, ix: int) -> bool:
        return 0 <= iy < self.occ.shape[0] and 0 <= ix < self.occ.shape[1]

    def inflated(self, radius: float) -> np.ndarray:
        """Boolean mask of cells whose centre is closer than ``radius`` to an obstacle (or the border)."""
        padded = np.pad(self.occ, 1, constant_values=True)
        dist = ndimage.distance_transform_edt(~padded)[1:-1, 1:-1] * self.resolution
        return dist < radius

    def clearance(self) -> np.ndarray:
        """Distance (m) to the nearest obstacle/border for each cell."""
        padded = np.pad(self.occ, 1, constant_values=True)
        return ndimage.distance_transform_edt(~padded)[1:-1, 1:-1] * self.resolution

    def free_fraction(self) -> float:
        return float((~self.occ).mean())

    # -- files ------------------------------------------------------------------------------
    def save(self, yaml_path: Path, extra: dict | None = None) -> None:
        """Write ``<name>.png`` + Nav2 map_server yaml (free 254, occupied 0)."""
        yaml_path = Path(yaml_path)
        png = yaml_path.with_suffix(".png")
        img = np.where(self.occ, 0, 254).astype(np.uint8)[::-1]
        Image.fromarray(np.ascontiguousarray(img)).save(png)
        meta = {
            "image": png.name,
            "mode": "trinary",
            "resolution": float(self.resolution),
            "origin": [float(self.origin[0]), float(self.origin[1]), 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.25,
        }
        if extra:
            meta.update(extra)
        yaml_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    @staticmethod
    def load(yaml_path: Path) -> "GridMap":
        yaml_path = Path(yaml_path)
        meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        img = np.array(Image.open(yaml_path.parent / meta["image"]).convert("L"))[::-1]
        occ = img < 128
        return GridMap(occ, float(meta["resolution"]), (float(meta["origin"][0]), float(meta["origin"][1])))


def line_of_sight(blocked: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if the supercover Bresenham line between cells a and b touches no blocked cell."""
    (y0, x0), (y1, x1) = a, b
    n = max(abs(y1 - y0), abs(x1 - x0))
    if n == 0:
        return not blocked[y0, x0]
    ys = np.rint(np.linspace(y0, y1, 2 * n + 1)).astype(int)
    xs = np.rint(np.linspace(x0, x1, 2 * n + 1)).astype(int)
    return not blocked[ys, xs].any()


def astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """8-connected A* on cells (iy, ix); returns the cell path or None."""
    h, w = blocked.shape
    if blocked[start] or blocked[goal]:
        return None
    gy, gx = goal

    def heur(y: int, x: int) -> float:
        dy, dx = abs(y - gy), abs(x - gx)
        return (dy + dx) + (SQRT2 - 2.0) * min(dy, dx)

    openq: list = [(heur(*start), 0.0, start)]
    g = {start: 0.0}
    parent: dict = {}
    closed = set()
    while openq:
        _, cost, cur = heapq.heappop(openq)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        cy, cx = cur
        for dy, dx in _NEIGH:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w) or blocked[ny, nx]:
                continue
            if dy and dx and (blocked[cy, nx] or blocked[ny, cx]):
                continue  # no corner cutting
            nc = cost + (SQRT2 if dy and dx else 1.0)
            if nc < g.get((ny, nx), 1e30):
                g[(ny, nx)] = nc
                parent[(ny, nx)] = cur
                heapq.heappush(openq, (nc + heur(ny, nx), nc, (ny, nx)))
    return None


def path_length(path: list[tuple[int, int]], resolution: float) -> float:
    total = 0.0
    for (y0, x0), (y1, x1) in zip(path, path[1:]):
        total += math.hypot(y1 - y0, x1 - x0)
    return total * resolution


def shortcut_path(blocked: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy line-of-sight shortening (string pulling) of a cell path."""
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(blocked, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def distance_field(blocked: np.ndarray, sources: list[tuple[int, int]], resolution: float = 1.0) -> np.ndarray:
    """Geodesic distance (m) from the nearest source over free 8-connected cells; inf if unreachable."""
    h, w = blocked.shape
    free = ~blocked
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, vals = [], [], []
    for dy, dx in _NEIGH:
        ys0, ys1 = max(0, -dy), h - max(0, dy)
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        a = (slice(ys0, ys1), slice(xs0, xs1))
        b = (slice(ys0 + dy, ys1 + dy), slice(xs0 + dx, xs1 + dx))
        ok = free[a] & free[b]
        if dy and dx:  # forbid corner cutting
            ok &= free[ys0:ys1, xs0 + dx : xs1 + dx] & free[ys0 + dy : ys1 + dy, xs0:xs1]
        rows.append(idx[a][ok])
        cols.append(idx[b][ok])
        vals.append(np.full(ok.sum(), SQRT2 if dy and dx else 1.0))
    graph = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w)
    ).tocsr()
    src = [int(idx[s]) for s in sources if free[s]]
    if not src:
        return np.full((h, w), np.inf)
    d = dijkstra(graph, directed=True, indices=src, min_only=True)
    return d.reshape(h, w) * resolution


def largest_component(blocked: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 8-connected free component."""
    lab, n = ndimage.label(~blocked, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(blocked)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def component_of(blocked: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
    lab, _ = ndimage.label(~blocked, structure=np.ones((3, 3)))
    if lab[cell] == 0:
        return np.zeros_like(blocked)
    return lab == lab[cell]


def turn_count(points: list[tuple[float, float]], min_angle_deg: float = 25.0) -> int:
    """Number of heading changes above ``min_angle_deg`` along a polyline in world coordinates."""
    n = 0
    for p0, p1, p2 in zip(points, points[1:], points[2:]):
        a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        b = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        d = abs((b - a + math.pi) % (2 * math.pi) - math.pi)
        if math.degrees(d) >= min_angle_deg:
            n += 1
    return n
