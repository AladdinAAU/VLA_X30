"""Planning context of a registered world: free-space grid, ground-truth objects, rooms."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .. import gridmap as gm
from ..gridmap import GridMap
from ..paths import worlds_pkg_dir
from ..registry import world_entry

ROBOT_RADIUS = 0.35


def coarsen(grid: GridMap, factor: int) -> GridMap:
    """Block-max downsampling (a coarse cell is blocked if any fine cell in it is blocked)."""
    if factor <= 1:
        return grid
    h, w = grid.occ.shape
    h2, w2 = h // factor, w // factor
    occ = grid.occ[: h2 * factor, : w2 * factor].reshape(h2, factor, w2, factor).any(axis=(1, 3))
    return GridMap(occ, grid.resolution * factor, grid.origin)


@dataclass
class WorldContext:
    world_id: str
    grid: GridMap  # planning grid
    blocked: np.ndarray  # inflated by the robot radius
    reach: np.ndarray  # free cells connected to the spawn
    objects: list
    rooms: list
    spawn: dict
    by_name: dict = field(default_factory=dict)

    # -- helpers --------------------------------------------------------------------------------
    def cell(self, x: float, y: float) -> tuple[int, int]:
        return self.grid.world_to_cell(x, y)

    def xy(self, cell: tuple[int, int]) -> tuple[float, float]:
        return self.grid.cell_to_world(*cell)

    def room_at(self, x: float, y: float) -> dict | None:
        hits = [r for r in self.rooms if r["rect"][0] <= x <= r["rect"][2] and r["rect"][1] <= y <= r["rect"][3]]
        return min(hits, key=lambda r: r["area"]) if hits else None

    def room_name(self, x: float, y: float) -> str:
        r = self.room_at(x, y)
        return r["name"] if r else ""

    def bbox_distance_field(self, obj: dict) -> np.ndarray:
        """Distance (m) from each planning-grid cell centre to the object's footprint rectangle."""
        h, w = self.grid.shape
        xs = self.grid.origin[0] + (np.arange(w) + 0.5) * self.grid.resolution
        ys = self.grid.origin[1] + (np.arange(h) + 0.5) * self.grid.resolution
        x0, y0, _, x1, y1, _ = obj["bbox"]
        dx = np.maximum(np.maximum(x0 - xs, xs - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - ys, ys - y1), 0.0)
        return np.hypot(dx[None, :], dy[:, None])

    def path(self, a: tuple[float, float], b: tuple[float, float]) -> list[tuple[float, float]] | None:
        """Shortened world-frame path between two points over the inflated free space."""
        ca, cb = self.cell(*a), self.cell(*b)
        p = gm.astar(self.blocked, ca, cb)
        if p is None:
            return None
        p = gm.shortcut_path(self.blocked, p)
        return [self.xy(c) for c in p]


def load_context(world_id: str, robot_radius: float = ROBOT_RADIUS, plan_res: float = 0.1,
                 max_cells: int = 1_500_000) -> WorldContext:
    entry = world_entry(world_id)
    root = worlds_pkg_dir()
    grid = GridMap.load(root / entry["map_file"])
    factor = max(1, int(round(plan_res / grid.resolution)))
    while grid.occ.size // (factor * factor) > max_cells:
        factor += 1
    grid = coarsen(grid, factor)
    blocked = grid.inflated(entry.get("robot_radius_m", robot_radius))
    sp = entry["spawn"]
    sc = grid.world_to_cell(sp["x"], sp["y"])
    if not grid.in_bounds(*sc) or blocked[sc]:
        # spawn can sit within inflation on coarse grids: take the nearest free cell
        free = np.argwhere(~blocked)
        d = np.abs(free - np.array(sc)).sum(1)
        sc = tuple(free[int(np.argmin(d))])
    reach = gm.component_of(blocked, sc)
    objects = json.loads((root / entry["objects_file"]).read_text(encoding="utf-8")) if entry.get("objects_file") else []
    rooms = []
    if entry.get("rooms_file"):
        rooms = json.loads((root / entry["rooms_file"]).read_text(encoding="utf-8"))["rooms"]
    return WorldContext(world_id, grid, blocked, reach, objects, rooms, sp, {o["instance"]: o for o in objects})


def heading(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])
