"""Scene grids and optimal path lengths (for SPL), with caching."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import project_root, success_radius
from .episodes import Episode
from .geometry import point_to_box_distance
from .grid import OccGrid, load_map_yaml, shortest_path
from .metrics import goal_distance_mode, goal_instances


def _box_dist(xs: np.ndarray, ys: np.ndarray, o: dict, mode: str) -> np.ndarray:
    if mode == "surface" and o.get("sx") and o.get("sy"):
        dx = np.maximum(np.abs(xs - o["x"]) - o["sx"] / 2.0, 0.0)
        dy = np.maximum(np.abs(ys - o["y"]) - o["sy"] / 2.0, 0.0)
        return np.hypot(dx, dy)
    return np.hypot(xs - o["x"], ys - o["y"])


class OptimumProvider:
    """Loads the GT grid of a scene (first existing ``shortest_path.grid_paths`` entry, or a grid registered
    in code) and returns the shortest path from the episode start to the success region of its goal."""

    def __init__(self, cfg: dict, root: Path | None = None, grids: dict[str, OccGrid] | None = None):
        self.cfg = cfg
        self.root = root or project_root()
        self._grids: dict[str, OccGrid | None] = dict(grids or {})
        self._cache: dict[tuple, tuple[float | None, str]] = {}

    def grid_for(self, scene: str) -> OccGrid | None:
        if scene not in self._grids:
            self._grids[scene] = None
            for pattern in self.cfg["shortest_path"]["grid_paths"]:
                p = self.root / pattern.format(scene=scene)
                if p.exists():
                    self._grids[scene] = load_map_yaml(p)
                    break
        return self._grids[scene]

    def optimum(self, ep: Episode, gt_objects: list[dict]) -> tuple[float | None, str]:
        """(length_m, basis) with basis in {'episode', 'grid', 'euclidean', 'unavailable'}."""
        if ep.shortest_path_m is not None:
            return ep.shortest_path_m, "episode"
        sc = self.cfg["success"]
        radius = success_radius(self.cfg, ep.goal)
        mode = goal_distance_mode(ep.goal, sc["object_distance_mode"])
        key = (ep.scene, ep.start["x"], ep.start["y"], repr(sorted(ep.goal.items())), radius, mode)
        if key in self._cache:
            return self._cache[key]
        out = self._compute(ep, gt_objects, radius, mode)
        self._cache[key] = out
        return out

    def _compute(self, ep: Episode, gt_objects: list[dict], radius: float, mode: str) -> tuple[float | None, str]:
        start = (ep.start["x"], ep.start["y"])
        goal = ep.goal
        if goal["type"] == "pose":
            target_pts = [(goal["x"], goal["y"])]
            region = lambda xs, ys: np.hypot(xs - goal["x"], ys - goal["y"]) <= radius  # noqa: E731
        else:
            inst = goal_instances(goal, gt_objects, self.cfg["success"].get("synonyms"))
            if not inst:
                return None, "unavailable"
            target_pts = [(o["x"], o["y"]) for o in inst]
            region = lambda xs, ys: np.min([_box_dist(xs, ys, o, mode) for o in inst], axis=0) <= radius  # noqa: E731
        grid = self.grid_for(ep.scene)
        if grid is not None:
            res = shortest_path(grid, start, region, inflate_m=self.cfg["shortest_path"]["inflate_m"])
            if res is not None:
                return res[0], "grid"
        hint = ep.extra.get("geodesic_hint_m")
        if hint is not None:
            return float(hint), "dataset_geodesic"           # computed by datasets/ on the same scene; to the goal point, not the success region
        if self.cfg["shortest_path"]["fallback"] == "euclidean":
            d = min(math.hypot(start[0] - x, start[1] - y) for x, y in target_pts)
            if goal["type"] == "object":
                d = min(point_to_box_distance(start, (o["x"], o["y"]), (o.get("sx") or 0.0, o.get("sy") or 0.0)) for o in inst)
            return max(d - radius, 0.0), "euclidean"
        return None, "unavailable"
