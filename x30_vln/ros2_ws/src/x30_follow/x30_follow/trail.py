"""Trail of the followed person and line-of-sight on the occupancy grid (no ROS imports).

follow_person drives straight at the person while the straight line to her is free. When a wall or corner is in between (she went through a door, turned a
corner) it follows HER TRAIL instead: the route she walked is known to be passable, so the goal handed to the planner is the point on the trail
d_target metres behind her, which moves with her around the corner. If she is out of sight the planner is sent to the last place she was seen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class Breadcrumbs:
    """Positions of the target at sure sightings, at least `step` metres apart, newest last; the trail is trimmed to `max_len` metres."""

    def __init__(self, step: float = 0.3, max_len: float = 40.0):
        self.step, self.max_len = step, max_len
        self.pts: list[tuple[float, float]] = []

    def clear(self) -> None:
        self.pts = []

    def add(self, x: float, y: float) -> None:
        if not self.pts or math.hypot(x - self.pts[-1][0], y - self.pts[-1][1]) >= self.step:
            self.pts.append((float(x), float(y)))
            total, k = 0.0, len(self.pts) - 1
            while k > 0 and total <= self.max_len:
                total += math.hypot(self.pts[k][0] - self.pts[k - 1][0], self.pts[k][1] - self.pts[k - 1][1])
                k -= 1
            if k > 0:
                del self.pts[:k]

    def last(self) -> tuple[float, float] | None:
        return self.pts[-1] if self.pts else None

    def point_behind(self, head: tuple[float, float], dist: float) -> tuple[float, float] | None:
        """The point `dist` metres back ALONG THE TRAIL from `head` (the person's current position). If the trail is shorter, its oldest point."""
        if not self.pts:
            return None
        remaining, cur = dist, (float(head[0]), float(head[1]))
        for p in reversed(self.pts):
            seg = math.hypot(p[0] - cur[0], p[1] - cur[1])
            if seg >= remaining and seg > 1e-9:
                f = remaining / seg
                return cur[0] + (p[0] - cur[0]) * f, cur[1] + (p[1] - cur[1]) * f
            remaining -= seg
            cur = p
        return self.pts[0]


@dataclass
class GridMap:
    data: np.ndarray                    # (H, W) int8: -1 unknown, 0..100 occupancy, row = y, col = x
    res: float
    ox: float
    oy: float

    def value(self, x: float, y: float) -> int:
        c, r = int(math.floor((x - self.ox) / self.res)), int(math.floor((y - self.oy) / self.res))
        if r < 0 or c < 0 or r >= self.data.shape[0] or c >= self.data.shape[1]:
            return -1
        return int(self.data[r, c])

    def free(self, x: float, y: float, thr: int = 50) -> bool:
        v = self.value(x, y)
        return 0 <= v < thr or v == -1 and False

    def line_free(self, a: tuple[float, float], b: tuple[float, float], thr: int = 50, step: float | None = None) -> bool:
        """True when no cell on the segment a-b is occupied (>= thr). Unknown cells count as free (the planner map has none inside the building)."""
        step = step or self.res * 0.5
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for i in range(n + 1):
            f = i / n
            if self.value(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f) >= thr:
                return False
        return True
