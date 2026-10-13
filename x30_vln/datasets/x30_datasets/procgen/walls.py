"""Wall segments from a plan: union of room boundaries minus door gaps."""
from __future__ import annotations

from dataclasses import dataclass

from .layout import WALL_T, Plan

JAMB = 0.06  # door frame post width


@dataclass
class WallSeg:
    orient: str  # 'h' runs along x at y=line; 'v' runs along y at x=line
    line: float
    t0: float
    t1: float

    def rect(self, thickness: float = WALL_T) -> tuple:
        h = thickness / 2.0
        if self.orient == "h":
            return (self.t0, self.line - h, self.t1, self.line + h)
        return (self.line - h, self.t0, self.line + h, self.t1)


def _union(intervals: list[tuple]) -> list[tuple]:
    out: list = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def wall_segments(plan: Plan, thickness: float = WALL_T) -> list[WallSeg]:
    lines: dict = {}
    for r in plan.rooms:
        if not r.walled:
            continue
        x0, y0, x1, y1 = r.rect
        for key, lo, hi in (
            (("h", round(y0, 4)), x0, x1), (("h", round(y1, 4)), x0, x1),
            (("v", round(x0, 4)), y0, y1), (("v", round(x1, 4)), y0, y1),
        ):
            lines.setdefault(key, []).append((lo, hi))
    segs: list[WallSeg] = []
    for (orient, line), ivs in sorted(lines.items()):
        cuts = sorted(
            (d.center - d.width / 2 - JAMB, d.center + d.width / 2 + JAMB)
            for d in plan.doors
            if d.orient == orient and abs(d.line - line) < 1e-3
        )
        for lo, hi in _union(ivs):
            inside = [c for c in cuts if c[0] >= lo - 1e-6 and c[1] <= hi + 1e-6]
            pos = lo
            pieces = []
            for a, b in inside:
                if a - pos > 1e-6:
                    pieces.append((pos, a))
                pos = b
            if hi - pos > 1e-6:
                pieces.append((pos, hi))
            for a, b in pieces:
                # extend only the outer ends so corners close; door cuts stay exact
                segs.append(WallSeg(orient, line, a - thickness / 2 if a == lo else a,
                                    b + thickness / 2 if b == hi else b))
    return segs
