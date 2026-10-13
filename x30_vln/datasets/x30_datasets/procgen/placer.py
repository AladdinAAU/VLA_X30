"""Collision-free furniture placement that never disconnects the floor plan for the robot.

Conflict rule: two placed items conflict when their footprints (axis-aligned bounding rects,
grown by a margin) overlap *and* their height ranges overlap. Rugs never conflict. After every
placement of a robot-blocking item the free space (inflated by the robot radius) is re-checked:
all door approaches must stay in one connected component, otherwise the item is rejected.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .furniture import Item
from .layout import WALL_T, Door, Plan, Room
from .walls import WallSeg, wall_segments


@dataclass
class Placed:
    name: str
    label: str
    item: Item
    x: float
    y: float
    yaw: float
    z: float
    room: str
    rect: tuple  # footprint AABB (x0, y0, x1, y1)
    parent: str | None = None

    @property
    def z_range(self) -> tuple:
        return self.z, self.z + self.item.size[2]


def footprint_rect(item: Item, x: float, y: float, yaw: float) -> tuple:
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    fx, fy = item.footprint
    hx = (fx * c + fy * s) / 2.0
    hy = (fx * s + fy * c) / 2.0
    return (x - hx, y - hy, x + hx, y + hy)


def overlap(a: tuple, b: tuple, margin: float = 0.0) -> bool:
    return not (
        a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1]
    )


def snap_yaw(yaw: float) -> float:
    return round(yaw / (math.pi / 2)) * (math.pi / 2)


class Placer:
    def __init__(self, plan: Plan, rng: random.Random, resolution: float = 0.1, robot_radius: float = 0.35):
        self.plan = plan
        self.rng = rng
        self.res = resolution
        self.r = robot_radius
        self.items: list[Placed] = []
        self.counts: dict = {}
        x0, y0, x1, y1 = plan.outer
        self.origin = (x0 - 1.5, y0 - 1.5)
        self.shape = (int(math.ceil((y1 - y0 + 3.0) / resolution)), int(math.ceil((x1 - x0 + 3.0) / resolution)))
        self.wall_rects = [w.rect(WALL_T) for w in wall_segments(plan)]
        self.keepouts: list[tuple] = []
        for d in plan.doors:
            self.keepouts += self._door_keepouts(d)
        self.base = np.zeros(self.shape, bool)
        for r in self.wall_rects:
            self._paint(self.base, r)
        self.anchors = self._anchors()
        self.mask = self.base.copy()

    # -- geometry -------------------------------------------------------------------------------
    def usable(self, room: Room) -> tuple:
        h = WALL_T / 2.0 + 0.01
        x0, y0, x1, y1 = room.rect
        return (x0 + h, y0 + h, x1 - h, y1 - h)

    def _door_keepouts(self, d: Door) -> list[tuple]:
        depth, half = 1.1, d.width / 2 + 0.3
        if d.orient == "h":
            return [(d.center - half, d.line - depth, d.center + half, d.line + depth)]
        return [(d.line - depth, d.center - half, d.line + depth, d.center + half)]

    def _anchors(self) -> list[tuple]:
        pts = []
        for d in self.plan.doors:
            off = 0.9
            if d.orient == "h":
                pts += [(d.center, d.line - off), (d.center, d.line + off)]
            else:
                pts += [(d.line - off, d.center), (d.line + off, d.center)]
        # keep only anchors inside some room (not outside the building)
        return [p for p in pts if any(r.contains(*p) for r in self.plan.rooms)]

    def _paint(self, grid: np.ndarray, rect: tuple) -> None:
        ix0 = max(0, int(math.floor((rect[0] - self.origin[0]) / self.res)))
        ix1 = min(grid.shape[1], int(math.ceil((rect[2] - self.origin[0]) / self.res)))
        iy0 = max(0, int(math.floor((rect[1] - self.origin[1]) / self.res)))
        iy1 = min(grid.shape[0], int(math.ceil((rect[3] - self.origin[1]) / self.res)))
        grid[iy0:iy1, ix0:ix1] = True

    def _cell(self, x: float, y: float) -> tuple:
        return int((y - self.origin[1]) / self.res), int((x - self.origin[0]) / self.res)

    def _connected(self, mask: np.ndarray, spawn: tuple) -> bool:
        pad = np.pad(mask, 1, constant_values=True)
        dist = ndimage.distance_transform_edt(~pad)[1:-1, 1:-1] * self.res
        free = dist >= self.r
        lab, _ = ndimage.label(free, structure=np.ones((3, 3)))
        sy, sx = self._cell(spawn[0], spawn[1])
        ref = lab[sy, sx]
        if ref == 0:
            return False
        for p in self.anchors:
            cy, cx = self._cell(*p)
            if lab[cy, cx] != ref:
                return False
        return True

    # -- placement ------------------------------------------------------------------------------
    def name_for(self, label: str) -> str:
        n = self.counts.get(label, 0) + 1
        self.counts[label] = n
        return f"{label}_{n:02d}"

    def conflicts(self, item: Item, rect: tuple, z: float, margin: float, ignore: tuple = ()) -> bool:
        z0, z1 = z, z + item.size[2]
        for o in self.items:
            if (item.label == "rug") != (o.label == "rug") or o.label in ignore:
                continue
            oz0, oz1 = o.z_range
            if oz1 <= z0 or z1 <= oz0:
                continue
            m = margin if (item.blocks and o.item.blocks) else 0.02
            if overlap(rect, o.rect, m):
                return True
        return False

    def front_rect(self, item: Item, x: float, y: float, yaw: float) -> tuple:
        r = footprint_rect(item, x, y, yaw)
        fx, fy = math.cos(yaw) * item.clearance, math.sin(yaw) * item.clearance
        r2 = (r[0] + fx, r[1] + fy, r[2] + fx, r[3] + fy)
        return (min(r[0], r2[0]), min(r[1], r2[1]), max(r[2], r2[2]), max(r[3], r2[3]))

    def try_place(
        self,
        room: Room,
        item: Item,
        x: float,
        y: float,
        yaw: float,
        z: float | None = None,
        parent: str | None = None,
        margin: float = 0.08,
        ignore: tuple = (),
        check_clearance: bool = True,
    ) -> Placed | None:
        rect = footprint_rect(item, x, y, yaw)
        ux0, uy0, ux1, uy1 = self.usable(room)
        tol = 0.02
        if parent is None and (rect[0] < ux0 - tol or rect[1] < uy0 - tol or rect[2] > ux1 + tol or rect[3] > uy1 + tol):
            return None
        zz = item.mount_z if z is None else z
        if item.blocks and zz < 1.0 and parent is None:
            if any(overlap(rect, k) for k in self.keepouts):
                return None
        if self.conflicts(item, rect, zz, margin, ignore):
            return None
        if check_clearance and item.clearance > 0 and item.blocks:
            fr = self.front_rect(item, x, y, yaw)
            for o in self.items:
                if o.item.blocks and o.label not in ignore and o.label != "rug" and o.z_range[0] < 1.0 \
                        and overlap(fr, o.rect, 0.0) and o.parent is None:
                    return None
        if item.blocks and parent is None:
            test = self.mask.copy()
            self._paint(test, rect)
            if not self._connected(test, self.plan.spawn):
                return None
            self.mask = test
        placed = Placed(self.name_for(item.label), item.label, item, x, y, yaw, zz, room.id, rect, parent)
        self.items.append(placed)
        return placed

    def along_wall(
        self,
        room: Room,
        item: Item,
        sides: str = "SNWE",
        at: float | None = None,
        tries: int = 24,
        gap: float = 0.02,
        **kw,
    ) -> Placed | None:
        ux0, uy0, ux1, uy1 = self.usable(room)
        fx, fy = item.footprint
        if item.wall_mounted:
            gap = 0.0
        order = list(sides)
        for _ in range(tries):
            side = self.rng.choice(order)
            if side in "SN":
                yaw = math.pi / 2 if side == "S" else -math.pi / 2
                hx, hy = fy / 2, fx / 2
                lo, hi = ux0 + hx, ux1 - hx
                if hi < lo:
                    continue
                x = min(max(at, lo), hi) if at is not None else self.rng.uniform(lo, hi)
                y = uy0 + hy + gap if side == "S" else uy1 - hy - gap
            else:
                yaw = 0.0 if side == "W" else math.pi
                hx, hy = fx / 2, fy / 2
                lo, hi = uy0 + hy, uy1 - hy
                if hi < lo:
                    continue
                y = min(max(at, lo), hi) if at is not None else self.rng.uniform(lo, hi)
                x = ux0 + hx + gap if side == "W" else ux1 - hx - gap
            p = self.try_place(room, item, x, y, yaw, **kw)
            if p:
                return p
        return None

    def corner(self, room: Room, item: Item, **kw) -> Placed | None:
        ux0, uy0, ux1, uy1 = self.usable(room)
        fx, fy = item.footprint
        corners = [(ux0 + fx / 2, uy0 + fy / 2, 0.0), (ux1 - fx / 2, uy0 + fy / 2, math.pi),
                   (ux0 + fx / 2, uy1 - fy / 2, 0.0), (ux1 - fx / 2, uy1 - fy / 2, math.pi)]
        self.rng.shuffle(corners)
        for x, y, yaw in corners:
            p = self.try_place(room, item, x, y, yaw, **kw)
            if p:
                return p
        return None

    def center(self, room: Room, item: Item, spread: float = 0.15, tries: int = 20, yaw: float | None = None,
               **kw) -> Placed | None:
        cx, cy = room.center
        w, h = room.rect[2] - room.rect[0], room.rect[3] - room.rect[1]
        for _ in range(tries):
            x = cx + self.rng.uniform(-spread, spread) * w
            y = cy + self.rng.uniform(-spread, spread) * h
            yw = yaw if yaw is not None else self.rng.choice([0.0, math.pi / 2])
            p = self.try_place(room, item, x, y, yw, **kw)
            if p:
                return p
        return None

    def anywhere(self, room: Room, item: Item, tries: int = 40, **kw) -> Placed | None:
        ux0, uy0, ux1, uy1 = self.usable(room)
        for _ in range(tries):
            x, y = self.rng.uniform(ux0, ux1), self.rng.uniform(uy0, uy1)
            p = self.try_place(room, item, x, y, self.rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2]), **kw)
            if p:
                return p
        return None

    def relative(self, anchor: Placed, room: Room, item: Item, fwd: float, lat: float = 0.0, rel_yaw: float = 0.0,
                 **kw) -> Placed | None:
        """Place ``item`` at (fwd, lat) in the anchor's frame; ``rel_yaw`` rotates it relative to the anchor."""
        c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
        x = anchor.x + fwd * c - lat * s
        y = anchor.y + fwd * s + lat * c
        return self.try_place(room, item, x, y, snap_yaw(anchor.yaw + rel_yaw), **kw)

    def on_top(self, parent: Placed, room: Room, item: Item, spread: float = 0.5) -> Placed | None:
        top = parent.item.sub.get("top")
        if top is None:
            return None
        pfx, pfy = parent.item.footprint
        for _ in range(12):
            lx = self.rng.uniform(-0.5, 0.5) * max(0.0, pfx - item.footprint[0] - 0.1) * spread
            ly = self.rng.uniform(-0.5, 0.5) * max(0.0, pfy - item.footprint[1] - 0.1) * spread * 2
            c, s = math.cos(parent.yaw), math.sin(parent.yaw)
            x, y = parent.x + lx * c - ly * s, parent.y + lx * s + ly * c
            p = self.try_place(room, item, x, y, parent.yaw, z=top, parent=parent.name, margin=0.02)
            if p:
                return p
        return None

    def exterior_sides(self, room: Room) -> str:
        sides = ""
        x0, y0, x1, y1 = room.rect
        probes = {
            "S": [(x0 + (x1 - x0) * f, y0 - 0.3) for f in (0.2, 0.5, 0.8)],
            "N": [(x0 + (x1 - x0) * f, y1 + 0.3) for f in (0.2, 0.5, 0.8)],
            "W": [(x0 - 0.3, y0 + (y1 - y0) * f) for f in (0.2, 0.5, 0.8)],
            "E": [(x1 + 0.3, y0 + (y1 - y0) * f) for f in (0.2, 0.5, 0.8)],
        }
        for side, pts in probes.items():
            if all(not any(r.walled and r.contains(*p) for r in self.plan.rooms) for p in pts):
                sides += side
        return sides
