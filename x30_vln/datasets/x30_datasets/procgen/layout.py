"""Floor-plan generation: axis-aligned rooms around a spine corridor, or a warehouse hall.

A plan is a list of rectangular rooms and door openings. Walls are derived later from the room
boundaries (union of all boundary segments, minus door gaps). Every door opening is at least
``MIN_DOOR`` wide (the X30 is 0.98 m long, 0.4 m wide).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

MIN_DOOR = 1.0
WALL_T = 0.15

PREF_WIDTH = {
    "living_room": 5.0, "kitchen": 3.6, "bedroom": 3.6, "bathroom": 2.5, "living_kitchen": 6.5,
    "office": 3.4, "meeting_room": 5.0, "pantry": 3.4, "restroom": 3.0, "server_room": 3.0,
    "lab": 5.5, "storage": 3.6, "open_office": 9.0,
}
MIN_WIDTH = 2.4


@dataclass
class Room:
    id: str
    type: str
    rect: tuple  # x0, y0, x1, y1
    name: str = ""
    walled: bool = True  # False: virtual region (zone) without its own walls

    @property
    def center(self) -> tuple:
        return (self.rect[0] + self.rect[2]) / 2.0, (self.rect[1] + self.rect[3]) / 2.0

    @property
    def area(self) -> float:
        return (self.rect[2] - self.rect[0]) * (self.rect[3] - self.rect[1])

    def contains(self, x: float, y: float) -> bool:
        return self.rect[0] <= x <= self.rect[2] and self.rect[1] <= y <= self.rect[3]


@dataclass
class Door:
    id: str
    orient: str  # 'h': wall runs along x at y=line (opening spans x); 'v': wall runs along y at x=line
    line: float
    center: float
    width: float
    rooms: tuple  # (room id a, room id b or "outside")
    height: float = 2.1

    @property
    def xy(self) -> tuple:
        return (self.center, self.line) if self.orient == "h" else (self.line, self.center)


@dataclass
class Plan:
    kind: str
    rooms: list = field(default_factory=list)
    doors: list = field(default_factory=list)
    wall_height: float = 2.6
    outer: tuple = (0, 0, 0, 0)
    spawn: tuple = (0.0, 0.0, 0.0)  # x, y, yaw suggestion

    def room(self, rid: str) -> Room:
        return next(r for r in self.rooms if r.id == rid)

    def room_at(self, x: float, y: float) -> Room | None:
        """Smallest room or zone containing the point."""
        hits = [r for r in self.rooms if r.contains(x, y)]
        return min(hits, key=lambda r: r.area) if hits else None


def _finish_names(plan: Plan) -> None:
    counts: dict = {}
    for r in plan.rooms:
        counts[r.type] = counts.get(r.type, 0) + 1
    seen: dict = {}
    for r in plan.rooms:
        seen[r.type] = seen.get(r.type, 0) + 1
        r.id = f"{r.type}_{seen[r.type]}"
        label = r.type.replace("_", " ")
        r.name = label if counts[r.type] == 1 else f"{label} {seen[r.type]}"


def _split_widths(rng: random.Random, types: list, total: float) -> list[float]:
    w = [PREF_WIDTH.get(t, 3.5) * rng.uniform(0.9, 1.15) for t in types]
    s = total / sum(w)
    w = [x * s for x in w]
    # lift narrow rooms to MIN_WIDTH by taking from the widest
    for _ in range(10):
        small = [i for i, x in enumerate(w) if x < MIN_WIDTH]
        if not small:
            break
        deficit = sum(MIN_WIDTH - w[i] for i in small)
        for i in small:
            w[i] = MIN_WIDTH
        big = max(range(len(w)), key=lambda i: w[i])
        w[big] -= deficit
    return w


def shared_edge(a: tuple, b: tuple, eps: float = 1e-6):
    """Shared boundary of two rects: (orient, line, lo, hi) or None."""
    if abs(a[3] - b[1]) < eps or abs(b[3] - a[1]) < eps:
        line = a[3] if abs(a[3] - b[1]) < eps else a[1]
        lo, hi = max(a[0], b[0]), min(a[2], b[2])
        if hi - lo > eps:
            return "h", line, lo, hi
    if abs(a[2] - b[0]) < eps or abs(b[2] - a[0]) < eps:
        line = a[2] if abs(a[2] - b[0]) < eps else a[0]
        lo, hi = max(a[1], b[1]), min(a[3], b[3])
        if hi - lo > eps:
            return "v", line, lo, hi
    return None


def connect(plan: Plan, rng: random.Random, a: Room, b: Room, width: float = 1.1, margin: float = 0.7) -> Door | None:
    edge = shared_edge(a.rect, b.rect)
    if edge is None:
        return None
    orient, line, lo, hi = edge
    width = max(width, MIN_DOOR)
    span = hi - lo
    if span < width + 0.4:
        return None
    lo_c, hi_c = lo + margin + width / 2, hi - margin - width / 2
    center = (lo + hi) / 2 if hi_c <= lo_c else rng.uniform(lo_c, hi_c)
    d = Door(f"door_{len(plan.doors) + 1:02d}", orient, line, center, width, (a.id, b.id))
    plan.doors.append(d)
    return d


def spine_plan(
    rng: random.Random,
    kind: str,
    length: float,
    corridor: float,
    depth_n: float,
    depth_s: float,
    north: list,
    south: list,
    end_room: tuple | None = None,
    cross: dict | None = None,
    corridor_type: str = "corridor",
    inner_doors: list | None = None,
    wall_height: float = 2.6,
) -> Plan:
    """Rooms on both sides of a corridor along +x.

    ``end_room=(type, width)`` adds a room closing the east end; ``cross`` adds a perpendicular
    corridor at the east end with rooms on its east side:
    ``{"width": 2.4, "types": [...], "depth": 5.0}``.
    ``inner_doors`` lists pairs ``(type_a, type_b)`` that get a connecting door when adjacent.
    """
    plan = Plan(kind, wall_height=wall_height)
    cw = corridor
    hall = Room("", corridor_type, (0.0, -cw / 2, length, cw / 2))
    plan.rooms.append(hall)
    side_rooms: dict = {}
    for side, types, depth in (("n", north, depth_n), ("s", south, depth_s)):
        widths = _split_widths(rng, types, length)
        x = 0.0
        row = []
        for t, w in zip(types, widths):
            if side == "n":
                rect = (x, cw / 2, x + w, cw / 2 + depth)
            else:
                rect = (x, -cw / 2 - depth, x + w, -cw / 2)
            r = Room("", t, rect)
            plan.rooms.append(r)
            row.append(r)
            x += w
        side_rooms[side] = row
    end = None
    if end_room:
        t, w = end_room
        y0 = -cw / 2 - depth_s
        y1 = cw / 2 + depth_n
        end = Room("", t, (length, y0, length + w, y1))
        plan.rooms.append(end)
    cross_rooms = []
    cross_hall = None
    if cross:
        cwid = cross["width"]
        y0, y1 = -cw / 2 - depth_s, cw / 2 + depth_n
        cross_hall = Room("", "corridor", (length, y0, length + cwid, y1))
        plan.rooms.append(cross_hall)
        widths = _split_widths(rng, cross["types"], y1 - y0)
        y = y0
        for t, h in zip(cross["types"], widths):
            r = Room("", t, (length + cwid, y, length + cwid + cross["depth"], y + h))
            plan.rooms.append(r)
            cross_rooms.append(r)
            y += h
    _finish_names(plan)
    hall = plan.rooms[0]
    for r in side_rooms["n"] + side_rooms["s"]:
        connect(plan, rng, hall, r, width=rng.choice([1.0, 1.1, 1.2]))
    if end:
        connect(plan, rng, hall, end, width=1.4)
    if cross_hall:
        connect(plan, rng, hall, cross_hall, width=min(2.0, corridor - 0.3), margin=0.1)
        for r in cross_rooms:
            connect(plan, rng, cross_hall, r, width=1.1)
    pairs = inner_doors or []
    for row in (side_rooms["n"], side_rooms["s"]):
        for a, b in zip(row, row[1:]):
            if any({a.type, b.type} == set(p) for p in pairs):
                connect(plan, rng, a, b, width=1.2, margin=0.5)
    plan.outer = (
        0.0,
        -cw / 2 - depth_s,
        length + (end_room[1] if end_room else 0.0) + (cross["width"] + cross["depth"] if cross else 0.0),
        cw / 2 + depth_n,
    )
    plan.spawn = (1.2, 0.0, 0.0)
    return plan


def warehouse_plan(rng: random.Random, length: float, width: float, office: tuple = (6.0, 4.5)) -> Plan:
    """One large hall with a small office in the north-west corner and a wide dock opening in the south wall."""
    plan = Plan("warehouse", wall_height=5.0)
    ow, od = office
    plan.rooms.append(Room("", "office", (0.0, width - od, ow, width)))
    plan.rooms += [
        Room("", "loading_area", (0.0, 0.0, length, 6.0), walled=False),
        Room("", "storage_area", (0.0, 6.0, length / 2, width - od), walled=False),
        Room("", "storage_area", (length / 2, 6.0, length, width), walled=False),
        Room("", "storage_area", (ow, width - od, length / 2, width), walled=False),
    ]
    hall = Room("", "warehouse_hall", (0.0, 0.0, length, width))
    plan.rooms.append(hall)
    _finish_names(plan)
    off = plan.rooms[0]
    # office door opens onto the hall: the south wall of the office
    d = Door("door_01", "h", off.rect[1], off.rect[0] + ow * rng.uniform(0.35, 0.65), 1.1, (off.id, hall.id))
    plan.doors.append(d)
    dock_x = length * rng.uniform(0.45, 0.65)
    plan.doors.append(Door("door_02", "h", 0.0, dock_x, 4.0, (hall.id, "outside"), height=3.6))
    plan.doors.append(Door("door_03", "v", length, width * rng.uniform(0.3, 0.6), 1.6, (hall.id, "outside"), height=2.4))
    plan.outer = (0.0, 0.0, length, width)
    plan.spawn = (dock_x, 2.0, 1.5708)
    return plan
