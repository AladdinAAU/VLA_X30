"""Per-room-type furnishing recipes. Each recipe places items through a :class:`Placer`."""
from __future__ import annotations

import math
import random
from typing import Callable

from .furniture import make
from .layout import Plan, Room
from .placer import Placed, Placer

TABLE_IGNORE = ("chair", "office_chair")


def _p(P: Placer, rng: random.Random, prob: float) -> bool:
    return rng.random() < prob


def surround(P: Placer, room: Room, table: Placed, rng: random.Random, n: int) -> None:
    """Chairs facing the table on its two long sides, then the short ends."""
    W, L = table.item.footprint
    per_side = max(1, (n + 1) // 2)
    spacing = L / per_side
    count = 0
    for sign in (1, -1):
        for k in range(per_side):
            if count >= n:
                return
            lat = (k + 0.5) * spacing - L / 2
            kind = "chair"
            ch = P.relative(table, room, make(kind, rng), fwd=sign * (W / 2 + 0.26), lat=lat,
                            rel_yaw=math.pi if sign > 0 else 0.0, margin=0.0, ignore=("table",), check_clearance=False)
            count += 1 if ch else 0


def window_and_radiator(P: Placer, room: Room, rng: random.Random, n_windows: int | None = None) -> None:
    sides = P.exterior_sides(room)
    if not sides:
        return
    n = n_windows if n_windows is not None else (2 if room.area > 20 else 1)
    for _ in range(n):
        w = P.along_wall(room, make("window", rng), sides=sides, margin=0.3)
        if w:
            along = w.x if w.yaw in (math.pi / 2, -math.pi / 2) else w.y
            side = "S" if w.yaw == math.pi / 2 else "N" if w.yaw == -math.pi / 2 else "W" if w.yaw == 0 else "E"
            if _p(P, rng, 0.8):
                P.along_wall(room, make("radiator", rng), sides=side, at=along, tries=3)


def pictures(P: Placer, room: Room, rng: random.Random, n: int) -> None:
    for _ in range(n):
        P.along_wall(room, make("picture", rng), tries=10, margin=0.4)


def maybe_person(P: Placer, room: Room, rng: random.Random, prob: float) -> None:
    if _p(P, rng, prob):
        P.anywhere(room, make("person", rng), margin=0.4)


def living_room(P: Placer, room: Room, rng: random.Random, open_plan: bool = False) -> None:
    sofa = P.along_wall(room, make("sofa", rng))
    if sofa:
        tbl = P.relative(sofa, room, make("coffee_table", rng), fwd=1.05, ignore=("sofa", "rug"))
        if tbl:
            P.try_place(room, make("rug", rng), tbl.x, tbl.y, tbl.yaw, margin=0.0)
            if _p(P, rng, 0.8):
                P.relative(sofa, room, make("armchair", rng), fwd=1.0, lat=sofa.item.footprint[1] / 2 + 0.3,
                           rel_yaw=-math.pi / 2, ignore=("sofa", "coffee_table", "rug"), margin=0.0)
    stand = P.along_wall(room, make("tv_stand", rng))
    if stand:
        P.on_top(stand, room, make("tv", rng, big=True))
    P.along_wall(room, make("bookshelf", rng))
    P.corner(room, make("plant", rng))
    if _p(P, rng, 0.7):
        P.corner(room, make("lamp", rng))
    window_and_radiator(P, room, rng)
    pictures(P, room, rng, 2)
    maybe_person(P, room, rng, 0.35)
    if open_plan:
        kitchen_run(P, room, rng, dining=room.area > 24)


def kitchen_run(P: Placer, room: Room, rng: random.Random, dining: bool = True) -> None:
    P.corner(room, make("refrigerator", rng))
    counter = P.along_wall(room, make("kitchen_counter", rng))
    if counter:
        P.on_top(counter, room, make("microwave", rng))
    P.along_wall(room, make("stove", rng))
    P.along_wall(room, make("sink", rng, kitchen=True))
    if dining and room.area > 9:
        t = P.center(room, make("table", rng), spread=0.08)
        if t:
            surround(P, room, t, rng, rng.choice([2, 4]))
    P.along_wall(room, make("trash_can", rng), tries=10)


def kitchen(P: Placer, room: Room, rng: random.Random) -> None:
    kitchen_run(P, room, rng)
    window_and_radiator(P, room, rng, 1)
    maybe_person(P, room, rng, 0.2)


def bedroom(P: Placer, room: Room, rng: random.Random) -> None:
    bed = P.along_wall(room, make("bed", rng))
    if bed:
        wd = bed.item.footprint[1]
        for sign in (1, -1):
            if sign == 1 or _p(P, rng, 0.7):
                ns = P.relative(bed, room, make("nightstand", rng), fwd=-(bed.item.footprint[0] / 2 - 0.2),
                                lat=sign * (wd / 2 + 0.25), margin=0.0, ignore=("bed",))
                if ns and _p(P, rng, 0.5):
                    P.on_top(ns, room, make("laptop", rng))
        P.try_place(room, make("rug", rng), bed.x + math.cos(bed.yaw) * 0.9, bed.y + math.sin(bed.yaw) * 0.9,
                    bed.yaw, margin=0.0)
    P.along_wall(room, make("wardrobe", rng))
    if _p(P, rng, 0.7) and room.area > 10:
        d = P.along_wall(room, make("desk", rng))
        if d:
            P.relative(d, room, make("office_chair", rng), fwd=0.55, rel_yaw=math.pi, margin=0.0, ignore=("desk",),
                       check_clearance=False)
            P.on_top(d, room, make("laptop", rng))
    P.corner(room, make("plant", rng))
    if _p(P, rng, 0.5):
        P.corner(room, make("lamp", rng))
    window_and_radiator(P, room, rng, 1)
    pictures(P, room, rng, 1)
    maybe_person(P, room, rng, 0.15)


def bathroom(P: Placer, room: Room, rng: random.Random) -> None:
    P.along_wall(room, make("toilet", rng))
    P.along_wall(room, make("sink", rng, kitchen=False))
    if room.area > 6 and _p(P, rng, 0.6):
        P.along_wall(room, make("bathtub", rng))
    else:
        P.along_wall(room, make("washing_machine", rng))
    P.along_wall(room, make("trash_can", rng), tries=10)
    P.along_wall(room, make("mirror", rng), tries=10)


def corridor(P: Placer, room: Room, rng: random.Random, wide: bool) -> None:
    P.along_wall(room, make("fire_extinguisher", rng), sides="SN")
    if wide:
        P.along_wall(room, make("water_cooler", rng), sides="SN")
        P.along_wall(room, make("plant", rng), sides="SN")
        P.along_wall(room, make("trash_can", rng), sides="SN", tries=10)
        if _p(P, rng, 0.5):
            P.along_wall(room, make("cabinet", rng, tall=False), sides="SN")
        if _p(P, rng, 0.4):
            P.along_wall(room, make("person", rng), sides="SN", margin=0.5)
    else:
        P.along_wall(room, make("cabinet", rng, tall=False), sides="SN", tries=10)
        P.along_wall(room, make("plant", rng), sides="SN", tries=10)
    pictures(P, room, rng, 3)


def desk_setup(P: Placer, room: Room, rng: random.Random, desk: Placed) -> None:
    P.relative(desk, room, make("office_chair", rng), fwd=0.55, rel_yaw=math.pi, margin=0.0, ignore=("desk",),
               check_clearance=False)
    if desk.item.footprint[1] >= 1.2:
        P.on_top(desk, room, make("monitor", rng))
    if _p(P, rng, 0.4):
        P.on_top(desk, room, make("laptop", rng))


def office(P: Placer, room: Room, rng: random.Random) -> None:
    desk = P.along_wall(room, make("desk", rng))
    if desk:
        desk_setup(P, room, rng, desk)
    if room.area > 12:
        d2 = P.along_wall(room, make("desk", rng))
        if d2:
            desk_setup(P, room, rng, d2)
    P.along_wall(room, make("filing_cabinet", rng))
    P.along_wall(room, make("bookshelf", rng))
    P.corner(room, make("plant", rng))
    P.along_wall(room, make("whiteboard", rng), tries=10, margin=0.3)
    P.along_wall(room, make("trash_can", rng), tries=10)
    window_and_radiator(P, room, rng, 1)
    maybe_person(P, room, rng, 0.3)


def open_office(P: Placer, room: Room, rng: random.Random) -> None:
    x0, y0, x1, y1 = P.usable(room)
    pitch_x, pitch_y = 3.4, 3.2
    nx = max(1, int((x1 - x0 - 1.6) // pitch_x))
    ny = max(1, int((y1 - y0 - 1.6) // pitch_y))
    for i in range(nx):
        for j in range(ny):
            x = x0 + 0.9 + 0.8 + i * pitch_x + rng.uniform(-0.1, 0.1)
            y = y0 + 1.2 + j * pitch_y
            d = P.try_place(room, make("desk", rng), x, y, math.pi / 2, margin=0.1)
            if d:
                desk_setup(P, room, rng, d)
            d2 = P.try_place(room, make("desk", rng), x, y + 1.35, -math.pi / 2, margin=0.1)
            if d2:
                desk_setup(P, room, rng, d2)
            if d2 and rng.random() < 0.4:
                P.relative(d2, room, make("person", rng), fwd=1.0, lat=0.9, margin=0.2, check_clearance=False)
    P.along_wall(room, make("printer", rng))
    P.along_wall(room, make("water_cooler", rng))
    P.corner(room, make("plant", rng))
    P.corner(room, make("plant", rng))
    P.along_wall(room, make("trash_can", rng), tries=10)
    P.along_wall(room, make("whiteboard", rng), tries=10, margin=0.3)
    window_and_radiator(P, room, rng)


def meeting_room(P: Placer, room: Room, rng: random.Random) -> None:
    t = P.center(room, make("table", rng, length=min(3.2, room.rect[2] - room.rect[0] - 2.0), width=1.2),
                 spread=0.04, yaw=0.0 if (room.rect[2] - room.rect[0]) >= (room.rect[3] - room.rect[1]) else math.pi / 2)
    if t:
        surround(P, room, t, rng, rng.choice([6, 8]))
    P.along_wall(room, make("whiteboard", rng), tries=10, margin=0.3)
    stand = P.along_wall(room, make("cabinet", rng, tall=False))
    if stand:
        P.on_top(stand, room, make("tv", rng))
    P.corner(room, make("plant", rng))
    window_and_radiator(P, room, rng, 1)


def pantry(P: Placer, room: Room, rng: random.Random) -> None:
    kitchen_run(P, room, rng, dining=True)
    P.along_wall(room, make("water_cooler", rng))


def restroom(P: Placer, room: Room, rng: random.Random) -> None:
    for _ in range(3):
        P.along_wall(room, make("toilet", rng), margin=0.25)
    for _ in range(2):
        P.along_wall(room, make("sink", rng, kitchen=False), margin=0.2)
    P.along_wall(room, make("trash_can", rng), tries=10)
    P.along_wall(room, make("mirror", rng), tries=10)


def server_room(P: Placer, room: Room, rng: random.Random) -> None:
    for _ in range(5):
        P.along_wall(room, make("server_rack", rng), margin=0.3)
    P.along_wall(room, make("fire_extinguisher", rng), tries=15)
    P.along_wall(room, make("printer", rng), tries=10)


def lab(P: Placer, room: Room, rng: random.Random) -> None:
    n = 2 if room.area > 20 else 1
    cx, cy = room.center
    horizontal = (room.rect[2] - room.rect[0]) >= (room.rect[3] - room.rect[1])
    for k in range(n):
        off = (k - (n - 1) / 2) * 2.4
        x, y = (cx, cy + off) if horizontal else (cx + off, cy)
        bench = P.try_place(room, make("workbench", rng), x, y, math.pi / 2 if horizontal else 0.0, margin=0.1)
        if bench:
            P.on_top(bench, room, make("monitor", rng))
            P.relative(bench, room, make("office_chair", rng), fwd=0.7, rel_yaw=math.pi, margin=0.0,
                       ignore=("workbench",), check_clearance=False)
            if rng.random() < 0.5:
                P.on_top(bench, room, make("laptop", rng))
    P.along_wall(room, make("cabinet", rng, tall=True))
    P.along_wall(room, make("cabinet", rng, tall=False))
    P.along_wall(room, make("server_rack", rng), margin=0.3)
    P.along_wall(room, make("whiteboard", rng), tries=10, margin=0.3)
    P.along_wall(room, make("fire_extinguisher", rng), tries=15)
    P.along_wall(room, make("trash_can", rng), tries=10)
    maybe_person(P, room, rng, 0.5)
    window_and_radiator(P, room, rng, 1)


def storage(P: Placer, room: Room, rng: random.Random) -> None:
    for _ in range(3):
        P.along_wall(room, make("cabinet", rng, tall=True), margin=0.2)
    for _ in range(4):
        P.anywhere(room, make("box", rng), margin=0.3)
    P.along_wall(room, make("fire_extinguisher", rng), tries=15)


def warehouse(P: Placer, plan: Plan, rng: random.Random) -> None:
    hall = plan.room("warehouse_hall_1")
    x0, y0, x1, y1 = P.usable(hall)
    row_y = y0 + 8.0
    pitch = 4.2
    cross = (x0 + x1) / 2
    while row_y + 0.9 < y1 - 1.5:
        x = x0 + 3.0
        while x + 2.7 < x1 - 2.0:
            if abs(x + 1.35 - cross) < 1.9:
                x = cross + 1.9
                continue
            for dy, yaw in ((0.0, -math.pi / 2), (0.9, math.pi / 2)):
                if row_y + dy + 0.45 < y1 - 0.9 and rng.random() < 0.97:
                    P.try_place(hall, make("shelf", rng, filled=rng.uniform(0.4, 0.9)), x + 1.35, row_y + dy, yaw,
                                margin=0.0, check_clearance=False)
            x += 2.8
        row_y += pitch
    # loading area: pallets with cartons, barrels, carts
    for _ in range(rng.randint(3, 5)):
        pal = P.anywhere(hall, make("pallet", rng), margin=0.8, tries=60)
        if pal:
            for _k in range(rng.randint(1, 3)):
                P.on_top(pal, hall, make("box", rng, size=rng.uniform(0.35, 0.5)))
    for _ in range(rng.randint(3, 5)):
        P.anywhere(hall, make("barrel", rng), margin=0.5, tries=60)
    for _ in range(2):
        P.anywhere(hall, make("cart", rng), margin=0.8, tries=60)
    for _ in range(rng.randint(2, 4)):
        P.anywhere(hall, make("box", rng), margin=0.6, tries=60)
    for _ in range(2):
        P.along_wall(hall, make("fire_extinguisher", rng), tries=40)
    for _ in range(rng.randint(2, 3)):
        P.anywhere(hall, make("person", rng), margin=0.6, tries=60)
    P.along_wall(hall, make("trash_can", rng), tries=20)
    off = plan.room("office_1")
    office(P, off, rng)


RECIPES: dict[str, Callable] = {
    "living_room": living_room,
    "living_kitchen": lambda P, r, rng: living_room(P, r, rng, open_plan=True),
    "kitchen": kitchen,
    "bedroom": bedroom,
    "bathroom": bathroom,
    "corridor": lambda P, r, rng: corridor(P, r, rng, wide=(min(r.rect[2] - r.rect[0], r.rect[3] - r.rect[1]) >= 1.9)),
    "hall": lambda P, r, rng: corridor(P, r, rng, wide=False),
    "office": office,
    "open_office": open_office,
    "meeting_room": meeting_room,
    "pantry": pantry,
    "restroom": restroom,
    "server_room": server_room,
    "lab": lab,
    "storage": storage,
}


def furnish(P: Placer, plan: Plan, rng: random.Random) -> None:
    if plan.kind == "warehouse":
        warehouse(P, plan, rng)
        return
    for room in plan.rooms:
        fn = RECIPES.get(room.type)
        if fn:
            fn(P, room, rng)
