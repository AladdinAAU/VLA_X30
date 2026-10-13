"""Furniture and prop catalogue built from SDF primitives.

Local item frame: origin at the centre of the footprint on the floor, ``+x`` is the front of the item
(the side a person uses it from), ``+y`` left, ``z`` up. ``size = (depth_x, width_y, height)``.
Heat sources carry ``temp_k`` (kelvin) which becomes the Gazebo thermal plugin temperature.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..sdfgen import Prim

# kelvin
AMBIENT_K = 293.15
TEMP = {
    "person": 309.15,  # skin/clothing surface of a standing person, about 36 C
    "radiator": 333.15,  # 60 C
    "server_rack": 318.15,
    "printer": 313.15,
    "monitor": 306.15,
    "laptop": 309.15,
    "tv": 304.15,
    "stove": 338.15,
    "microwave": 300.15,
    "refrigerator": 281.15,
    "water_cooler": 284.15,
    "washing_machine": 296.15,
}


@dataclass
class Item:
    label: str
    size: tuple  # (depth_x, width_y, height)
    prims: list
    color: str = "gray"
    temp_k: float | None = None
    attrs: list = field(default_factory=list)
    blocks: bool = True  # blocks the robot at body height (used by the placement checks)
    wall_mounted: bool = False
    mount_z: float = 0.0  # height of the item's bottom above the floor for wall-mounted items
    clearance: float = 0.0  # extra free space wanted in front (m)
    sub: dict = field(default_factory=dict)  # free-form parameters (e.g. desk top height)

    @property
    def footprint(self) -> tuple:
        return self.size[0], self.size[1]


def B(sx, sy, sz, x=0.0, y=0.0, zb=0.0, color="gray", collide=True, alpha=1.0, emissive=0.0) -> Prim:
    """Box with its bottom face at ``zb``."""
    return Prim("box", (sx, sy, sz), (x, y, zb + sz / 2.0), color=color, collide=collide, alpha=alpha,
                emissive=emissive)


def C(r, h, x=0.0, y=0.0, zb=0.0, color="gray", collide=True, emissive=0.0) -> Prim:
    """Vertical cylinder with its bottom face at ``zb``."""
    return Prim("cylinder", (r, h), (x, y, zb + h / 2.0), color=color, collide=collide, emissive=emissive)


def S(r, x=0.0, y=0.0, zc=0.0, color="gray", collide=True) -> Prim:
    return Prim("sphere", (r,), (x, y, zc), color=color, collide=collide)


FURNITURE_COLORS = ["red", "blue", "green", "yellow", "orange", "gray", "beige", "brown", "black", "white", "purple"]
UPHOLSTERY = ["red", "blue", "green", "gray", "beige", "brown", "orange", "purple", "black"]
WOODS = ["wood", "brown", "beige", "white", "black"]


def _legs(sx, sy, h, inset=0.04, w=0.05, color="wood") -> list[Prim]:
    ox, oy = sx / 2 - inset - w / 2, sy / 2 - inset - w / 2
    return [B(w, w, h, x, y, 0.0, color) for x in (-ox, ox) for y in (-oy, oy)]


def _envelope_collision(prims: list, sx: float, sy: float, sz: float) -> list:
    """Detail prims become visual-only; one invisible box of the full envelope is the collision.

    Keeps the physics step cheap (a 36 m warehouse otherwise has >1300 static colliders and runs at
    ~0.3 real-time factor) and a quadruped cannot enter a shelf bay anyway."""
    for p in prims:
        p.collide = False
    return prims + [Prim("box", (sx, sy, sz), (0.0, 0.0, sz / 2.0), collide=True, visible=False)]


def chair(rng: random.Random, color: str | None = None) -> Item:
    col = color or rng.choice(FURNITURE_COLORS)
    prims = _legs(0.46, 0.46, 0.43, 0.02, 0.04, "metal")
    prims += [B(0.46, 0.46, 0.05, 0, 0, 0.43, col), B(0.04, 0.46, 0.45, -0.21, 0, 0.48, col)]
    return Item("chair", (0.46, 0.46, 0.93), prims, col)


def office_chair(rng: random.Random, color: str | None = None) -> Item:
    col = color or rng.choice(["black", "gray", "blue", "red"])
    prims = [
        C(0.30, 0.04, 0, 0, 0.04, "black"),
        C(0.03, 0.40, 0, 0, 0.06, "metal"),
        B(0.48, 0.48, 0.07, 0, 0, 0.46, col),
        B(0.07, 0.46, 0.50, -0.22, 0, 0.53, col),
    ]
    return Item("chair", (0.6, 0.6, 1.03), prims, col, sub={"office": True})


def armchair(rng: random.Random, color: str | None = None) -> Item:
    col = color or rng.choice(UPHOLSTERY)
    prims = [
        B(0.80, 0.80, 0.25, 0, 0, 0.10, col),
        B(0.62, 0.50, 0.14, 0.08, 0, 0.35, col),
        B(0.18, 0.80, 0.50, -0.31, 0, 0.35, col),
        B(0.62, 0.15, 0.30, 0.08, -0.325, 0.35, col),
        B(0.62, 0.15, 0.30, 0.08, 0.325, 0.35, col),
        B(0.06, 0.06, 0.10, -0.34, -0.34, 0, "wood"), B(0.06, 0.06, 0.10, 0.34, -0.34, 0, "wood"),
        B(0.06, 0.06, 0.10, -0.34, 0.34, 0, "wood"), B(0.06, 0.06, 0.10, 0.34, 0.34, 0, "wood"),
    ]
    return Item("armchair", (0.8, 0.8, 0.85), prims, col, clearance=0.4)


def sofa(rng: random.Random, color: str | None = None, length: float | None = None) -> Item:
    col = color or rng.choice(UPHOLSTERY)
    L = length or rng.choice([1.8, 2.0, 2.2])
    prims = [
        B(0.90, L, 0.20, 0, 0, 0.08, col),
        B(0.70, L - 0.3, 0.16, 0.10, 0, 0.28, col),
        B(0.22, L, 0.50, -0.34, 0, 0.28, col),
        B(0.78, 0.15, 0.30, 0.0, -(L / 2 - 0.075), 0.28, col),
        B(0.78, 0.15, 0.30, 0.0, (L / 2 - 0.075), 0.28, col),
        B(0.06, 0.06, 0.08, -0.4, -(L / 2 - 0.1), 0, "wood"), B(0.06, 0.06, 0.08, 0.4, -(L / 2 - 0.1), 0, "wood"),
        B(0.06, 0.06, 0.08, -0.4, (L / 2 - 0.1), 0, "wood"), B(0.06, 0.06, 0.08, 0.4, (L / 2 - 0.1), 0, "wood"),
    ]
    return Item("sofa", (0.9, L, 0.85), prims, col, clearance=0.6)


def table(rng: random.Random, length: float | None = None, width: float | None = None,
          color: str | None = None) -> Item:
    col = color or rng.choice(["wood", "white", "brown", "beige", "black"])
    L = length or rng.choice([1.2, 1.4, 1.6, 1.8])
    W = width or rng.choice([0.8, 0.9])
    prims = _legs(W, L, 0.71, 0.05, 0.06, col) + [B(W, L, 0.04, 0, 0, 0.71, col)]
    return Item("table", (W, L, 0.75), prims, col, sub={"top": 0.75})


def coffee_table(rng: random.Random, color: str | None = None) -> Item:
    col = color or rng.choice(["wood", "white", "black", "brown"])
    prims = _legs(0.55, 1.0, 0.38, 0.03, 0.05, col) + [B(0.55, 1.0, 0.04, 0, 0, 0.38, col)]
    return Item("coffee_table", (0.55, 1.0, 0.42), prims, col, sub={"top": 0.42})


def desk(rng: random.Random, color: str | None = None, length: float | None = None) -> Item:
    col = color or rng.choice(["wood", "white", "beige", "gray"])
    L = length or rng.choice([1.2, 1.4, 1.6])
    D = 0.7
    prims = [
        B(D, L, 0.04, 0, 0, 0.71, col),
        B(D - 0.08, 0.04, 0.71, 0, -(L / 2 - 0.04), 0, col),
        B(D - 0.08, 0.04, 0.71, 0, (L / 2 - 0.04), 0, col),
        B(0.03, L - 0.1, 0.4, -(D / 2 - 0.05), 0, 0.3, col),
    ]
    return Item("desk", (D, L, 0.75), prims, col, clearance=0.6, sub={"top": 0.75})


def workbench(rng: random.Random, color: str | None = None) -> Item:
    col = color or "metal"
    L = rng.choice([1.8, 2.0, 2.4])
    prims = _legs(0.8, L, 0.86, 0.02, 0.06, "metal") + [
        B(0.8, L, 0.05, 0, 0, 0.86, "gray"),
        B(0.7, L - 0.1, 0.03, 0, 0, 0.25, "metal"),
    ]
    return Item("workbench", (0.8, L, 0.91), prims, col, clearance=0.5, sub={"top": 0.91})


def bed(rng: random.Random, color: str | None = None) -> Item:
    col = color or rng.choice(["blue", "gray", "beige", "red", "green"])
    W = rng.choice([1.4, 1.6])
    prims = [
        B(2.0, W, 0.30, 0, 0, 0.0, "wood"),
        B(1.96, W - 0.04, 0.20, 0.0, 0, 0.30, "white"),
        B(1.30, W - 0.02, 0.07, 0.30, 0, 0.50, col),
        B(0.06, W + 0.04, 0.95, -0.99, 0, 0.0, "wood"),
        B(0.40, 0.60, 0.12, -0.72, -W / 4, 0.50, "white"),
        B(0.40, 0.60, 0.12, -0.72, W / 4, 0.50, "white"),
    ]
    return Item("bed", (2.04, W, 0.95), prims, col, clearance=0.5)


def nightstand(rng: random.Random, color: str | None = None) -> Item:
    col = color or rng.choice(WOODS)
    prims = [B(0.4, 0.4, 0.5, 0, 0, 0.0, col), B(0.02, 0.3, 0.12, 0.2, 0, 0.3, "black", collide=False)]
    return Item("nightstand", (0.4, 0.4, 0.5), prims, col, sub={"top": 0.5})


def wardrobe(rng: random.Random, color: str | None = None) -> Item:
    col = color or rng.choice(["wood", "white", "beige", "brown"])
    W = rng.choice([1.0, 1.2, 1.5])
    prims = [B(0.6, W, 2.0, 0, 0, 0.0, col), B(0.02, 0.02, 0.4, 0.3, -0.05, 0.9, "metal", collide=False),
             B(0.02, 0.02, 0.4, 0.3, 0.05, 0.9, "metal", collide=False)]
    return Item("wardrobe", (0.6, W, 2.0), prims, col, clearance=0.6)


def bookshelf(rng: random.Random, color: str | None = None) -> Item:
    col = color or rng.choice(["wood", "white", "brown", "black"])
    W, D, H = 0.9, 0.3, rng.choice([1.6, 1.8, 2.0])
    prims = [
        B(D, 0.03, H, 0, -W / 2 + 0.015, 0, col), B(D, 0.03, H, 0, W / 2 - 0.015, 0, col),
        B(0.02, W, H, -D / 2 + 0.01, 0, 0, col),
    ]
    n = int(H / 0.38)
    for k in range(n + 1):
        prims.append(B(D, W, 0.03, 0, 0, k * (H - 0.03) / n, col))
    rng2 = random.Random(int(H * 100) + int(W * 10))
    for k in range(n):
        x = -0.02
        zb = k * (H - 0.03) / n + 0.03
        for j in range(5):
            h = rng2.uniform(0.18, 0.30)
            prims.append(B(0.2, 0.05, h, x, -0.35 + j * 0.08, zb, rng2.choice(FURNITURE_COLORS), collide=False))
    prims = _envelope_collision(prims, D, W, H)
    return Item("bookshelf", (D, W, H), prims, col, clearance=0.5)


def shelf_unit(rng: random.Random, length: float = 2.7, depth: float = 0.9, height: float = 2.4,
               filled: float = 0.7) -> Item:
    """Warehouse pallet rack: uprights, boards and cartons on the shelves."""
    prims = []
    for xi in (-depth / 2 + 0.04, depth / 2 - 0.04):
        for yi in (-length / 2 + 0.04, 0.0, length / 2 - 0.04):
            prims.append(B(0.08, 0.08, height, xi, yi, 0.0, "blue"))
    levels = [0.15, 0.85, 1.55, 2.25] if height >= 2.3 else [0.15, 0.8, 1.45]
    for zb in levels:
        prims.append(B(depth, length, 0.05, 0, 0, zb, "orange"))
    for zb in levels[:-1]:
        for k in range(3):
            if rng.random() < filled:
                w = rng.uniform(0.6, 0.8)
                prims.append(B(rng.uniform(0.5, 0.8), w, rng.uniform(0.35, 0.6), 0,
                               (k - 1) * (length / 3), zb + 0.05, rng.choice(["beige", "brown", "wood", "white"])))
    prims = _envelope_collision(prims, depth, length, height)
    return Item("shelf", (depth, length, height), prims, "blue", clearance=1.2)


def cabinet(rng: random.Random, tall: bool | None = None, color: str | None = None) -> Item:
    col = color or rng.choice(["white", "gray", "wood", "beige"])
    tall = rng.random() < 0.5 if tall is None else tall
    H = 1.8 if tall else 0.9
    W = rng.choice([0.8, 1.0, 1.2])
    prims = [B(0.4, W, H, 0, 0, 0.0, col), B(0.02, 0.02, 0.3, 0.2, -0.04, H * 0.45, "metal", collide=False)]
    return Item("cabinet", (0.4, W, H), prims, col, clearance=0.5, sub={"top": H})


def filing_cabinet(rng: random.Random) -> Item:
    col = rng.choice(["gray", "metal", "black"])
    prims = [B(0.6, 0.45, 1.3, 0, 0, 0.0, col)] + [
        B(0.02, 0.3, 0.02, 0.3, 0, 0.25 + 0.3 * k, "black", collide=False) for k in range(4)
    ]
    return Item("filing_cabinet", (0.6, 0.45, 1.3), prims, col, clearance=0.5)


def tv_stand(rng: random.Random) -> Item:
    col = rng.choice(["wood", "black", "white"])
    prims = [B(0.4, 1.4, 0.45, 0, 0, 0.0, col)]
    return Item("tv_stand", (0.4, 1.4, 0.45), prims, col, sub={"top": 0.45})


def tv(rng: random.Random, big: bool = False) -> Item:
    W = 1.1 if big else 0.9
    prims = [B(0.05, W, W * 0.56, 0, 0, 0.0, "black"), B(0.14, 0.5, 0.03, -0.02, 0, 0.0, "black", collide=False),
             B(0.01, W - 0.06, W * 0.56 - 0.06, 0.026, 0, 0.03, "blue", collide=False, emissive=0.3)]
    return Item("tv", (0.14, W, W * 0.56), prims, "black", temp_k=TEMP["tv"], attrs=["warm"])


def monitor(rng: random.Random) -> Item:
    prims = [B(0.04, 0.55, 0.33, 0, 0, 0.12, "black"), B(0.16, 0.22, 0.02, -0.02, 0, 0.0, "black", collide=False),
             B(0.03, 0.04, 0.12, -0.02, 0, 0.0, "black", collide=False),
             B(0.005, 0.5, 0.28, 0.022, 0, 0.14, "blue", collide=False, emissive=0.3)]
    return Item("monitor", (0.2, 0.55, 0.45), prims, "black", temp_k=TEMP["monitor"], attrs=["warm"])


def laptop(rng: random.Random) -> Item:
    col = rng.choice(["gray", "black", "metal"])
    prims = [B(0.24, 0.33, 0.02, 0, 0, 0.0, col), B(0.01, 0.33, 0.22, -0.115, 0, 0.02, col)]
    return Item("laptop", (0.24, 0.33, 0.24), prims, col, temp_k=TEMP["laptop"], attrs=["warm"])


def refrigerator(rng: random.Random) -> Item:
    col = rng.choice(["white", "metal", "gray"])
    prims = [B(0.7, 0.7, 1.8, 0, 0, 0.0, col), B(0.03, 0.03, 0.5, 0.36, 0.28, 1.1, "black", collide=False),
             B(0.03, 0.03, 0.3, 0.36, 0.28, 0.5, "black", collide=False),
             B(0.005, 0.68, 0.01, 0.352, 0, 0.92, "black", collide=False)]
    return Item("refrigerator", (0.7, 0.7, 1.8), prims, col, temp_k=TEMP["refrigerator"], attrs=["cold"],
                clearance=0.7)


def kitchen_counter(rng: random.Random, length: float | None = None) -> Item:
    col = rng.choice(["white", "wood", "gray", "beige"])
    L = length or rng.choice([1.8, 2.4, 3.0])
    prims = [B(0.56, L, 0.84, -0.02, 0, 0.0, col), B(0.62, L, 0.04, 0, 0, 0.86, "gray")]
    prims += [B(0.01, 0.03, 0.2, 0.28, (k - (L / 0.6) / 2 + 0.5) * 0.6, 0.55, "metal", collide=False)
              for k in range(int(L / 0.6))]
    return Item("kitchen_counter", (0.62, L, 0.9), prims, col, clearance=0.8, sub={"top": 0.9})


def sink(rng: random.Random, kitchen: bool = True) -> Item:
    if kitchen:
        prims = [B(0.6, 0.8, 0.86, 0, 0, 0.0, "white"), B(0.62, 0.8, 0.04, 0, 0, 0.86, "metal"),
                 B(0.38, 0.5, 0.01, 0.0, 0, 0.9, "gray", collide=False), B(0.05, 0.05, 0.25, -0.2, 0, 0.9, "metal", collide=False)]
        return Item("sink", (0.62, 0.8, 1.15), prims, "white", clearance=0.8, sub={"top": 0.9})
    prims = [C(0.07, 0.8, 0, 0, 0.0, "white"), B(0.45, 0.5, 0.12, 0, 0, 0.8, "white"),
             B(0.05, 0.05, 0.12, -0.17, 0, 0.92, "metal", collide=False)]
    return Item("sink", (0.45, 0.5, 0.95), prims, "white", clearance=0.6)


def stove(rng: random.Random) -> Item:
    prims = [B(0.6, 0.6, 0.86, 0, 0, 0.0, "metal"), B(0.62, 0.62, 0.04, 0, 0, 0.86, "black")]
    prims += [C(0.08, 0.01, x, y, 0.9, "gray", collide=False) for x in (-0.12, 0.12) for y in (-0.12, 0.12)]
    return Item("stove", (0.62, 0.62, 0.9), prims, "metal", temp_k=TEMP["stove"], attrs=["hot"], clearance=0.8)


def microwave(rng: random.Random) -> Item:
    prims = [B(0.35, 0.45, 0.27, 0, 0, 0.0, "black"), B(0.005, 0.3, 0.19, 0.176, -0.04, 0.04, "gray", collide=False)]
    return Item("microwave", (0.35, 0.45, 0.27), prims, "black", temp_k=TEMP["microwave"], attrs=["warm"])


def toilet(rng: random.Random) -> Item:
    prims = [B(0.2, 0.38, 0.4, -0.22, 0, 0.0, "white"), C(0.19, 0.38, 0.06, 0, 0.0, "white"),
             B(0.3, 0.36, 0.03, 0.03, 0, 0.38, "white", collide=False)]
    return Item("toilet", (0.7, 0.4, 0.8), prims, "white", clearance=0.6)


def bathtub(rng: random.Random) -> Item:
    prims = [B(0.75, 1.7, 0.55, 0, 0, 0.0, "white"), B(0.6, 1.5, 0.01, 0, 0, 0.55, "blue", collide=False)]
    return Item("bathtub", (0.75, 1.7, 0.55), prims, "white", clearance=0.4)


def washing_machine(rng: random.Random) -> Item:
    prims = [B(0.6, 0.6, 0.85, 0, 0, 0.0, "white"),
             Prim("cylinder", (0.18, 0.02), (0.30, 0, 0.45), (0, 1.5708, 0), "black", False)]
    return Item("washing_machine", (0.6, 0.6, 0.85), prims, "white", temp_k=TEMP["washing_machine"], clearance=0.7)


def radiator(rng: random.Random) -> Item:
    prims = [B(0.08, 1.0, 0.6, 0, 0, 0.15, "white")] + [B(0.1, 0.02, 0.6, 0, -0.45 + 0.1 * k, 0.15, "white", collide=False)
                                                       for k in range(10)]
    return Item("radiator", (0.1, 1.0, 0.75), prims, "white", temp_k=TEMP["radiator"], attrs=["hot"], wall_mounted=True,
                mount_z=0.15, blocks=True)


def person(rng: random.Random, color: str | None = None) -> Item:
    shirt = color or rng.choice(["red", "blue", "green", "yellow", "orange", "gray", "purple"])
    trousers = rng.choice(["black", "blue", "brown", "gray"])
    prims = [
        C(0.075, 0.85, 0, -0.09, 0.0, trousers), C(0.075, 0.85, 0, 0.09, 0.0, trousers),
        C(0.17, 0.62, 0, 0, 0.85, shirt),
        C(0.05, 0.55, 0.0, -0.22, 0.88, shirt), C(0.05, 0.55, 0.0, 0.22, 0.88, shirt),
        S(0.11, 0, 0, 1.58, "beige"),
    ]
    return Item("person", (0.45, 0.5, 1.7), prims, shirt, temp_k=TEMP["person"], attrs=["person-like", "warm"])


def plant(rng: random.Random) -> Item:
    pot = rng.choice(["brown", "white", "orange"])
    h = rng.uniform(0.3, 0.45)
    prims = [C(0.15, h, 0, 0, 0.0, pot), S(rng.uniform(0.26, 0.34), 0, 0, h + 0.3, "green"),
             S(0.18, 0.1, 0.1, h + 0.55, "green", collide=False)]
    return Item("plant", (0.6, 0.6, h + 0.65), prims, "green")


def floor_lamp(rng: random.Random) -> Item:
    prims = [C(0.12, 0.03, 0, 0, 0.0, "black"), C(0.015, 1.5, 0, 0, 0.0, "black"),
             C(0.16, 0.25, 0, 0, 1.45, "cream", emissive=0.6)]
    return Item("lamp", (0.32, 0.32, 1.7), prims, "cream")


def trash_can(rng: random.Random) -> Item:
    col = rng.choice(["gray", "black", "metal", "green", "blue"])
    return Item("trash_can", (0.32, 0.32, 0.42), [C(0.16, 0.42, 0, 0, 0.0, col)], col)


def fire_extinguisher(rng: random.Random) -> Item:
    prims = [C(0.075, 0.45, 0, 0, 0.05, "red"), C(0.025, 0.1, 0, 0, 0.5, "black"),
             B(0.2, 0.2, 0.05, 0, 0, 0.0, "gray"), B(0.1, 0.03, 0.03, 0.06, 0, 0.55, "black", collide=False)]
    return Item("fire_extinguisher", (0.2, 0.2, 0.62), prims, "red")


def water_cooler(rng: random.Random) -> Item:
    prims = [B(0.35, 0.35, 1.0, 0, 0, 0.0, "white"), C(0.14, 0.4, 0, 0, 1.0, "blue"),
             B(0.02, 0.1, 0.05, 0.18, 0, 0.9, "gray", collide=False)]
    return Item("water_cooler", (0.35, 0.35, 1.4), prims, "white", temp_k=TEMP["water_cooler"], attrs=["cold"],
                clearance=0.5)


def printer(rng: random.Random) -> Item:
    prims = [B(0.6, 0.65, 0.78, 0, 0, 0.0, "gray"), B(0.45, 0.55, 0.12, 0, 0, 0.78, "white"),
             B(0.01, 0.4, 0.06, 0.3, 0, 0.45, "black", collide=False)]
    return Item("printer", (0.6, 0.65, 0.9), prims, "gray", temp_k=TEMP["printer"], attrs=["warm"], clearance=0.6)


def server_rack(rng: random.Random) -> Item:
    prims = [B(0.9, 0.6, 2.0, 0, 0, 0.0, "black")] + [
        B(0.01, 0.5, 0.04, 0.455, 0, 0.2 + 0.18 * k, "gray", collide=False) for k in range(9)
    ] + [B(0.005, 0.02, 0.02, 0.456, 0.2, 0.2 + 0.18 * k, "green", collide=False, emissive=1.0) for k in range(9)]
    return Item("server_rack", (0.9, 0.6, 2.0), prims, "black", temp_k=TEMP["server_rack"], attrs=["hot"], clearance=0.9)


def whiteboard(rng: random.Random) -> Item:
    prims = [B(0.04, 1.5, 1.0, 0, 0, 0.0, "white"), B(0.05, 1.54, 0.03, 0, 0, 0.0, "metal", collide=False)]
    return Item("whiteboard", (0.05, 1.5, 1.0), prims, "white", wall_mounted=True, mount_z=0.9, blocks=False)


def picture(rng: random.Random) -> Item:
    col = rng.choice(["red", "blue", "green", "yellow", "orange", "purple"])
    prims = [B(0.03, 0.6, 0.45, 0, 0, 0.0, "wood", collide=False), B(0.035, 0.52, 0.37, 0, 0, 0.04, col, collide=False)]
    return Item("picture", (0.04, 0.6, 0.45), prims, col, wall_mounted=True, mount_z=1.4, blocks=False)


def mirror(rng: random.Random) -> Item:
    prims = [B(0.03, 0.6, 0.8, 0, 0, 0.0, "metal", collide=False), B(0.035, 0.54, 0.74, 0, 0, 0.03, "cream", collide=False)]
    return Item("mirror", (0.04, 0.6, 0.8), prims, "metal", wall_mounted=True, mount_z=1.0, blocks=False)


def window(rng: random.Random, width: float | None = None) -> Item:
    W = width or rng.choice([1.2, 1.5, 1.8])
    prims = [
        B(0.05, W, 1.2, 0, 0, 0.0, "white", collide=False),
        Prim("box", (0.06, W - 0.12, 1.08), (0.0, 0.0, 0.6), color=(0.6, 0.78, 0.95), collide=False, alpha=0.55),
        B(0.065, 0.04, 1.2, 0, 0, 0.0, "white", collide=False),
    ]
    return Item("window", (0.06, W, 1.2), prims, "white", wall_mounted=True, mount_z=0.9, blocks=False)


def rug(rng: random.Random) -> Item:
    col = rng.choice(["red", "blue", "green", "beige", "purple", "orange"])
    sx, sy = rng.choice([(2.0, 1.4), (2.4, 1.6), (1.6, 1.2)])
    prims = [Prim("box", (sx, sy, 0.004), (0, 0, 0.003), color=col, collide=False)]
    return Item("rug", (sx, sy, 0.005), prims, col, blocks=False)


def pallet(rng: random.Random) -> Item:
    prims = [B(1.2, 0.8, 0.04, 0, 0, 0.11, "wood")] + [
        B(1.2, 0.1, 0.11, 0, y, 0.0, "wood") for y in (-0.35, 0.0, 0.35)
    ]
    return Item("pallet", (1.2, 0.8, 0.15), prims, "wood", sub={"top": 0.15}, clearance=0.5)


def box(rng: random.Random, size: float | None = None) -> Item:
    s = size or rng.uniform(0.35, 0.6)
    col = rng.choice(["beige", "brown", "wood", "white", "orange"])
    sx, sy, sz = s, s * rng.uniform(0.8, 1.2), s * rng.uniform(0.6, 1.0)
    prims = [B(sx, sy, sz, 0, 0, 0.0, col), B(sx + 0.002, 0.05, 0.004, 0, 0, sz, "brown", collide=False)]
    return Item("box", (sx, sy, sz), prims, col, sub={"top": sz})


def barrel(rng: random.Random) -> Item:
    col = rng.choice(["blue", "red", "green", "yellow", "gray"])
    prims = [C(0.29, 0.88, 0, 0, 0.0, col), C(0.30, 0.04, 0, 0, 0.3, "black", collide=False),
             C(0.30, 0.04, 0, 0, 0.6, "black", collide=False)]
    return Item("barrel", (0.6, 0.6, 0.88), prims, col)


def cart(rng: random.Random) -> Item:
    col = rng.choice(["blue", "gray", "red", "metal"])
    prims = _legs(0.9, 0.6, 0.12, 0.0, 0.06, "black") + [
        B(0.9, 0.6, 0.04, 0, 0, 0.12, col), B(0.04, 0.6, 0.7, -0.43, 0, 0.16, col),
        B(0.7, 0.03, 0.2, 0, 0.3, 0.16, col, collide=False), B(0.7, 0.03, 0.2, 0, -0.3, 0.16, col, collide=False)]
    return Item("cart", (0.9, 0.6, 0.86), prims, col, clearance=0.5, sub={"top": 0.16})


def clutter(rng: random.Random) -> Item:
    prims = []
    for _ in range(rng.randint(3, 5)):
        s = rng.uniform(0.25, 0.4)
        prims.append(B(s, s, s * 0.8, rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.0,
                       rng.choice(["beige", "brown", "wood", "orange"])))
    return Item("clutter", (0.9, 0.9, 0.4), prims, "brown")


CATALOG = {
    "chair": chair, "office_chair": office_chair, "armchair": armchair, "sofa": sofa, "table": table,
    "coffee_table": coffee_table, "desk": desk, "workbench": workbench, "bed": bed, "nightstand": nightstand,
    "wardrobe": wardrobe, "bookshelf": bookshelf, "shelf": shelf_unit, "cabinet": cabinet,
    "filing_cabinet": filing_cabinet, "tv_stand": tv_stand, "tv": tv, "monitor": monitor, "laptop": laptop,
    "refrigerator": refrigerator, "kitchen_counter": kitchen_counter, "sink": sink, "stove": stove,
    "microwave": microwave, "toilet": toilet, "bathtub": bathtub, "washing_machine": washing_machine,
    "radiator": radiator, "person": person, "plant": plant, "lamp": floor_lamp, "trash_can": trash_can,
    "fire_extinguisher": fire_extinguisher, "water_cooler": water_cooler, "printer": printer,
    "server_rack": server_rack, "whiteboard": whiteboard, "window": window, "picture": picture, "mirror": mirror, "rug": rug,
    "pallet": pallet, "box": box, "barrel": barrel, "cart": cart, "clutter": clutter,
}


def make(kind: str, rng: random.Random, **kw) -> Item:
    return CATALOG[kind](rng, **kw)
