"""Assemble a procedural world: plan -> furniture -> SDF + objects.json + rooms.json + map.

Outputs go to ``x30_worlds/worlds/<id>/``. The map is computed by parsing the written SDF back
(``sdfparse``) so it is exactly what the collision geometry of the file produces.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .. import gridmap as gm
from .. import labels
from ..paths import world_dir
from ..sdfgen import COLORS, Prim, WorldSpec, model_xml, world_xml
from ..sdfparse import parse_world, rasterize
from .furniture import AMBIENT_K
from .layout import WALL_T, Door, Plan, spine_plan, warehouse_plan
from .placer import Placed, Placer
from .recipes import furnish
from .walls import JAMB as JAMB_W
from .walls import wall_segments

ROBOT_RADIUS = 0.35
MAP_RES = 0.05
ZBAND = (0.08, 0.75)

FLOOR_COLOR = {
    "living_room": (0.62, 0.46, 0.30), "living_kitchen": (0.62, 0.46, 0.30), "bedroom": (0.66, 0.52, 0.36),
    "kitchen": (0.78, 0.76, 0.70), "bathroom": (0.70, 0.78, 0.82), "restroom": (0.70, 0.78, 0.82),
    "corridor": (0.70, 0.70, 0.68), "hall": (0.66, 0.62, 0.56), "office": (0.40, 0.45, 0.55),
    "open_office": (0.38, 0.43, 0.52), "meeting_room": (0.45, 0.40, 0.50), "pantry": (0.78, 0.76, 0.70),
    "server_room": (0.45, 0.47, 0.50), "lab": (0.82, 0.83, 0.85), "storage": (0.60, 0.58, 0.55),
    "warehouse_hall": (0.58, 0.58, 0.56),
}

PRESETS: dict[str, dict] = {
    "apt_01": dict(kind="apartment", seed=11, length=(9.8, 10.8), corridor=1.5, depth_n=4.4, depth_s=3.6,
                   north=["living_room", "kitchen"], south=["bedroom", "bathroom", "bedroom"],
                   corridor_type="hall", inner=[("living_room", "kitchen")], wall_height=2.6, wall="cream"),
    "apt_02": dict(kind="apartment", seed=22, length=(13.0, 14.0), corridor=1.6, depth_n=4.6, depth_s=3.8,
                   north=["kitchen", "living_room", "bedroom"], south=["bedroom", "bathroom", "bedroom", "bedroom"],
                   corridor_type="hall", inner=[("living_room", "kitchen")], wall_height=2.6, wall="white"),
    "apt_03": dict(kind="apartment", seed=33, length=(8.6, 9.2), corridor=1.5, depth_n=5.2, depth_s=3.6,
                   north=["living_kitchen"], south=["bedroom", "bathroom"], corridor_type="hall", inner=[],
                   wall_height=2.6, wall="cream"),
    "apt_04": dict(kind="apartment", seed=44, length=(9.0, 9.8), corridor=1.5, depth_n=4.2, depth_s=3.8,
                   north=["kitchen", "living_room"], south=["bathroom", "bedroom"], end_room=("bedroom", 3.8),
                   corridor_type="hall", inner=[("living_room", "kitchen")], wall_height=2.6, wall="white"),
    "office_01": dict(kind="office", seed=55, length=(23.0, 25.0), corridor=2.2, depth_n=4.6, depth_s=4.6,
                      north=["office", "office", "meeting_room", "pantry", "office", "office"],
                      south=["office", "office", "restroom", "server_room", "office", "storage"],
                      end_room=("open_office", 8.5), wall_height=3.0, wall="white"),
    "office_02": dict(kind="office", seed=66, length=(21.0, 23.0), corridor=2.0, depth_n=5.0, depth_s=4.4,
                      north=["open_office", "meeting_room", "meeting_room"],
                      south=["office", "office", "pantry", "restroom", "office"], wall_height=3.0, wall="cream"),
    "lab_01": dict(kind="lab", seed=77, length=(25.0, 27.0), corridor=2.4, depth_n=5.2, depth_s=5.0,
                   north=["lab", "lab", "lab", "server_room"], south=["lab", "lab", "office", "storage"],
                   wall_height=3.2, wall="white"),
    "lab_02": dict(kind="lab", seed=88, length=(17.0, 19.0), corridor=2.4, depth_n=5.0, depth_s=4.8,
                   north=["lab", "lab", "office"], south=["lab", "storage", "server_room"],
                   cross=dict(width=2.4, types=["lab", "lab", "office", "lab"], depth=5.6),
                   wall_height=3.2, wall="cream"),
    "warehouse_01": dict(kind="warehouse", seed=99, length=28.0, width=20.0, wall_height=5.0, wall="gray"),
    "warehouse_02": dict(kind="warehouse", seed=111, length=36.0, width=26.0, wall_height=6.0, wall="gray"),
}


def _plan_for(preset: dict, rng: random.Random) -> Plan:
    if preset["kind"] == "warehouse":
        plan = warehouse_plan(rng, preset["length"], preset["width"])
    else:
        length = rng.uniform(*preset["length"])
        plan = spine_plan(
            rng, preset["kind"], length, preset["corridor"], preset["depth_n"], preset["depth_s"],
            preset["north"], preset["south"], end_room=preset.get("end_room"), cross=preset.get("cross"),
            corridor_type=preset.get("corridor_type", "corridor"), inner_doors=preset.get("inner"),
            wall_height=preset["wall_height"],
        )
    plan.wall_height = preset["wall_height"]
    return plan


def _door_prims(d: Door, wall_h: float) -> list[Prim]:
    w, h = d.width, d.height
    t = WALL_T + 0.01
    prims = [
        Prim("box", (JAMB_W, t, h), (-(w / 2 + JAMB_W / 2), 0, h / 2), color="brown"),
        Prim("box", (JAMB_W, t, h), ((w / 2 + JAMB_W / 2), 0, h / 2), color="brown"),
        Prim("box", (w + 2 * JAMB_W, t, 0.1), (0, 0, h + 0.05), color="brown"),
    ]
    rest = wall_h - h - 0.1
    if rest > 0.02:
        prims.append(Prim("box", (w + 2 * JAMB_W, WALL_T, rest), (0, 0, h + 0.1 + rest / 2), color="white"))
    return prims


@dataclass
class BuildResult:
    world_id: str
    entry: dict
    objects: list
    rooms: dict


def _objects_records(plan: Plan, placed: list[Placed]) -> list[dict]:
    recs = []
    for p in placed:
        it = p.item
        x0, y0, x1, y1 = p.rect
        z0, z1 = p.z_range
        room = plan.room_at(p.x, p.y)
        attrs = list(dict.fromkeys(it.attrs + [it.color] if it.label not in ("rug",) else it.attrs))
        recs.append({
            "label": p.label,
            "label_id": labels.label_id(p.label),
            "instance": p.name,
            "pose": {"x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3), "yaw": round(p.yaw, 4)},
            "size": [round(x1 - x0, 3), round(y1 - y0, 3), round(z1 - z0, 3)],
            "bbox": [round(v, 3) for v in (x0, y0, z0, x1, y1, z1)],
            "size_local": [round(v, 3) for v in it.size],
            "room": room.id if room else "",
            "color": it.color,
            "temperature_c": round(it.temp_k - 273.15, 1) if it.temp_k else None,
            "attributes": attrs,
            "parent": p.parent,
            "blocks_robot": bool(it.blocks and p.parent is None),
            "source": "procgen",
        })
    return recs


def generate_world(world_id: str, out_dir: Path | None = None, seed: int | None = None, max_tries: int = 6) -> BuildResult:
    preset = PRESETS[world_id]
    base_seed = preset["seed"] if seed is None else seed
    out = Path(out_dir) if out_dir else world_dir(world_id)
    out.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_tries):
        rng = random.Random(base_seed + 7919 * attempt)
        plan = _plan_for(preset, rng)
        placer = Placer(plan, rng, resolution=0.1, robot_radius=ROBOT_RADIUS)
        if not placer._connected(placer.base, plan.spawn):
            continue
        furnish(placer, plan, rng)
        if len(placer.items) >= 10:
            break
    else:
        raise RuntimeError(f"{world_id}: could not generate a connected furnished plan")

    walls = wall_segments(plan)
    wh = plan.wall_height
    wall_color = preset.get("wall", "white")
    models: list[str] = []
    wall_prims = []
    for s in walls:
        x0, y0, x1, y1 = s.rect(WALL_T)
        wall_prims.append(Prim("box", (x1 - x0, y1 - y0, wh), ((x0 + x1) / 2, (y0 + y1) / 2, wh / 2), color=wall_color))
    models.append(model_xml("walls", wall_prims, label_id=labels.label_id("wall"), comment="walls with door gaps"))
    floor_prims = []
    for r in sorted((r for r in plan.rooms if r.walled), key=lambda r: -r.area):
        x0, y0, x1, y1 = r.rect
        col = FLOOR_COLOR.get(r.type, (0.6, 0.6, 0.6))
        floor_prims.append(Prim("box", (x1 - x0, y1 - y0, 0.004), ((x0 + x1) / 2, (y0 + y1) / 2, 0.002 + 0.0005 * len(floor_prims)),
                                color=col, collide=False))
    models.append(model_xml("floors", floor_prims, label_id=labels.label_id("floor"), comment="visual floor slabs"))

    door_recs = []
    for i, d in enumerate(plan.doors):
        cx, cy = d.xy
        yaw = 0.0 if d.orient == "h" else math.pi / 2
        name = f"door_{i + 1:02d}"
        models.append(model_xml(name, _door_prims(d, wh), pose=(cx, cy, 0, 0, 0, yaw), label_id=labels.label_id("door")))
        half = d.width / 2 + JAMB_W
        bb = ([cx - half, cy - WALL_T / 2, 0.0, cx + half, cy + WALL_T / 2, d.height + 0.1] if d.orient == "h"
              else [cx - WALL_T / 2, cy - half, 0.0, cx + WALL_T / 2, cy + half, d.height + 0.1])
        room_a = plan.room(d.rooms[0]).id
        door_recs.append({
            "label": "door", "label_id": labels.label_id("door"), "instance": name,
            "pose": {"x": round(cx, 3), "y": round(cy, 3), "z": 0.0, "yaw": round(yaw, 4)},
            "size": [round(bb[3] - bb[0], 3), round(bb[4] - bb[1], 3), round(bb[5] - bb[2], 3)],
            "bbox": [round(v, 3) for v in bb], "size_local": [round(d.width, 3), WALL_T, d.height],
            "room": room_a, "connects": list(d.rooms), "clear_width": round(d.width, 3), "color": "brown",
            "temperature_c": None, "attributes": ["brown"], "parent": None, "blocks_robot": False,
            "source": "procgen",
        })
    for p in placer.items:
        it = p.item
        models.append(model_xml(p.name, it.prims, pose=(p.x, p.y, p.z, 0, 0, p.yaw),
                                label_id=labels.label_id(p.label), temperature_k=it.temp_k))
    objects = door_recs + _objects_records(plan, placer.items)
    for i, o in enumerate(objects):
        o["id"] = i

    bx0, by0, bx1, by1 = plan.outer
    margin = 1.5
    bounds = (round(bx0 - margin, 2), round(by0 - margin, 2), round(bx1 + margin, 2), round(by1 + margin, 2))
    spec = WorldSpec(world_id, bounds, models, ambient_k=AMBIENT_K,
                     comment=f"Procedural {preset['kind']} world '{world_id}', seed {base_seed}. Generated by x30_datasets.procgen.")
    sdf_path = out / "world.sdf"
    sdf_path.write_text(world_xml(spec), encoding="utf-8")

    geo, missing = parse_world(sdf_path)
    grid = rasterize(geo.geoms("collision"), bounds, MAP_RES, ZBAND)
    spawn = (round(plan.spawn[0], 2), round(plan.spawn[1], 2), round(plan.spawn[2], 3))
    sy, sx = grid.world_to_cell(spawn[0], spawn[1])
    infl = grid.inflated(ROBOT_RADIUS)
    comp = gm.component_of(infl, (sy, sx))
    reach = float(comp.sum() * MAP_RES ** 2)
    # every door approach must be inside the spawn component of the final map
    bad = []
    for d in plan.doors:
        for off in (-0.9, 0.9):
            px, py = (d.center, d.line + off) if d.orient == "h" else (d.line + off, d.center)
            if any(r.contains(px, py) for r in plan.rooms):
                cy, cx = grid.world_to_cell(px, py)
                if not comp[cy, cx]:
                    bad.append(d.id)
    if bad:
        raise RuntimeError(f"{world_id}: door approaches not reachable in final map: {sorted(set(bad))}")
    grid.save(out / "map.yaml", {"robot_radius_m": ROBOT_RADIUS, "zband_m": list(ZBAND)})

    rooms = {
        "world": world_id,
        "wall_height": wh,
        "rooms": [
            {"id": r.id, "type": r.type, "name": r.name, "rect": [round(v, 3) for v in r.rect],
             "center": [round(c, 3) for c in r.center], "area": round(r.area, 2), "walled": r.walled,
             "doors": [d.id for d in plan.doors if r.id in d.rooms]}
            for r in plan.rooms
        ],
        "doors": [
            {"id": f"door_{i + 1:02d}", "x": round(d.xy[0], 3), "y": round(d.xy[1], 3), "width": round(d.width, 3),
             "orient": d.orient, "connects": list(d.rooms)}
            for i, d in enumerate(plan.doors)
        ],
    }
    (out / "objects.json").write_text(json.dumps(objects, indent=1), encoding="utf-8")
    (out / "rooms.json").write_text(json.dumps(rooms, indent=1), encoding="utf-8")

    n_by_label: dict = {}
    for o in objects:
        n_by_label[o["label"]] = n_by_label.get(o["label"], 0) + 1
    entry = {
        "sdf": f"worlds/{world_id}/world.sdf",
        "spawn": {"x": spawn[0], "y": spawn[1], "z": 0.0, "yaw": spawn[2]},
        "bounds": list(bounds),
        "objects_file": f"worlds/{world_id}/objects.json",
        "rooms_file": f"worlds/{world_id}/rooms.json",
        "map_file": f"worlds/{world_id}/map.yaml",
        "license": "generated in this repository from SDF primitives (no third-party assets); CC0-1.0",
        "source": f"x30_datasets.procgen preset '{world_id}' seed {base_seed}",
        "source_dataset": "procedural",
        "rtf_note": "not measured yet",
        "stats": {
            "kind": preset["kind"], "rooms": len([r for r in plan.rooms if r.walled and r.type != "warehouse_hall"]),
            "objects": len(objects), "reachable_area_m2": round(reach, 1),
            "models": len(models), "objects_by_label": dict(sorted(n_by_label.items())),
            "robot_radius_m": ROBOT_RADIUS, "min_door_width_m": min([d.width for d in plan.doors if d.height < 3]),
        },
    }
    meta = dict(entry)
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return BuildResult(world_id, entry, objects, rooms)
