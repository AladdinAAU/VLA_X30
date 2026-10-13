"""AWS RoboMaker worlds (Gazebo Classic, MIT-0) -> Gazebo Harmonic worlds in x30_worlds.

Clone the repos (branch ``ros1``; the default branch is an archive stub) into
``datasets/external/aws/``. The hospital world additionally needs 43 OpenRobotics Fuel models
(``scripts/fetch_fuel.py``, CC-BY-4.0) in ``datasets/external/fuel/models``.

Conversion: flat ``<include>``s with instance names and Label/Thermal plugins, our physics/systems
header, a plane ground, ``<meta>`` / material scripts removed from the copied model files, the models
used copied to ``x30_worlds/models`` (Collada ``<unit>`` is honoured by Gazebo itself), objects.json and
a 2D map computed from the collision meshes.
"""
from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .. import gridmap as gm
from .. import labels
from ..paths import external_dir, world_dir, worlds_pkg_dir
from ..procgen.furniture import TEMP
from ..sdfgen import WorldSpec, world_xml
from ..sdfparse import model_aabb, parse_world, rasterize
from ..transforms import matrix_to_rpy
from .scene2sdf import MAP_RES, ROBOT_RADIUS, ZBAND, ImportResult

WORLDS = {
    "small_house": dict(repo="aws-robomaker-small-house-world", world="small_house.world", id="aws_small_house"),
    "hospital": dict(repo="aws-robomaker-hospital-world", world="hospital.world", id="aws_hospital"),
    "bookstore": dict(repo="aws-robomaker-bookstore-world", world="bookstore.world", id="aws_bookstore"),
    "small_warehouse": dict(repo="aws-robomaker-small-warehouse-world", world="small_warehouse.world",
                            id="aws_small_warehouse"),
}

# first match on "<include name> <uri tail>" wins
LABEL_RULES: list[tuple[str, str]] = [
    (r"Bed_01|TrolleyBed|MalePatientBed|Bed\b", "bed"), (r"NightStand|BedsideTable|BedTable", "nightstand"),
    (r"Sofa", "sofa"), (r"CoffeeTable", "coffee_table"), (r"KitchenTable|BalconyTable|AdjTable|DeskB|ReadingDesk", "table"),
    (r"Desk|InfoDesk|nursesstation", "desk"), (r"Chair|Sit\b|WheelChair", "chair"),
    (r"Refrigerator", "refrigerator"), (r"KitchenCabinet|CookingBench|KitchenSink", "kitchen_counter"),
    (r"Toilet", "toilet"), (r"Shower", "bathtub"), (r"TV", "tv"), (r"Computer|Tablet", "monitor"),
    (r"Bookshelf", "bookshelf"), (r"Shelf|StorageRack", "shelf"), (r"MetalCabinet|Drawer|ShoeRack", "cabinet"),
    (r"Trash", "trash_can"), (r"Carpet", "rug"), (r"Lamp", "lamp"), (r"Bicycle", "bicycle"),
    (r"Ball_", "ball"), (r"Aircondition|AirConditioner", "air_conditioner"), (r"Bucket", "bucket"),
    (r"PalletJack", "pallet_jack"), (r"Cluttering", "clutter"), (r"Vending", "vending_machine"),
    (r"Scrubs|Patient|Visitor|ElderLady|ElderMale", "person"),
    (r"Cart|Trolley|IVStand", "cart"), (r"XRay|Anesthesia|BloodPressure|CGMClassic", "medical_device"),
    (r"Door|FoldingDoor", "door"), (r"Window", "window"),
]
STRUCTURE_RX = re.compile(r"Wall|Floor|Ceiling|Roof|Ground|Column|Portrait|Board|Curtain|Handle|Pillow|Quilt|Utensils|"
                          r"SeasoningBox|Rangehood|Logo|Horologe|Spotlight|SecurityCamera|Speaker|Holder|Dumbbell|Fitness|"
                          r"Book[A-F]|Light|sign|elevator|ramp|portal", re.I)


def classify(name: str, uri_tail: str) -> str | None:
    """Semantic label of an include, or None for structure / decoration."""
    if STRUCTURE_RX.search(uri_tail) and not re.search(r"Door|Window", uri_tail):
        return None
    text = f"{name} {uri_tail}"
    for rx, label in LABEL_RULES:
        if re.search(rx, text):
            return label
    return None


def _pose_text(elem: ET.Element | None) -> list[float]:
    if elem is None or not (elem.text or "").strip():
        return [0.0] * 6
    v = [float(x) for x in elem.text.split()]
    return (v + [0.0] * 6)[:6]


def _sanitize_model(dst: Path) -> None:
    sdf = dst / "model.sdf"
    if not sdf.exists():
        return
    tree = ET.parse(sdf)
    for parent in tree.getroot().iter():
        for tag in ("meta",):
            for ch in list(parent.findall(tag)):
                parent.remove(ch)
        for mat in parent.findall("material"):
            for sc in mat.findall("script"):
                mat.remove(sc)
    for uri in tree.getroot().iter("uri"):  # bookstore models use Classic-style file://models/<name>/...
        if uri.text and uri.text.startswith("file://models/"):
            uri.text = "model://" + uri.text[len("file://models/"):]
    tree.write(sdf, encoding="utf-8", xml_declaration=True)


def _copy_model(name: str, sources: list[Path], dest_root: Path) -> bool:
    dst = dest_root / name
    if dst.exists():
        return True
    for src_root in sources:
        src = src_root / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("thumbnails", "*.md", ".git*", "LICENSE_FUEL.json"))
            shutil.copy(src / "LICENSE_FUEL.json", dst / "LICENSE_FUEL.json") if (src / "LICENSE_FUEL.json").exists() else None
            _sanitize_model(dst)
            return True
    return False


def convert(key: str, src_root: Path | None = None, out_dir: Path | None = None, models_dir: Path | None = None) -> ImportResult:
    cfg = WORLDS[key]
    wid = cfg["id"]
    src_root = Path(src_root) if src_root else external_dir() / "aws"
    repo = src_root / cfg["repo"]
    world_file = repo / "worlds" / cfg["world"]
    if not world_file.exists():
        raise FileNotFoundError(f"{world_file} not found: clone {cfg['repo']} (branch ros1) into {src_root}")
    out = Path(out_dir) if out_dir else world_dir(wid)
    models_root = Path(models_dir) if models_dir else worlds_pkg_dir() / "models"
    out.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    sources = [repo / "models", external_dir() / "fuel" / "models"]
    world = ET.parse(world_file).getroot().find("world")

    items = []  # (orig name, uri, pose6)
    skipped = []
    for m in world.findall("model"):
        inc = m.find("include")
        if inc is None:
            skipped.append(m.get("name"))
            continue
        pose = np.array(_pose_text(m.find("pose"))) + np.array(_pose_text(inc.find("pose")))  # wrappers carry the pose
        items.append((m.get("name"), inc.findtext("uri", "").strip(), pose.tolist()))
    for inc in world.findall("include"):
        uri = inc.findtext("uri", "").strip()
        if uri in ("model://sun", "model://ground_plane"):
            continue
        items.append((inc.findtext("name") or uri.split("//")[-1], uri, _pose_text(inc.find("pose"))))

    counters: dict = {}
    includes, missing = [], set()
    for name, uri, pose in items:
        mname = uri.split("//")[-1].split("/")[0]
        if not _copy_model(mname, sources, models_root):
            missing.add(mname)
            continue
        label = classify(name, mname)
        extra = ""
        inst = name
        if label:
            counters[label] = counters.get(label, 0) + 1
            inst = f"{label}_{counters[label]:02d}"
            extra = (f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
                     f"<label>{labels.label_id(label)}</label></plugin>")
            if label in TEMP:
                extra += (f'<plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal">'
                          f"<temperature>{TEMP[label]}</temperature></plugin>")
        includes.append((inst, uri, pose, extra, label, name, mname))
    if missing:
        raise FileNotFoundError(f"models not found for {wid}: {sorted(missing)} (hospital needs scripts/fetch_fuel.py)")

    spec = WorldSpec(wid, (-1, -1, 1, 1), includes=[i[:4] for i in includes], ambient_light=0.6,
                     comment=f"Converted from {cfg['repo']} ({cfg['world']}), MIT-0, by x30_datasets.adapters.aws")
    sdf_path = out / "world.sdf"
    sdf_path.write_text(world_xml(spec), encoding="utf-8")

    geo, miss = parse_world(sdf_path, [models_root])
    if miss:
        raise RuntimeError(f"unresolved uris after conversion: {sorted(miss)[:5]}")
    boxes = {n: model_aabb(info) for n, info in geo.models.items() if n != "ground_plane"}
    allb = np.array([b for b in boxes.values() if b is not None])
    lo, hi = allb[:, :3].min(0), allb[:, 3:].max(0)
    # use robust extents: ignore far outliers (decor hanging outside the building)
    lo_x, lo_y = np.percentile(allb[:, 0], 1), np.percentile(allb[:, 1], 1)
    hi_x, hi_y = np.percentile(allb[:, 3], 99), np.percentile(allb[:, 4], 99)
    margin = 2.0
    bounds = (round(float(lo_x) - margin, 1), round(float(lo_y) - margin, 1),
              round(float(hi_x) + margin, 1), round(float(hi_y) + margin, 1))
    sdf_text = world_xml(WorldSpec(wid, bounds, includes=[i[:4] for i in includes], ambient_light=0.6,
                                   comment=spec.comment))
    sdf_path.write_text(sdf_text, encoding="utf-8")

    grid = rasterize(geo.geoms("collision"), bounds, MAP_RES, ZBAND)
    infl = grid.inflated(ROBOT_RADIUS)
    comp = gm.largest_component(infl)
    cand = np.where(comp, grid.clearance(), -1.0)
    iy, ix = np.unravel_index(int(np.argmax(cand)), cand.shape)
    sx, sy = grid.cell_to_world(int(iy), int(ix))
    nav = float(comp.sum() * MAP_RES ** 2)
    grid.save(out / "map.yaml", {"robot_radius_m": ROBOT_RADIUS, "zband_m": list(ZBAND), "navigable_area_m2": round(nav, 1)})

    objects = []
    for inst, uri, pose, extra, label, name, mname in includes:
        if not label or label in ("rug",) and boxes.get(inst) is None:
            continue
        bb = boxes.get(inst)
        if bb is None:
            continue
        size = bb[3:] - bb[:3]
        objects.append({
            "id": len(objects), "label": label, "label_id": labels.label_id(label), "instance": inst,
            "pose": {"x": round(pose[0], 3), "y": round(pose[1], 3), "z": round(pose[2], 3), "yaw": round(pose[5], 4)},
            "size": [round(float(v), 3) for v in size], "bbox": [round(float(v), 3) for v in bb],
            "size_local": [round(float(v), 3) for v in size], "room": "", "color": "",
            "temperature_c": round(TEMP[label] - 273.15, 1) if label in TEMP else None,
            "attributes": (["person-like", "warm"] if label == "person" else []),
            "parent": None, "blocks_robot": bool(label not in ("rug", "door", "window", "lamp", "clutter") and bb[2] < 0.5
                                                  and size[2] > 0.3),
            "source": f"aws:{mname}",
        })
    (out / "objects.json").write_text(json.dumps(objects, indent=1), encoding="utf-8")
    counts: dict = {}
    for o in objects:
        counts[o["label"]] = counts.get(o["label"], 0) + 1
    stats = {"includes": len(includes), "objects": len(objects), "navigable_area_m2": round(nav, 1),
             "objects_by_label": dict(sorted(counts.items())), "skipped_non_include_models": skipped,
             "models_copied": len({i[6] for i in includes})}
    entry = {
        "sdf": f"worlds/{wid}/world.sdf",
        "spawn": {"x": round(sx, 2), "y": round(sy, 2), "z": 0.0, "yaw": 0.0},
        "bounds": list(bounds), "objects_file": f"worlds/{wid}/objects.json", "map_file": f"worlds/{wid}/map.yaml",
        "license": "MIT-0 (AWS RoboMaker world and models)" + ("; OpenRobotics Fuel models CC-BY-4.0" if key == "hospital" else ""),
        "source": f"https://github.com/aws-robotics/{cfg['repo']} branch ros1, file worlds/{cfg['world']}",
        "source_dataset": "aws-robomaker", "rtf_note": "not measured yet", "stats": stats,
    }
    (out / "meta.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
    return ImportResult(wid, entry, objects, stats)
