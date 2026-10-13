"""ReplicaCAD (ai-habitat/ReplicaCAD_dataset, Hugging Face) -> composed mesh scene + ground-truth instances.

A ReplicaCAD scene is a stage (static architecture, ``.glb``) plus placed object instances (``.glb``) and articulated
objects (URDF with ``.glb`` links), listed in ``configs/scenes/<name>.scene_instance.json``. All coordinates are
Habitat's (Y up, quaternions ``[w, x, y, z]``). This module composes the scene in that frame; the common
:func:`scene2sdf` pipeline converts it to Z-up and builds the Gazebo world.

Facts checked against the data (not assumed):
* URDF link meshes are stored Z-up and the URDF's ``root_rotation`` joint (about 90 deg about x) turns them Y-up, so
  the placement of ``articulated_object_instances`` applies directly to the URDF root frame.
* ``uniform_scale`` of an articulated instance scales the whole chain, joint offsets included.
* Doors are revolute ``root_rotation`` joints with a negative lower limit; they are opened to ``door_open`` of that range
  so the apartment is traversable (a closed door is a wall for the robot).
"""
from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from ..transforms import rpy_to_matrix

# object-name prefix -> canonical label, overrides the (partly wrong) semantic ids of the object configs
NAME_LABELS = {
    "tv_object": "tv", "tv_screen": "tv", "tvstand": "tv_stand", "setupbox": "small_appliance", "clothes_hanger": "clutter",
    "monitor_stand": "clutter", "refrigerator": "refrigerator", "kitchen_counter": "kitchen_counter",
    "kitchencupboard": "cabinet", "kitchenCupboard": "cabinet", "chestofdrawers": "cabinet", "chestOfDrawers": "cabinet",
    "fridge": "refrigerator", "cabinet": "cabinet", "door": "door", "doubledoor": "door",
}
# ReplicaCAD lexicon names the label table has no direct class for
LEXICON_LABELS = {
    "bin": "trash_can", "indoor_plant": "plant", "wall_cabinet": "cabinet", "base_cabinet": "cabinet", "bike": "bicycle",
    "mat": "rug", "handbag": "bag", "backpack": "bag", "knife_block": "kitchen_utensil", "chopping_board": "kitchen_utensil",
    "cloth": "clothes", "clothing": "clothes", "tv_screen": "tv", "tablet": "small_appliance", "rack": "shelf",
}
SKIP_LABELS = {"clutter"}  # not reported as ground-truth objects


@dataclass
class Composed:
    scene: trimesh.Scene
    instances: list[dict] = field(default_factory=list)
    stage_name: str = ""
    counts: dict = field(default_factory=dict)


def quat_wxyz_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _tf(R: np.ndarray, t=(0.0, 0.0, 0.0), s=(1.0, 1.0, 1.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R @ np.diag(s)
    T[:3, 3] = t
    return T


def _floats(txt: str | None, n: int, default: float) -> list[float]:
    if not txt:
        return [default] * n
    return [float(v) for v in re.split(r"[\s,]+", txt.strip()) if v]


@dataclass
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def parse_urdf(path: Path) -> tuple[dict, list[_Joint]]:
    root = ET.parse(path).getroot()
    links: dict[str, list] = {}
    for link in root.findall("link"):
        vis = []
        for v in link.findall("visual"):
            o = v.find("origin")
            xyz = _floats(o.get("xyz") if o is not None else None, 3, 0.0)
            rpy = _floats(o.get("rpy") if o is not None else None, 3, 0.0)
            m = v.find("geometry/mesh")
            if m is None:
                continue
            vis.append((_tf(rpy_to_matrix(*rpy), xyz), m.get("filename"), _floats(m.get("scale"), 3, 1.0)))
        links[link.get("name")] = vis
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = _floats(o.get("xyz") if o is not None else None, 3, 0.0)
        rpy = _floats(o.get("rpy") if o is not None else None, 3, 0.0)
        ax = j.find("axis")
        lim = j.find("limit")
        joints.append(_Joint(j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                             _tf(rpy_to_matrix(*rpy), xyz), np.array(_floats(ax.get("xyz") if ax is not None else None, 3, 0.0)
                                                                     if ax is not None else [0, 0, 1]),
                             float(lim.get("lower", 0)) if lim is not None else 0.0,
                             float(lim.get("upper", 0)) if lim is not None else 0.0))
    return links, joints


def urdf_geoms(urdf: Path, door_open: float = 0.8) -> list[trimesh.Trimesh]:
    """Visual meshes of a URDF in its root frame at the zero configuration (doors opened by ``door_open`` of the range)."""
    links, joints = parse_urdf(urdf)
    child_of = {j.child: j for j in joints}
    T: dict[str, np.ndarray] = {}

    def link_tf(name: str) -> np.ndarray:
        if name in T:
            return T[name]
        j = child_of.get(name)
        if j is None:
            T[name] = np.eye(4)
        else:
            q = 0.0
            if j.kind == "revolute" and j.name == "root_rotation":
                q = door_open * (j.lower if abs(j.lower) > abs(j.upper) else j.upper)
            rot = trimesh.transformations.rotation_matrix(q, j.axis / (np.linalg.norm(j.axis) or 1.0))
            T[name] = link_tf(j.parent) @ j.origin @ rot
        return T[name]

    out = []
    for lname, vis in links.items():
        for (vt, fname, scale) in vis:
            mesh_path = (urdf.parent / fname).resolve()
            sc = trimesh.load(str(mesh_path), process=False)
            for g in (sc.dump(concatenate=False) if isinstance(sc, trimesh.Scene) else [sc]):
                if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
                    continue
                g = g.copy()
                g.metadata["tex_key"] = f"{urdf.parent.name}/{fname}/{len(out)}"
                g.apply_transform(link_tf(lname) @ vt @ np.diag([*scale, 1.0]))
                out.append(g)
    return out


def _label_for(name: str, semantic_id: int | None, lexicon: dict[int, str]) -> str:
    low = name.lower().replace("frl_apartment_", "")
    for prefix, lab in NAME_LABELS.items():
        if low.startswith(prefix.lower()):
            return lab
    lex = lexicon.get(semantic_id, "")
    from .. import labels as lb
    return LEXICON_LABELS.get(lex) or lb.canonical(lex) if lex else "other"


def compose(root: Path, scene_name: str, door_open: float = 0.8) -> Composed:
    """Stage + objects + articulated objects of ``configs/scenes/<scene_name>.scene_instance.json`` (Habitat Y-up frame)."""
    root = Path(root)
    cfg = json.loads((root / "configs" / "scenes" / f"{scene_name}.scene_instance.json").read_text(encoding="utf-8"))
    lex = {c["id"]: c["name"] for c in json.loads((root / "configs" / "ssd" / "replicaCAD_semantic_lexicon.json")
                                                  .read_text(encoding="utf-8"))["classes"]}
    scene = trimesh.Scene()
    inst: list[dict] = []
    counts: dict[str, int] = {}

    tn = cfg["stage_instance"]["template_name"]
    stage_cfg = root / "configs" / (tn + ".stage_config.json")
    if not stage_cfg.exists():
        stage_cfg = root / "configs" / "stages" / (Path(tn).name + ".stage_config.json")
    stage_glb = (stage_cfg.parent / json.loads(stage_cfg.read_text(encoding="utf-8"))["render_asset"]).resolve()
    sc = trimesh.load(str(stage_glb), process=False)
    for k, g in enumerate(sc.dump(concatenate=False)):
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            g.metadata["tex_key"] = f"stage/{stage_glb.name}/{k}"
            scene.add_geometry(g)

    cache: dict[str, list] = {}

    def add(name: str, geoms: list, T: np.ndarray, label: str, source_id: str) -> None:
        placed = []
        for g in geoms:
            g2 = g.copy()
            g2.metadata["tex_key"] = g.metadata.get("tex_key", name)
            g2.apply_transform(T)
            scene.add_geometry(g2)
            placed.append(g2)
        if label in SKIP_LABELS or not placed:
            return
        lo = np.min([p.bounds[0] for p in placed], axis=0)
        hi = np.max([p.bounds[1] for p in placed], axis=0)
        counts[label] = counts.get(label, 0) + 1
        inst.append({"id": len(inst), "label": label, "category_raw": name, "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
                     "source_id": source_id})

    for o in cfg.get("object_instances", []):
        tmpl = o["template_name"]  # objects/frl_apartment_chair_01
        oc_path = root / "configs" / (tmpl + ".object_config.json")
        if not oc_path.exists():
            continue
        oc = json.loads(oc_path.read_text(encoding="utf-8"))
        glb = (oc_path.parent / oc["render_asset"]).resolve()
        if tmpl not in cache:
            sc = trimesh.load(str(glb), process=False)
            geoms = [g for g in sc.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.faces)]
            for k, g in enumerate(geoms):
                g.metadata["tex_key"] = f"{tmpl}/{k}"
            cache[tmpl] = geoms
        s = o.get("non_uniform_scale") or [o.get("uniform_scale", 1.0)] * 3
        T = _tf(quat_wxyz_matrix(o.get("rotation", [1, 0, 0, 0])), o["translation"], s)
        base = tmpl.split("/")[-1]
        add(base, cache[tmpl], T, _label_for(base, oc.get("semantic_id"), lex), tmpl)

    for a in cfg.get("articulated_object_instances", []):
        tmpl = a["template_name"]
        urdf = next(iter(sorted((root / "urdf").glob(f"*/{tmpl}.urdf"))), None) or next(
            iter(sorted((root / "urdf").glob(f"*/{tmpl}*.urdf"))), None)
        if urdf is None:
            continue
        key = f"urdf/{tmpl}"
        if key not in cache:
            cache[key] = urdf_geoms(urdf, door_open)
        sc_ = a.get("uniform_scale", 1.0)
        T = _tf(quat_wxyz_matrix(a.get("rotation", [1, 0, 0, 0])), a["translation"], (sc_, sc_, sc_))
        add(tmpl, cache[key], T, _label_for(tmpl, None, lex), key)
    return Composed(scene, inst, stage_glb.stem, counts)
