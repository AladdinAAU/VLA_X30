"""Habitat episode importers: ObjectNav (``*.json.gz``) and R2R / RxR VLN-CE -> x30 episode YAML.

Coordinate handling: Habitat (y up, agent looks along -z) -> world (z up) with the per-scene
``scene_transform`` stored in ``datasets/registry.yaml`` by the scene importer (see transforms.py).
Episodes of scenes that are not imported, or whose start is on another floor than the imported one,
are skipped and counted.

Official split names are kept honest: ``val_unseen``/``val`` -> ``heldout``; ``train``/``val_seen`` -> ``dev``
(override with ``split_map``).
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np

from .. import labels
from ..episodes import language as lang
from ..episodes.io import assign_split
from ..registry import load as load_registry
from ..transforms import SceneTransform, habitat_pose_to_world

DEFAULT_SPLIT_MAP = {"train": "dev", "val_seen": "dev", "val_unseen": "heldout", "val": "heldout", "test": "heldout",
                     "minival": "dev"}


def read_json_gz(path: Path) -> dict:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def scene_key(scene_id: str) -> str:
    """'hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb' -> 'TEEsavR23oF'; 'mp3d/X/X.glb' -> 'X'."""
    name = Path(scene_id).name
    for suf in (".basis.glb", ".glb", ".ply"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def scene_worlds(registry: dict | None = None) -> dict[str, tuple[str, dict]]:
    """scene key -> (world id, registry entry) for worlds imported from a Habitat-style scene."""
    reg = registry or load_registry()
    out = {}
    for wid, e in reg["worlds"].items():
        if e.get("source_scene_id") and e.get("scene_transform"):
            out[scene_key(e["source_scene_id"])] = (wid, e)
    return out


def _objects(world_id: str, entry: dict, root: Path) -> list[dict]:
    f = root / entry["objects_file"]
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _nearest_instance(objects: list[dict], label: str, xyz) -> dict | None:
    best = None
    for o in objects:
        if o["label"] != label:
            continue
        b = o["bbox"]
        d = math.hypot(max(b[0] - xyz[0], 0, xyz[0] - b[3]), max(b[1] - xyz[1], 0, xyz[1] - b[4]))
        if best is None or d < best[0]:
            best = (d, o)
    return best[1] if best and best[0] <= 2.0 else None


def import_objectnav(path: Path, worlds_root: Path, split_name: str = "val", success_distance: float = 1.0,
                     split_map: dict | None = None, floor_tol: float = 0.6, max_episodes: int | None = None,
                     registry: dict | None = None) -> tuple[dict[str, list[dict]], dict]:
    """Returns ({world_id: [episodes]}, stats). ``path`` may be a dataset-level or per-scene ``*.json.gz``."""
    data = read_json_gz(path)
    smap = {**DEFAULT_SPLIT_MAP, **(split_map or {})}
    split = smap.get(split_name, "dev")
    worlds = scene_worlds(registry)
    goals_by_cat = data.get("goals_by_category", {})
    out: dict[str, list[dict]] = {}
    stats = {"total": 0, "imported": 0, "skipped_scene": 0, "skipped_floor": 0, "skipped_goal": 0}
    for ep in data["episodes"]:
        stats["total"] += 1
        key = scene_key(ep["scene_id"])
        if key not in worlds:
            stats["skipped_scene"] += 1
            continue
        wid, entry = worlds[key]
        tf = SceneTransform.from_dict(entry["scene_transform"])
        x, y, z, yaw = habitat_pose_to_world(ep["start_position"], ep["start_rotation"], tf)
        if abs(z) > floor_tol:
            stats["skipped_floor"] += 1
            continue
        cat = labels.canonical(ep["object_category"])
        goals = ep.get("goals") or goals_by_cat.get(f"{Path(ep['scene_id']).name}_{ep['object_category']}", [])
        objs = _objects(wid, entry, worlds_root)
        inst, pos = [], []
        for g in goals:
            gp = tf.point(g["position"])
            pos.append(gp)
            o = _nearest_instance(objs, cat, gp)
            if o:
                inst.append(o["instance"])
        if not goals:
            stats["skipped_goal"] += 1
            continue
        inst = list(dict.fromkeys(inst))
        gx, gy = (float(np.mean([p[0] for p in pos])), float(np.mean([p[1] for p in pos])))
        goal = {"type": "object", "label": cat, "radius": success_distance, "distance_to": "bbox", "any_instance": True,
                "instances": inst, "x": round(gx, 2), "y": round(gy, 2)}
        info = ep.get("info", {})
        geo = float(info.get("geodesic_distance", 0.0) or 0.0)
        eid = f"{wid}_hab_objnav_{ep['episode_id']}"
        sp = ep.get("shortest_paths")
        out.setdefault(wid, []).append({
            "id": eid, "scene": wid, "mode": "object_nav",
            "instruction": f"Find {lang.an(labels.spoken(cat))} and stop next to it.",
            "start": {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)},
            "goal": goal, "time_limit_s": int(min(600, max(120, 90 + 3.5 * geo))), "split": split,
            "difficulty": {"geodesic_m": round(geo, 2), "distractors": max(0, len(inst) - 1)},
            "tags": ["habitat_objectnav", f"category_{ep['object_category']}"],
            "source": {"dataset": "habitat_objectnav", "episode_id": str(ep["episode_id"]), "scene_id": ep["scene_id"],
                       "category": ep["object_category"], "split": split_name, "path_in_file": bool(sp)},
        })
        stats["imported"] += 1
        if max_episodes and stats["imported"] >= max_episodes:
            break
    return out, stats


def import_r2r(path: Path, split_name: str = "val_unseen", split_map: dict | None = None, floor_tol: float = 0.6,
               max_episodes: int | None = None, registry: dict | None = None, gt_paths: dict | None = None
               ) -> tuple[dict[str, list[dict]], dict]:
    """R2R / RxR VLN-CE episodes -> ``velocity_follow`` episodes (instruction following)."""
    data = read_json_gz(path)
    smap = {**DEFAULT_SPLIT_MAP, **(split_map or {})}
    split = smap.get(split_name, "dev")
    worlds = scene_worlds(registry)
    out: dict[str, list[dict]] = {}
    stats = {"total": 0, "imported": 0, "skipped_scene": 0, "skipped_floor": 0, "skipped_goal": 0}
    for ep in data["episodes"]:
        stats["total"] += 1
        key = scene_key(ep["scene_id"])
        if key not in worlds:
            stats["skipped_scene"] += 1
            continue
        wid, entry = worlds[key]
        tf = SceneTransform.from_dict(entry["scene_transform"])
        x, y, z, yaw = habitat_pose_to_world(ep["start_position"], ep["start_rotation"], tf)
        if abs(z) > floor_tol:
            stats["skipped_floor"] += 1
            continue
        ref_src = ep.get("reference_path") or (gt_paths or {}).get(str(ep["episode_id"]), {}).get("locations")
        goals = ep.get("goals") or []
        if not goals and not ref_src:
            stats["skipped_goal"] += 1
            continue
        ref = [tf.point(p) for p in ref_src] if ref_src else []
        gpos = tf.point(goals[0]["position"]) if goals else ref[-1]
        radius = float(goals[0].get("radius", 3.0)) if goals else 3.0
        text = ep["instruction"]["instruction_text"] if isinstance(ep.get("instruction"), dict) else str(ep["instruction"])
        path_len = sum(math.dist(a[:2], b[:2]) for a, b in zip(ref, ref[1:])) if ref else float(ep.get("info", {}).get("geodesic_distance", 10.0))
        pts = [(float(p[0]), float(p[1])) for p in ref]
        acts = lang.path_to_actions(lang.simplify_path(pts), yaw) if len(pts) > 1 else []
        rec = {
            "id": f"{wid}_hab_r2r_{ep['episode_id']}", "scene": wid, "mode": "velocity_follow", "instruction": text.strip(),
            "start": {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)},
            "goal": {"type": "pose", "x": round(float(gpos[0]), 2), "y": round(float(gpos[1]), 2), "radius": radius},
            "time_limit_s": int(min(900, max(180, 120 + 6 * path_len))), "split": split,
            "difficulty": {"geodesic_m": round(path_len, 2), "turns": lang.count_turns(acts)},
            "tags": ["habitat_r2r"],
            "source": {"dataset": "r2r_vlnce", "episode_id": str(ep["episode_id"]), "scene_id": ep["scene_id"],
                       "trajectory_id": ep.get("trajectory_id"), "split": split_name},
        }
        if pts:
            rec["reference_path"] = [{"x": round(p[0], 2), "y": round(p[1], 2)} for p in pts]
        if acts:
            rec["reference_actions"] = acts
        out.setdefault(wid, []).append(rec)
        stats["imported"] += 1
        if max_episodes and stats["imported"] >= max_episodes:
            break
    return out, stats


def resplit(episodes: list[dict], seed: int = 0, dev_fraction: float = 0.6, key=None) -> None:
    """Replace the official split by a deterministic hash split (default key: scene+id) in place."""
    for e in episodes:
        e["split"] = assign_split(key(e) if key else f"{e['scene']}:{e['id']}", seed, dev_fraction)
