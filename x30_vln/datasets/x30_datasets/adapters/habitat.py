"""Readers for Habitat-style scene datasets (HM3D, Matterport3D, Replica, ScanNet) laid out on disk.

The licensed datasets are NOT downloaded by this code; the user places them under
``datasets/external/<name>/`` (see docs/WORLDS.md). Each ``find_*`` function returns scene records
``{name, mesh, up, instances_fn, ...}`` that :func:`import_scene` turns into a Gazebo world.

Verified only against synthetic mini-datasets that follow the documented layouts (see tests);
the real files may differ in details (colour encoding of ``*.semantic.glb``, MP3D ``.house`` variants).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import trimesh

from .. import labels
from .scene2sdf import ImportResult, scene2sdf

HM3D_LINE = re.compile(r'^\s*(\d+)\s*,\s*0x([0-9A-Fa-f]{6})\s*,\s*"(.*)"\s*,\s*(\d+)\s*$')
MP3D_REGION_NAMES = {
    "a": "bathroom", "b": "bedroom", "c": "closet", "d": "dining room", "e": "entryway", "f": "family room",
    "g": "garage", "h": "hallway", "i": "library", "j": "laundry room", "k": "kitchen", "l": "living room",
    "m": "meeting room", "n": "lounge", "o": "office", "p": "porch", "r": "recreation room", "s": "stairs",
    "t": "toilet", "u": "utility room", "v": "tv room", "w": "workout room", "x": "outdoor", "y": "balcony",
    "z": "other room", "B": "bar", "C": "classroom", "D": "dining booth", "S": "spa", "Z": "junk", "-": "no label",
}


# -- HM3D -----------------------------------------------------------------------------------------


def parse_hm3d_semantic_txt(path: Path) -> dict[int, dict]:
    """``<scene>.semantic.txt``: ``id,0xRRGGBB,"category",region`` per line after a header line."""
    out: dict[int, dict] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = HM3D_LINE.match(line)
        if m:
            out[int(m.group(1))] = {"id": int(m.group(1)), "color": m.group(2).upper(), "category_raw": m.group(3),
                                    "region": int(m.group(4))}
    return out


def _face_colors(g: trimesh.Trimesh) -> np.ndarray:
    """RGB (uint8) per face from vertex colours, material colour or sampled texture."""
    vis = g.visual
    if vis.kind == "texture":
        try:
            vc = np.asarray(vis.to_color().vertex_colors)[:, :3]
        except Exception:
            mat = vis.material
            col = np.asarray(getattr(mat, "baseColorFactor", [128, 128, 128, 255]))[:3]
            vc = np.tile(col.astype(np.uint8), (len(g.vertices), 1))
    else:
        vc = np.asarray(vis.vertex_colors)[:, :3]
    return vc[np.asarray(g.faces)[:, 0]]


def hm3d_instances(semantic_glb: Path, semantic_txt: Path) -> list[dict]:
    """Instance AABBs (habitat y-up frame) from the colour-coded semantic mesh and its txt table."""
    table = parse_hm3d_semantic_txt(semantic_txt)
    by_color = {v["color"]: v for v in table.values()}
    scene = trimesh.load(str(semantic_glb), process=False)
    geoms = scene.dump(concatenate=False) if isinstance(scene, trimesh.Scene) else [scene]
    acc: dict[str, list] = {}
    for g in geoms:
        if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
            continue
        fc = _face_colors(g)
        keys = (fc[:, 0].astype(int) << 16) | (fc[:, 1].astype(int) << 8) | fc[:, 2].astype(int)
        for key in np.unique(keys):
            hexc = f"{int(key):06X}"
            if hexc not in by_color:
                continue
            tri = np.asarray(g.vertices)[np.asarray(g.faces)[keys == key]].reshape(-1, 3)
            acc.setdefault(hexc, []).append((tri.min(0), tri.max(0)))
    out = []
    for hexc, boxes in acc.items():
        rec = by_color[hexc]
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        out.append({"id": rec["id"], "label": rec["category_raw"], "category_raw": rec["category_raw"],
                    "region": rec["region"], "bbox_min": lo.tolist(), "bbox_max": hi.tolist()})
    return sorted(out, key=lambda r: r["id"])


def find_hm3d_scenes(root: Path) -> list[dict]:
    scenes = []
    for glb in sorted(Path(root).rglob("*.basis.glb")):
        stem = glb.name[: -len(".basis.glb")]
        sem_glb, sem_txt = glb.with_name(f"{stem}.semantic.glb"), glb.with_name(f"{stem}.semantic.txt")
        scenes.append({
            "name": stem, "dataset": "hm3d", "mesh": glb, "up": "y",
            "semantic": (sem_glb, sem_txt) if sem_glb.exists() and sem_txt.exists() else None,
            "navmesh": glb.with_name(f"{stem}.basis.navmesh") if glb.with_name(f"{stem}.basis.navmesh").exists() else None,
            "scene_id": glb.name,
            "folder": glb.parent.name,
        })
    return scenes


# -- Matterport3D ---------------------------------------------------------------------------------


def parse_mp3d_house(path: Path) -> tuple[list[dict], list[dict]]:
    """``<id>.house`` -> (object instances as AABBs in the original z-up frame, regions)."""
    cats: dict[int, tuple] = {}
    objs, regions = [], []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "C":  # C idx map_idx map_name mpcat40_idx mpcat40_name
            cats[int(t[1])] = (t[3].replace("#", " ") if len(t) > 3 else "", t[5] if len(t) > 5 else "")
        elif t[0] == "R":  # R idx level 0 0 label px py pz xlo ylo zlo xhi yhi zhi height
            lo, hi = [float(v) for v in t[9:12]], [float(v) for v in t[12:15]]
            regions.append({"index": int(t[1]), "label": t[5], "bbox_min": lo, "bbox_max": hi})
        elif t[0] == "O":  # O idx region cat px py pz a0(3) a1(3) r0 r1 r2
            v = [float(x) for x in t[4:19]]
            c, a0, a1, r = np.array(v[0:3]), np.array(v[3:6]), np.array(v[6:9]), np.array(v[9:12])
            a2 = np.cross(a0, a1)
            half = np.abs(a0) * r[0] + np.abs(a1) * r[1] + np.abs(a2) * r[2]
            name = cats.get(int(t[3]), ("", ""))
            objs.append({"id": int(t[1]), "label": name[0] or name[1] or "object", "category_raw": name[0] or name[1],
                         "region": int(t[2]), "bbox_min": (c - half).tolist(), "bbox_max": (c + half).tolist()})
    return objs, regions


def mp3d_rooms(regions: list[dict], tf) -> list[dict]:
    """Regions (z-up frame) -> rooms.json entries in the world frame."""
    rooms = []
    for r in regions:
        lo, hi = np.array(r["bbox_min"]), np.array(r["bbox_max"])
        # z-up original frame -> habitat -> world, same as objects
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        corners = corners @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
        w = tf.points(corners)
        name = MP3D_REGION_NAMES.get(r["label"], "room")
        rid = f"{name.replace(' ', '_')}_{r['index']}"
        rooms.append({"id": rid, "type": name.replace(" ", "_"), "name": f"{name} {r['index']}",
                      "rect": [round(float(v), 3) for v in (w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max())],
                      "center": [round(float(w[:, 0].mean()), 3), round(float(w[:, 1].mean()), 3)],
                      "area": round(float((w[:, 0].max() - w[:, 0].min()) * (w[:, 1].max() - w[:, 1].min())), 2),
                      "walled": False, "doors": [], "z": [round(float(w[:, 2].min()), 2), round(float(w[:, 2].max()), 2)]})
    return rooms


def find_mp3d_scenes(root: Path) -> list[dict]:
    scenes = []
    for glb in sorted(Path(root).rglob("*.glb")):
        d = glb.parent
        if glb.stem != d.name:
            continue
        house = d / f"{glb.stem}.house"
        scenes.append({"name": glb.stem, "dataset": "mp3d", "mesh": glb, "up": "y",
                       "house": house if house.exists() else None, "scene_id": f"mp3d/{glb.stem}/{glb.name}",
                       "navmesh": d / f"{glb.stem}.navmesh" if (d / f"{glb.stem}.navmesh").exists() else None})
    return scenes


# -- Replica / ScanNet ----------------------------------------------------------------------------


def find_replica_scenes(root: Path) -> list[dict]:
    scenes = []
    for ply in sorted(Path(root).rglob("mesh.ply")):
        info = ply.parent / "habitat" / "info_semantic.json"
        scenes.append({"name": ply.parent.name, "dataset": "replica", "mesh": ply, "up": "auto",
                       "info": info if info.exists() else None, "scene_id": ply.parent.name})
    return scenes


def replica_instances(info_json: Path) -> list[dict]:
    """``habitat/info_semantic.json``: objects with class_id and an axis-aligned box (mesh frame)."""
    data = json.loads(Path(info_json).read_text(encoding="utf-8"))
    cls = {c["id"]: c["name"] for c in data.get("classes", [])}
    out = []
    for o in data.get("objects", []):
        abb = o.get("oriented_bbox", {}).get("abb")
        if not abb:
            continue
        c, sz = np.array(abb["center"], float), np.array(abb["size"], float)
        name = cls.get(o.get("class_id"), "object")
        out.append({"id": o["id"], "label": name, "category_raw": name, "bbox_min": (c - sz / 2).tolist(),
                    "bbox_max": (c + sz / 2).tolist()})
    return out


def find_scannet_scenes(root: Path) -> list[dict]:
    scenes = []
    for ply in sorted(Path(root).rglob("*_vh_clean_2.ply")):
        stem = ply.name[: -len("_vh_clean_2.ply")]
        agg = ply.with_name(f"{stem}.aggregation.json")
        segs = ply.with_name(f"{stem}_vh_clean_2.0.010000.segs.json")
        scenes.append({"name": stem, "dataset": "scannet", "mesh": ply, "up": "z",
                       "aggregation": (agg, segs) if agg.exists() and segs.exists() else None, "scene_id": stem})
    return scenes


def scannet_instances(aggregation: Path, segs: Path, mesh_path: Path) -> list[dict]:
    mesh = trimesh.load(str(mesh_path), process=False)
    verts = np.asarray(mesh.vertices)
    seg_idx = np.array(json.loads(Path(segs).read_text())["segIndices"])
    out = []
    for g in json.loads(Path(aggregation).read_text())["segGroups"]:
        m = np.isin(seg_idx, g["segments"])
        if m.any():
            out.append({"id": g["objectId"], "label": g["label"], "category_raw": g["label"],
                        "bbox_min": verts[m].min(0).tolist(), "bbox_max": verts[m].max(0).tolist()})
    return out


# -- generic entry --------------------------------------------------------------------------------


def find_scenes(root: Path, dataset: str = "auto") -> list[dict]:
    root = Path(root)
    finders = {"hm3d": find_hm3d_scenes, "mp3d": find_mp3d_scenes, "replica": find_replica_scenes,
               "scannet": find_scannet_scenes}
    if dataset != "auto":
        return finders[dataset](root)
    for name in ("hm3d", "replica", "scannet", "mp3d"):  # mp3d last: its pattern is the loosest
        found = finders[name](root)
        if found:
            return found
    return []


def import_scene(scene: dict, world_id: str, floor: float | str = "auto", license: str = "", **kw) -> ImportResult:
    """Import one scene record from :func:`find_scenes` as a registered-ready world."""
    instances, frame, rooms = None, "mesh", None
    ds = scene["dataset"]
    if ds == "hm3d" and scene.get("semantic"):
        instances = hm3d_instances(*scene["semantic"])
    elif ds == "mp3d" and scene.get("house"):
        instances, regions = parse_mp3d_house(scene["house"])
        frame = "zup"
    elif ds == "replica" and scene.get("info"):
        instances = replica_instances(scene["info"])
    elif ds == "scannet" and scene.get("aggregation"):
        instances = scannet_instances(*scene["aggregation"], scene["mesh"])
    if ds == "mp3d" and scene.get("house"):
        rooms = lambda tf: mp3d_rooms(regions, tf)  # noqa: E731  (needs the floor-shifted transform)
    return scene2sdf(scene["mesh"], world_id, up=scene.get("up", "auto"), floor=floor, instances=instances,
                     instances_frame=frame, rooms=rooms,
                     license=license or f"{ds}: licence terms of the dataset apply (not redistributable)",
                     source=f"{ds} scene {scene['name']}", source_dataset=ds, source_scene_id=scene["scene_id"], **kw)


def label_summary(objects: list[dict]) -> dict[str, int]:
    out: dict = {}
    for o in objects:
        out[o["label"]] = out.get(o["label"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


__all__ = ["find_scenes", "import_scene", "hm3d_instances", "parse_mp3d_house", "parse_hm3d_semantic_txt",
           "label_summary", "labels"]
