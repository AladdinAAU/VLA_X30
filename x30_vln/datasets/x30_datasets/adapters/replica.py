"""Replica scans (facebookresearch/Replica-Dataset) -> mesh scene + ground-truth instances.

Uses ``habitat/mesh_semantic.ply`` (binary PLY: float x y z nx ny nz, uchar r g b per vertex; per quad face a uint16
``object_id``, split into two triangles here) and ``habitat/info_semantic.json`` (objects: id, class_name). The mesh is Z up in metres.
Instance boxes are the axis-aligned extents of the faces carrying the object id (measured, not the oriented boxes of the json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

# classes that are architecture / fixtures, not ground-truth objects for navigation
STRUCTURE = {"wall", "floor", "ceiling", "wall-plug", "switch", "vent", "undefined", "panel", "beam", "pipe", "stair",
             "faucet", "shower-stall", "plant-stand", "major-appliance", "small-appliance", "rack", "other-leaf"}


def read_mesh_semantic(path: Path) -> tuple[trimesh.Trimesh, np.ndarray]:
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("latin1")
    nv = int(head.split("element vertex ")[1].split()[0])
    nf = int(head.split("element face ")[1].split()[0])
    vdt = np.dtype([("p", "<f4", 3), ("n", "<f4", 3), ("c", "u1", 3)])
    v = np.frombuffer(raw, vdt, nv, end)
    fdt = np.dtype([("k", "u1"), ("i", "<u4", 4), ("o", "<u2")])  # quads in the files read so far
    f = np.frombuffer(raw, fdt, nf, end + nv * vdt.itemsize)
    assert (f["k"] == 4).all(), "unexpected face layout"
    q = np.array(f["i"], dtype=np.int64)
    tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    obj = np.concatenate([f["o"], f["o"]])
    m = trimesh.Trimesh(np.array(v["p"]), tris, process=False)
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.c_[v["c"], np.full(nv, 255, np.uint8)])
    return m, np.array(obj)


def instances(scene_dir: Path, mesh: trimesh.Trimesh, face_obj: np.ndarray) -> list[dict]:
    info = json.loads((Path(scene_dir) / "habitat" / "info_semantic.json").read_text(encoding="utf-8"))
    cls = {o["id"]: o["class_name"] for o in info["objects"]}
    order = np.argsort(face_obj, kind="stable")
    ids, start = np.unique(face_obj[order], return_index=True)
    bounds = np.append(start, len(order))
    tri = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)
    out = []
    for k, oid in enumerate(ids):
        name = cls.get(int(oid))
        if name is None or name in STRUCTURE:
            continue
        v = verts[np.unique(tri[order[bounds[k]:bounds[k + 1]]])]
        out.append({"id": int(oid), "label": name, "category_raw": name, "bbox_min": v.min(0).tolist(), "bbox_max": v.max(0).tolist()})
    return out
