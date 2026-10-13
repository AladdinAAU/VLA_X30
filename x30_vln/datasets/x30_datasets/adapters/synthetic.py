"""Synthetic mini-datasets that mimic the on-disk layout of HM3D / Matterport3D and the Habitat episode files.

Used by tests and by ``scripts/make_synthetic_habitat.py`` to prove the import pipeline without the
licensed data. The geometry is a two-room house (z-up here, exported y-up like Habitat scenes).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from ..transforms import SceneTransform, world_pose_to_habitat

# name, category, centre (x, y, z), size (sx, sy, sz), colour, region
OBJECTS = [
    ("chair_a", "chair", (1.5, 1.2, 0.45), (0.5, 0.5, 0.9), (200, 30, 30), 1),
    ("chair_b", "chair", (7.2, 3.8, 0.45), (0.5, 0.5, 0.9), (30, 30, 200), 2),
    ("table_a", "table", (2.2, 1.2, 0.375), (1.4, 0.8, 0.75), (120, 80, 40), 1),
    ("sofa_a", "sofa", (7.5, 1.0, 0.4), (2.0, 0.9, 0.8), (30, 160, 60), 2),
    ("bed_a", "bed", (8.0, 4.7, 0.3), (2.0, 1.4, 0.6), (220, 200, 40), 2),
]
WALL_COLOR = (230, 230, 220)
FLOOR_COLOR = (150, 120, 90)
HOUSE = (10.0, 6.0, 2.6)  # x, y extents, wall height; a wall at x=5 with a door gap y in [2.5, 3.5]


def _box(center, size, color) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(np.array([*color, 255], np.uint8), (len(m.faces), 1)))
    return m


def _shell() -> list[tuple[str, trimesh.Trimesh]]:
    X, Y, H = HOUSE
    t = 0.15
    parts = [
        ("wall", _box((X / 2, -t / 2, H / 2), (X + t, t, H), WALL_COLOR)),
        ("wall", _box((X / 2, Y + t / 2, H / 2), (X + t, t, H), WALL_COLOR)),
        ("wall", _box((-t / 2, Y / 2, H / 2), (t, Y, H), WALL_COLOR)),
        ("wall", _box((X + t / 2, Y / 2, H / 2), (t, Y, H), WALL_COLOR)),
        ("wall", _box((5.0, 1.25, H / 2), (t, 2.5, H), WALL_COLOR)),
        ("wall", _box((5.0, 4.75, H / 2), (t, 2.5, H), WALL_COLOR)),
    ]
    return parts


def _to_yup(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """z-up -> habitat y-up: (x, y, z) -> (x, z, -y)."""
    R = np.eye(4)
    R[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    m = m.copy()
    m.apply_transform(R)
    return m


def build_scene() -> tuple[trimesh.Scene, trimesh.Trimesh]:
    """(visual scene with a textured floor, semantic mesh with one colour per instance), both y-up."""
    X, Y, _ = HOUSE
    scene = trimesh.Scene()
    floor = trimesh.creation.box(extents=(X, Y, 0.1))
    floor.apply_translation((X / 2, Y / 2, -0.05))
    uv = np.c_[(floor.vertices[:, 0] / X), (floor.vertices[:, 1] / Y)]
    img = Image.fromarray((np.random.default_rng(0).integers(100, 200, (32, 32, 3))).astype(np.uint8))
    floor.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.SimpleMaterial(image=img))
    scene.add_geometry(_to_yup(floor), geom_name="floor")
    for i, (_, m) in enumerate(_shell()):
        scene.add_geometry(_to_yup(m), geom_name=f"wall{i}")
    for name, _, c, s, col, _ in OBJECTS:
        scene.add_geometry(_to_yup(_box(c, s, col)), geom_name=name)
    sem_parts = [_to_yup(_box((X / 2, Y / 2, -0.05), (X, Y, 0.1), (10, 10, 10)))]
    sem_parts += [_to_yup(m) for _, m in _shell()]
    sem_parts += [_to_yup(_box(c, s, col)) for _, _, c, s, col, _ in OBJECTS]
    return scene, trimesh.util.concatenate(sem_parts)


def semantic_txt() -> str:
    lines = ["HM3D Semantic Annotations", '1,0x0A0A0A,"floor",1', '2,0xE6E6DC,"wall",1']
    for i, (_, cat, _, _, col, region) in enumerate(OBJECTS, start=3):
        lines.append(f'{i},0x{col[0]:02X}{col[1]:02X}{col[2]:02X},"{cat}",{region}')
    return "\n".join(lines) + "\n"


def house_file() -> str:
    """MP3D ``.house`` subset: C (category), R (region), O (object) records."""
    lines = ["ASCII 1.0", "H synthetic 0 0 0 0 0 0 0 5 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"]
    cats = sorted({o[1] for o in OBJECTS})
    for i, c in enumerate(cats, start=1):
        lines.append(f"C {i} {i} {c} {i} {c} 0 0 0 0 0")
    X, Y, H = HOUSE
    lines.append(f"R 0 0 0 0 l {X / 4} {Y / 2} 1 0 0 0 5 {Y} {H} {H} 0 0 0 0")
    lines.append(f"R 1 0 0 0 b {3 * X / 4} {Y / 2} 1 5 0 0 {X} {Y} {H} {H} 0 0 0 0")
    for i, (_, cat, c, s, _, region) in enumerate(OBJECTS):
        r = [v / 2 for v in s]
        lines.append(f"O {i} {region - 1} {cats.index(cat) + 1} {c[0]} {c[1]} {c[2]} 1 0 0 0 1 0 {r[0]} {r[1]} {r[2]} 0 0 0 0 0 0 0 0")
    return "\n".join(lines) + "\n"


def write_hm3d(root: Path, scene_name: str = "TESTscene01") -> Path:
    d = Path(root) / "hm3d" / "val" / f"00800-{scene_name}"
    d.mkdir(parents=True, exist_ok=True)
    vis, sem = build_scene()
    vis.export(str(d / f"{scene_name}.basis.glb"))
    sem.export(str(d / f"{scene_name}.semantic.glb"))
    (d / f"{scene_name}.semantic.txt").write_text(semantic_txt(), encoding="utf-8")
    (d / f"{scene_name}.basis.navmesh").write_bytes(b"\x00" * 16)  # placeholder, not parsed
    return d


def write_mp3d(root: Path, scene_name: str = "TESTmp3dXYZ") -> Path:
    d = Path(root) / "mp3d" / scene_name
    d.mkdir(parents=True, exist_ok=True)
    vis, _ = build_scene()
    vis.export(str(d / f"{scene_name}.glb"))
    (d / f"{scene_name}.house").write_text(house_file(), encoding="utf-8")
    (d / f"{scene_name}.navmesh").write_bytes(b"\x00" * 16)
    return d


def _hab_start(x: float, y: float, yaw: float) -> tuple[list, list]:
    return world_pose_to_habitat(x, y, 0.0, yaw, SceneTransform.habitat(0.0))


def write_objectnav(path: Path, scene_id: str) -> None:
    eps = []
    for k, (x, y, yaw) in enumerate([(1.0, 4.5, 0.3), (8.5, 1.0, 3.0), (3.0, 3.0, -1.0)]):
        p, q = _hab_start(x, y, yaw)
        eps.append({"episode_id": str(k), "scene_id": scene_id, "start_position": p, "start_rotation": q,
                    "info": {"geodesic_distance": 6.0 + k}, "goals": [], "start_room": None,
                    "shortest_paths": None, "object_category": "chair"})
    goals = {f"{Path(scene_id).name}_chair": [
        {"object_id": 3, "object_category": "chair", "object_name": "chair_a",
         "position": [1.5, 0.45, -1.2], "view_points": [{"agent_state": {"position": [1.5, 0.0, -2.2], "rotation": [0, 0, 0, 1]}, "iou": 0.2}]},
        {"object_id": 4, "object_category": "chair", "object_name": "chair_b",
         "position": [7.2, 0.45, -3.8], "view_points": []}]}
    data = {"episodes": eps, "goals_by_category": goals,
            "category_to_task_category_id": {"chair": 0}, "category_to_scene_annotation_category_id": {"chair": 3}}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def write_r2r(path: Path, scene_id: str) -> None:
    X = SceneTransform.habitat(0.0)
    eps = []
    for k, (a, b, yaw) in enumerate([((1.0, 1.0), (8.0, 3.0), 0.0), ((9.0, 5.0), (2.0, 2.0), 3.1)]):
        p, q = _hab_start(a[0], a[1], yaw)
        mid = (5.0, 3.0)
        ref = [_to_habitat_point(X, (a[0], a[1])), _to_habitat_point(X, mid), _to_habitat_point(X, b)]
        eps.append({"episode_id": 4991 + k, "trajectory_id": 1622 + k, "scene_id": scene_id,
                    "instruction": {"instruction_text": "Walk through the doorway into the next room and stop near the end.",
                                    "instruction_tokens": [1, 2, 3]},
                    "start_position": p, "start_rotation": q, "goals": [{"position": ref[-1], "radius": 3.0}],
                    "reference_path": ref, "info": {"geodesic_distance": 9.0}})
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"episodes": eps, "instruction_vocab": {"word_list": ["walk"]}}, f)


def _to_habitat_point(tf: SceneTransform, xy: tuple) -> list:
    p = np.linalg.inv(tf.R) @ (np.array([xy[0], xy[1], 0.0]) - tf.t)
    return [float(v) for v in p]
