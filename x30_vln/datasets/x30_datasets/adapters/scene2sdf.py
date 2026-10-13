"""Mesh scene (glb / gltf / obj / ply) -> Gazebo world + ground truth.

Pipeline
1. load with trimesh; detect or take the up axis; scale to metres; rotate to Z-up;
2. detect the floor (lowest large horizontal surface) or take ``floor`` and shift it to z = 0;
3. write the *visual* mesh (textured OBJ, or GLB for vertex-colour meshes) into ``models/<id>_scene``;
4. build a *simplified collision*: surface samples of everything between 0.12 m and 2 m above the
   floor, voxelised (``voxel`` metres), closed, merged into a few hundred axis-aligned boxes. The
   floor itself is the world ground plane (flat-floor assumption: stairs/ramps become solid steps);
5. rasterise the collision boxes at robot height into a 2D map and estimate the navigable area
   (floor surface present, no obstacle, inflated by the robot radius, largest connected component);
6. write ``world.sdf``, ``objects.json`` (from optional semantic instances), ``rooms.json``, ``map.yaml``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

from .. import gridmap as gm
from .. import labels
from ..budget import bake_decimated, cap_texture, split_ceiling
from ..gridmap import GridMap
from ..paths import world_dir, worlds_pkg_dir
from ..sdfgen import Prim, WorldSpec, mesh_model_xml, model_xml, object_model_xml, world_xml
from ..sdfparse import Geom, rasterize
from ..transforms import SceneTransform, pose_matrix

ROBOT_RADIUS = 0.35
MAP_RES = 0.05
ZBAND = (0.08, 0.75)
COLL_Z0, COLL_Z1 = 0.12, 2.0


@dataclass
class ImportResult:
    world_id: str
    entry: dict
    objects: list
    stats: dict


# -- loading and normalisation ---------------------------------------------------------------------


def load_scene(path: Path) -> trimesh.Scene:
    obj = trimesh.load(str(path), process=False)
    if isinstance(obj, trimesh.Scene):
        return obj
    return trimesh.Scene(obj)


def detect_up(scene: trimesh.Scene, path: Path) -> str:
    if path.suffix.lower() in (".glb", ".gltf"):
        return "y"  # glTF is Y-up by specification (Habitat scenes too)
    ext = np.ptp(scene.bounds, axis=0)
    axis = int(np.argmin(ext))
    return {1: "y", 2: "z"}.get(axis, "z")


def detect_floor(vertices: np.ndarray, faces: np.ndarray, bin_size: float = 0.05, rel: float = 0.2) -> float:
    """Z of the lowest large upward-facing surface (Z-up frame): bins the area of horizontal faces."""
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    nz = n[:, 2] / np.maximum(2 * area, 1e-12)
    horiz = nz > 0.95  # upward-facing only (a slab's underside is not a floor)
    if not horiz.any():
        horiz = np.abs(nz) > 0.95
    if not horiz.any():
        return float(vertices[:, 2].min())
    z = tri[horiz].mean(axis=1)[:, 2]
    a = area[horiz]
    lo = z.min()
    idx = np.floor((z - lo) / bin_size).astype(int)
    hist = np.bincount(idx, weights=a)
    hist = hist + np.pad(hist, 1)[:-2] + np.pad(hist, 1)[2:]  # +-1 bin smoothing
    strong = np.where(hist >= rel * hist.max())[0]
    b = int(strong[0])
    sel = (idx >= b - 1) & (idx <= b + 1)
    return float(np.average(z[sel], weights=a[sel]))


def _geometries(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    return [g for g in scene.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.faces)]


def _merged(geoms: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, off = [], [], 0
    for g in geoms:
        verts.append(np.asarray(g.vertices))
        faces.append(np.asarray(g.faces) + off)
        off += len(g.vertices)
    return np.concatenate(verts), np.concatenate(faces)


def _transformed_geoms(scene: trimesh.Scene, tf: SceneTransform) -> list[trimesh.Trimesh]:
    out = []
    for g in _geometries(scene):
        g = g.copy()
        g.vertices = tf.points(g.vertices)
        if np.linalg.det(tf.R) < 0:
            g.faces = g.faces[:, ::-1]
        out.append(g)
    return out


# -- visual mesh output ------------------------------------------------------------------------------


def write_obj(geoms: list[trimesh.Trimesh], out_dir: Path, name: str = "scene", tex_max: int = 2048,
              textures: dict | None = None) -> Path:
    """Write a multi-material OBJ with textures (PNG); material = base colour texture or factor."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mtl_lines, obj_lines = [], [f"mtllib {name}.mtl"]
    voff = 1
    textures = {} if textures is None else textures  # key -> file name, shared between several OBJ files
    for gi, g in enumerate(geoms):
        mat = getattr(g.visual, "material", None)
        mname = f"m{gi}"
        kd = (0.7, 0.7, 0.7)
        tex_img = None
        if mat is not None:
            tex_img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            col = getattr(mat, "baseColorFactor", None)
            if col is None and getattr(mat, "diffuse", None) is not None:
                col = mat.diffuse
            if col is not None:
                kd = tuple(float(c) / (255.0 if max(col[:3]) > 1.0 else 1.0) for c in col[:3])
        has_uv = tex_img is not None and getattr(g.visual, "uv", None) is not None
        if not has_uv and g.visual.kind == "vertex":
            c = np.asarray(g.visual.vertex_colors)[:, :3].mean(axis=0) / 255.0
            kd = tuple(float(v) for v in c)
        mtl_lines += [f"newmtl {mname}", f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}", "Ka 0.4 0.4 0.4", "Ks 0 0 0"]
        if has_uv:
            key = g.metadata.get("tex_key", id(tex_img))
            if key not in textures:
                img = cap_texture(tex_img, tex_max)
                img = img.convert("RGBA") if img.mode not in ("RGB", "RGBA") else img
                opaque = img.mode == "RGB" or img.getchannel("A").getextrema()[0] == 255
                fname = f"tex_{len(textures)}." + ("jpg" if opaque else "png")
                if opaque:
                    img.convert("RGB").save(out_dir / fname, quality=92)
                else:
                    img.save(out_dir / fname)
                textures[key] = fname
            mtl_lines.append(f"map_Kd {textures[key]}")
        v = np.asarray(g.vertices)
        obj_lines.append(f"o geom{gi}")
        obj_lines += [f"v {a:.5f} {b:.5f} {c:.5f}" for a, b, c in v]
        if has_uv:
            obj_lines += [f"vt {a:.5f} {b:.5f}" for a, b in np.asarray(g.visual.uv)]
        normals = g.vertex_normals if len(g.faces) else None
        has_n = normals is not None and len(normals) == len(v)
        if has_n:
            obj_lines += [f"vn {a:.4f} {b:.4f} {c:.4f}" for a, b, c in normals]
        obj_lines.append(f"usemtl {mname}")
        f = np.asarray(g.faces) + voff
        if has_uv and has_n:
            obj_lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in f]
        elif has_uv:
            obj_lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in f]
        elif has_n:
            obj_lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in f]
        else:
            obj_lines += [f"f {a} {b} {c}" for a, b, c in f]
        voff += len(v)
    (out_dir / f"{name}.mtl").write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
    path = out_dir / f"{name}.obj"
    path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")
    return path


# -- collision ---------------------------------------------------------------------------------------


def greedy_boxes(vox: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """Cover the True voxels with axis-aligned boxes (z first, then x, then y); half-open index ranges."""
    rem = vox.copy()
    boxes = []
    for i, j, k in np.argwhere(vox):
        if not rem[i, j, k]:
            continue
        k1 = k + 1
        while k1 < rem.shape[2] and rem[i, j, k1]:
            k1 += 1
        i1 = i + 1
        while i1 < rem.shape[0] and rem[i1, j, k:k1].all():
            i1 += 1
        j1 = j + 1
        while j1 < rem.shape[1] and rem[i:i1, j1, k:k1].all():
            j1 += 1
        rem[i:i1, j:j1, k:k1] = False
        boxes.append((i, j, k, i1, j1, k1))
    return boxes


def collision_boxes(
    vertices: np.ndarray, faces: np.ndarray, pitch: float = 0.1, max_boxes: int = 4000, rng_seed: int = 0
) -> tuple[list[tuple], float]:
    """Voxel-merged box collision for the obstacle layer of a Z-up, floor-at-0 mesh. Returns (boxes, pitch used).

    Each box is ``(cx, cy, cz, sx, sy, sz)`` in metres.
    """
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    nz = n[:, 2] / np.maximum(2 * area, 1e-12)
    zmin, zmax = tri[:, :, 2].min(1), tri[:, :, 2].max(1)
    keep = (zmax >= COLL_Z0) & (zmin <= COLL_Z1)
    keep &= ~((np.abs(nz) > 0.85) & (zmax < 0.35))  # floors, rugs, low steps: ground plane handles them
    if not keep.any():
        return [], pitch
    sub = trimesh.Trimesh(vertices, faces[keep], process=False)
    rng = np.random.default_rng(rng_seed)
    while True:
        n_samples = int(min(6_000_000, max(1000, sub.area / (pitch * pitch) * 3)))
        pts, _ = trimesh.sample.sample_surface(sub, n_samples, seed=rng)
        pts = np.concatenate([pts, np.asarray(sub.vertices)])
        pts = pts[(pts[:, 2] >= COLL_Z0) & (pts[:, 2] <= COLL_Z1)]
        origin = np.array([pts[:, 0].min(), pts[:, 1].min(), COLL_Z0]) - pitch
        idx = np.floor((pts - origin) / pitch).astype(int)
        shape = idx.max(0) + 2
        vox = np.zeros(shape, bool)
        vox[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        vox = ndimage.binary_closing(vox, structure=np.ones((3, 3, 3)), iterations=1)
        boxes = greedy_boxes(vox)
        if len(boxes) <= max_boxes:
            break
        pitch *= 1.4
    out = []
    for i0, j0, k0, i1, j1, k1 in boxes:
        lo = origin + np.array([i0, j0, k0]) * pitch
        hi = origin + np.array([i1, j1, k1]) * pitch
        c, s = (lo + hi) / 2, hi - lo
        out.append((float(c[0]), float(c[1]), float(c[2]), float(s[0]), float(s[1]), float(s[2])))
    return out, pitch


def floor_mask(vertices: np.ndarray, faces: np.ndarray, bounds: tuple, res: float, tol: float = 0.15) -> np.ndarray:
    """Cells covered by walkable floor surface (near-horizontal faces within ``tol`` of z = 0)."""
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1)
    nz = np.abs(n[:, 2]) / np.maximum(area, 1e-12)
    zc = tri[:, :, 2].mean(1)
    sel = (nz > 0.85) & (np.abs(zc) <= tol)
    x0, y0, x1, y1 = bounds
    w, h = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    img = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(img)
    for t in tri[sel]:
        d.polygon([((p[0] - x0) / res - 0.5, (p[1] - y0) / res - 0.5) for p in t], fill=1, outline=1)
    m = np.asarray(img, dtype=bool)
    return ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=2)


# -- objects ---------------------------------------------------------------------------------------


def objects_from_instances(instances: list[dict], tf: SceneTransform, floor_band: tuple = (-0.3, 3.0),
                           rooms: list[dict] | None = None, source: str = "") -> list[dict]:
    """Instances ``{label, bbox_min, bbox_max, [id, region, category_raw]}`` in the source frame -> objects.json records."""
    recs = []
    counters: dict = {}
    for ins in instances:
        lo, hi = np.array(ins["bbox_min"], float), np.array(ins["bbox_max"], float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = tf.points(corners)
        bmin, bmax = w.min(0), w.max(0)
        if bmax[2] < floor_band[0] or bmin[2] > floor_band[1]:
            continue
        raw = ins.get("category_raw") or ins["label"]
        label = labels.canonical(ins["label"])
        if label in ("wall", "floor", "ceiling"):
            continue  # structure is not a navigation target / ground-truth object
        counters[label] = counters.get(label, 0) + 1
        cx, cy = (bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2
        room_id = ""
        if rooms:
            for r in rooms:
                if r["rect"][0] <= cx <= r["rect"][2] and r["rect"][1] <= cy <= r["rect"][3]:
                    room_id = r["id"]
                    break
        recs.append({
            "label": label, "label_id": labels.label_id(label), "instance": f"{label}_{counters[label]:02d}",
            "pose": {"x": round(float(cx), 3), "y": round(float(cy), 3), "z": round(float(bmin[2]), 3), "yaw": 0.0},
            "size": [round(float(v), 3) for v in (bmax - bmin)],
            "bbox": [round(float(v), 3) for v in (*bmin, *bmax)],
            "size_local": [round(float(v), 3) for v in (bmax - bmin)],
            "room": room_id, "color": "", "temperature_c": None, "attributes": [], "parent": None,
            "blocks_robot": bool(labels.classes().get(label, {}).get("kind") in ("furniture", "appliance")
                                 and bmin[2] < 0.5 and label not in ("rug",)),
            "category_raw": raw, "source_id": ins.get("id"), "region": ins.get("region"), "source": source,
        })
    for i, r in enumerate(recs):
        r["id"] = i
    return recs


# -- main entry ------------------------------------------------------------------------------------


def scene2sdf(
    src: Path,
    world_id: str,
    up: str = "auto",
    scale: float = 1.0,
    floor: float | str = "auto",
    instances: list[dict] | None = None,
    instances_frame: str = "mesh",
    rooms=None,  # list of rooms.json room dicts, or callable(SceneTransform) -> list
    voxel: float = 0.1,
    max_boxes: int = 4000,
    robot_radius: float = ROBOT_RADIUS,
    license: str = "",
    source: str = "",
    source_dataset: str = "",
    source_scene_id: str = "",
    out_dir: Path | None = None,
    models_dir: Path | None = None,
    ambient_light: float = 0.8,
    scene: trimesh.Scene | None = None,
    budget_tris: int | None = None,
    atlas: int = 2048,
    tex_max: int = 2048,
    split_ceil: bool = True,
    object_models: bool = False,
    sun_diffuse: float = 0.55,
    ground_visible: bool = True,
    spawn: tuple | None = None,
) -> ImportResult:
    """``scene`` (already composed, e.g. ReplicaCAD stage + objects) replaces loading ``src``.

    ``budget_tris``: bake-decimate the visual mesh to about this many triangles (see budget.py);
    ``tex_max``: longest texture side; ``split_ceil``: write the ceiling faces as a second visual model;
    ``object_models``: also emit one labelled collision-less box model per ground-truth object, which is what the
    simulator reads for ``/gt/objects`` (docs/requests/sim-to-worlds.md).
    """
    src = Path(src)
    out = Path(out_dir) if out_dir else world_dir(world_id)
    models_root = Path(models_dir) if models_dir else worlds_pkg_dir() / "models"
    out.mkdir(parents=True, exist_ok=True)
    scene = scene if scene is not None else load_scene(src)
    up_axis = detect_up(scene, src) if up == "auto" else up
    tf0 = SceneTransform.from_up_axis(up_axis, scale, 0.0)
    geoms0 = _transformed_geoms(scene, tf0)
    v0, f0 = _merged(geoms0)
    floor_z = detect_floor(v0, f0) if floor == "auto" else float(floor)
    tf = SceneTransform(tf0.R, tf0.t + np.array([0.0, 0.0, -floor_z]))
    geoms = _transformed_geoms(scene, tf)
    verts, faces = _merged(geoms)
    bmin, bmax = verts.min(0), verts.max(0)

    # collision first: it is computed from the full-detail geometry, before any decimation
    boxes, pitch = collision_boxes(verts, faces, voxel, max_boxes)
    prims = [Prim("box", (sx, sy, sz), (cx, cy, cz), color="gray", visible=False)
             for cx, cy, cz, sx, sy, sz in boxes]

    # visual: optional render budget, ceiling split, textures capped
    n_tri_src = int(len(faces))
    vis_geoms = geoms
    if budget_tris and n_tri_src > budget_tris:
        vis_geoms = [bake_decimated(geoms, budget_tris, atlas)]
    ceil_geoms: list = []
    if split_ceil:
        vis_geoms, ceil_geoms = split_ceiling(vis_geoms)
    n_tri_vis = int(sum(len(g.faces) for g in vis_geoms + ceil_geoms))
    model_dir = models_root / f"{world_id}_scene"
    mesh_dir = model_dir / "meshes"
    if mesh_dir.exists():
        for old in mesh_dir.iterdir():
            old.unlink()
    shared_tex: dict = {}
    mesh_path = write_obj(vis_geoms, mesh_dir, "scene", tex_max, shared_tex)
    rel = lambda p: (Path("..") / ".." / "models" / model_dir.name / "meshes" / p.name).as_posix()  # noqa: E731
    models = [mesh_model_xml("scene_visual", rel(mesh_path))]
    if ceil_geoms:
        models.append(mesh_model_xml("scene_ceiling", rel(write_obj(ceil_geoms, mesh_dir, "ceiling", tex_max, shared_tex))))
    models.append(model_xml("scene_collision", prims, comment=f"voxel box collision, pitch {pitch:.2f} m, {len(boxes)} boxes"))
    margin = 1.0
    bounds = (round(float(bmin[0]) - margin, 2), round(float(bmin[1]) - margin, 2),
              round(float(bmax[0]) + margin, 2), round(float(bmax[1]) + margin, 2))

    # navigable area
    geoms_box = [Geom("box", pose_matrix(cx, cy, cz), (sx, sy, sz)) for cx, cy, cz, sx, sy, sz in boxes]
    grid = rasterize(geoms_box, bounds, MAP_RES, ZBAND)
    fl = floor_mask(verts, faces, bounds, MAP_RES)
    grid.occ = grid.occ | ~fl
    infl = grid.inflated(robot_radius)
    comp = gm.largest_component(infl)
    clear = grid.clearance()
    cand = np.where(comp, clear, -1.0)
    sy_, sx_ = np.unravel_index(int(np.argmax(cand)), cand.shape)
    spawn_xy = grid.cell_to_world(int(sy_), int(sx_))
    if spawn is not None:
        spawn_xy = (float(spawn[0]), float(spawn[1]))
    nav_area = float(comp.sum() * MAP_RES ** 2)
    grid.save(out / "map.yaml", {"robot_radius_m": robot_radius, "zband_m": list(ZBAND), "navigable_area_m2": round(nav_area, 1)})

    # ground truth
    if callable(rooms):
        rooms = rooms(tf)
    objects: list = []
    if instances:
        inst = instances
        if instances_frame == "zup":  # original MP3D frame -> habitat (y up) frame, then the common transform
            inst = []
            for i in instances:
                lo, hi = np.array(i["bbox_min"]), np.array(i["bbox_max"])
                c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                c = c @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T  # (x, y, z) -> (x, z, -y)
                inst.append({**i, "bbox_min": c.min(0).tolist(), "bbox_max": c.max(0).tolist()})
        objects = objects_from_instances(inst, tf, rooms=rooms, source=source_dataset or "scene")
    (out / "objects.json").write_text(json.dumps(objects, indent=1), encoding="utf-8")
    if rooms:
        (out / "rooms.json").write_text(json.dumps({"world": world_id, "rooms": rooms, "doors": []}, indent=1), encoding="utf-8")

    if object_models:
        models += [object_model_xml(o) for o in objects]
    spec = WorldSpec(world_id, bounds, models, ambient_light=ambient_light, sun_diffuse=sun_diffuse,
                     ground_visible=ground_visible,
                     comment=f"Imported mesh scene '{src.name}' by x30_datasets scene2sdf. {license}")
    (out / "world.sdf").write_text(world_xml(spec), encoding="utf-8")

    n_tri = n_tri_src
    stats = {
        "source_file": src.name, "up_axis": up_axis, "scale": scale, "floor_z_source": round(floor_z, 3),
        "extent_m": [round(float(v), 2) for v in (bmax - bmin)], "triangles": n_tri,
        "triangles_visual": n_tri_vis, "triangles_ceiling": int(sum(len(g.faces) for g in ceil_geoms)),
        "collision_boxes": len(boxes), "collision_pitch_m": round(pitch, 3), "navigable_area_m2": round(nav_area, 1),
        "objects": len(objects), "visual_mesh_mb": round(sum(p.stat().st_size for p in (model_dir / "meshes").iterdir()) / 1e6, 1),
    }
    entry = {
        "sdf": f"worlds/{world_id}/world.sdf",
        "spawn": {"x": round(spawn_xy[0], 2), "y": round(spawn_xy[1], 2), "z": 0.0,
                  "yaw": round(float(spawn[2]), 3) if spawn is not None and len(spawn) > 2 else 0.0},
        "bounds": list(bounds),
        "objects_file": f"worlds/{world_id}/objects.json",
        "map_file": f"worlds/{world_id}/map.yaml",
        "robot_radius_m": robot_radius,
        "license": license or "see source",
        "source": source or str(src.name),
        "source_dataset": source_dataset or "mesh-scene",
        "rtf_note": "not measured yet",
        "scene_transform": tf.to_dict(),
        "stats": stats,
    }
    if rooms:
        entry["rooms_file"] = f"worlds/{world_id}/rooms.json"
    if source_scene_id:
        entry["source_scene_id"] = source_scene_id
    (out / "meta.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
    return ImportResult(world_id, entry, objects, stats)
