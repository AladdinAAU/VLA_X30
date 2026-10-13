"""Render budget for scanned scenes: triangle decimation with texture re-baking, texture caps, ceiling split.

Software rendering (llvmpipe) makes the real-time factor depend strongly on mesh size (docs/reports/sim.md), so a
scan with millions of triangles has to be reduced to roughly 100-300 k triangles. Quadric decimation destroys the
vertex colours / UVs of a scan, so the colour is re-baked:

1. the colour source is a point cloud (vertex colours of the original, or surface samples of a textured original);
2. the geometry is decimated (``fast_simplification``), unwrapped (``xatlas``) into ONE atlas of at most
   ``atlas`` x ``atlas`` pixels;
3. every atlas texel is mapped to its 3D position on the decimated surface and takes the colour of the nearest
   source point.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


def _colors_of(g: trimesh.Trimesh, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(points, rgb uint8) colour samples of one geometry: its vertices for vertex colours, surface samples if textured."""
    vis = g.visual
    if vis.kind == "texture":
        mat = vis.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        fac = getattr(mat, "baseColorFactor", None)
        n = int(max(1000, min(n_samples, g.area * 4e5)))
        pts, fidx = trimesh.sample.sample_surface(g, n, seed=rng)[:2]
        if img is None or getattr(vis, "uv", None) is None:
            c = np.asarray(fac if fac is not None else [180, 180, 180])[:3]
            return pts, np.tile(c.astype(np.uint8), (len(pts), 1))
        bary = trimesh.triangles.points_to_barycentric(np.asarray(g.vertices)[np.asarray(g.faces)[fidx]], pts)
        uv = (np.asarray(vis.uv)[np.asarray(g.faces)[fidx]] * bary[:, :, None]).sum(axis=1)
        im = np.asarray(img.convert("RGB"))
        h, w = im.shape[:2]
        px = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
        py = np.clip((1.0 - (uv[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(int)
        return pts, im[py, px]
    vc = np.asarray(vis.vertex_colors)[:, :3] if vis.kind == "vertex" else np.full((len(g.vertices), 3), 180, np.uint8)
    pts = np.asarray(g.vertices)
    # add face centroids: dense scans are fine with vertices only, coarse meshes need the extra samples
    tri = np.asarray(g.faces)
    cen = pts[tri].mean(axis=1)
    cen_col = vc[tri].astype(np.float32).mean(axis=1).astype(np.uint8)
    return np.concatenate([pts, cen]), np.concatenate([vc, cen_col])


def color_cloud(geoms: list[trimesh.Trimesh], max_points: int = 20_000_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    P, C = [], []
    for g in geoms:
        p, c = _colors_of(g, max_points // max(1, len(geoms)), rng)
        P.append(p)
        C.append(c)
    P, C = np.concatenate(P), np.concatenate(C)
    if len(P) > max_points:
        keep = rng.choice(len(P), max_points, replace=False)
        P, C = P[keep], C[keep]
    return P.astype(np.float32), C


def _rasterize_uv(uv: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    """Per-texel triangle index + 1 (0 = empty), flipped so row 0 is v = 1 (image convention)."""
    img = Image.new("I", (size, size), 0)
    d = ImageDraw.Draw(img)
    px = uv * size
    for i, f in enumerate(faces):
        d.polygon([(float(px[k, 0]), float(px[k, 1])) for k in f], fill=i + 1)
    return np.asarray(img, dtype=np.int64)


def bake_decimated(geoms: list[trimesh.Trimesh], target_tris: int, atlas: int = 2048, seed: int = 0) -> trimesh.Trimesh:
    """One textured mesh of about ``target_tris`` triangles approximating ``geoms`` (all in the same frame)."""
    import fast_simplification
    import xatlas

    t0 = time.time()
    verts = np.concatenate([np.asarray(g.vertices) for g in geoms])
    faces, off = [], 0
    for g in geoms:
        faces.append(np.asarray(g.faces) + off)
        off += len(g.vertices)
    faces = np.concatenate(faces)
    n0 = len(faces)
    pts, cols = color_cloud(geoms, seed=seed)
    log.info("colour cloud %d points (%.1fs)", len(pts), time.time() - t0)

    if n0 > target_tris:
        red = 1.0 - target_tris / n0
        v2, f2 = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int32), target_reduction=red)
    else:
        v2, f2 = verts.astype(np.float32), faces.astype(np.int32)
    m = trimesh.Trimesh(v2, f2, process=False)
    keep = m.nondegenerate_faces()
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    v2, f2 = np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)
    log.info("decimated %d -> %d triangles (%.1fs)", n0, len(f2), time.time() - t0)

    at = xatlas.Atlas()
    at.add_mesh(v2, f2)
    co = xatlas.ChartOptions()
    co.max_iterations = 2
    po = xatlas.PackOptions()
    po.resolution = atlas
    po.padding = 2
    po.bilinear = True
    po.create_image = False
    at.generate(co, po, verbose=False)
    vmap, f3, uv = at[0]
    log.info("xatlas: %d vertices, atlas %d (%.1fs)", len(vmap), atlas, time.time() - t0)
    v3 = v2[vmap]
    # xatlas uv are in texel units when the resolution is fixed
    uv = np.asarray(uv, np.float64)
    if uv.max() > 1.5:
        uv = uv / atlas
    tri_id = _rasterize_uv(np.stack([uv[:, 0], 1.0 - uv[:, 1]], 1), f3, atlas)
    yy, xx = np.nonzero(tri_id)
    ti = tri_id[yy, xx] - 1
    cx, cy = (xx + 0.5) / atlas, 1.0 - (yy + 0.5) / atlas
    a, b, c = uv[f3[ti, 0]], uv[f3[ti, 1]], uv[f3[ti, 2]]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    w0 = ((b[:, 1] - c[:, 1]) * (cx - c[:, 0]) + (c[:, 0] - b[:, 0]) * (cy - c[:, 1])) / d
    w1 = ((c[:, 1] - a[:, 1]) * (cx - c[:, 0]) + (a[:, 0] - c[:, 0]) * (cy - c[:, 1])) / d
    w0, w1 = np.clip(w0, 0, 1), np.clip(w1, 0, 1)
    w2 = np.clip(1 - w0 - w1, 0, 1)
    pos = (v3[f3[ti, 0]] * w0[:, None] + v3[f3[ti, 1]] * w1[:, None] + v3[f3[ti, 2]] * w2[:, None]).astype(np.float32)
    tree = cKDTree(pts)
    _, nn = tree.query(pos, k=1, workers=-1)
    img = np.zeros((atlas, atlas, 3), np.uint8)
    img[yy, xx] = cols[nn]
    # pad the charts so bilinear filtering and mip-maps do not bleed black
    empty = tri_id == 0
    idx = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    img = img[idx[0], idx[1]]
    log.info("baked atlas (%.1fs)", time.time() - t0)

    mat = trimesh.visual.material.SimpleMaterial(image=Image.fromarray(img), diffuse=[255, 255, 255, 255])
    out = trimesh.Trimesh(v3, f3.astype(np.int64), process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return out


def cap_texture(img: Image.Image, tex_max: int) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if s <= tex_max:
        return img
    k = tex_max / s
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)


def split_ceiling(geoms: list[trimesh.Trimesh], z_min: float = 2.0) -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    """Separate downward-facing faces above ``z_min`` (Z-up, floor at 0) from the rest; keeps uv / colours."""
    body, ceil = [], []
    for g in geoms:
        tri = np.asarray(g.vertices)[np.asarray(g.faces)]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
        is_c = (nz < -0.7) & (tri[:, :, 2].min(axis=1) > z_min)
        for mask, dst in ((~is_c, body), (is_c, ceil)):
            if mask.any():
                s = g.copy()
                s.update_faces(mask)
                s.remove_unreferenced_vertices()
                dst.append(s)
    return body, ceil
