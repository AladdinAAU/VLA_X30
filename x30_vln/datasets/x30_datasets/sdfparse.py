"""Parse the geometry of an SDF world (models, includes, collisions) and rasterise it to a grid.

Supports box / cylinder / sphere / mesh geometry, nested models and ``<include>`` of
``model://`` URIs resolved against a list of model directories. Poses are composed with
4x4 matrices (``relative_to`` frames are not supported; none of our worlds use them).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull, QhullError

from .gridmap import GridMap
from .transforms import pose_matrix


@dataclass
class Geom:
    kind: str  # box | cylinder | sphere | mesh | plane
    T: np.ndarray  # world transform of the geometry frame
    dims: tuple = ()
    mesh: Path | None = None
    scale: tuple = (1.0, 1.0, 1.0)
    model: str = ""  # top-level model name
    role: str = "collision"  # collision | visual


@dataclass
class ModelInfo:
    name: str
    T: np.ndarray
    geoms: list = field(default_factory=list)
    uri: str = ""


@dataclass
class WorldGeometry:
    name: str
    models: dict  # top-level model name -> ModelInfo

    def geoms(self, role: str = "collision") -> list[Geom]:
        return [g for m in self.models.values() for g in m.geoms if g.role == role]


def _floats(text: str | None, n: int, default=0.0) -> list[float]:
    if not text:
        return [default] * n
    vals = [float(v) for v in text.replace(",", " ").split()]
    return (vals + [default] * n)[:n]


def parse_pose(elem: ET.Element | None) -> np.ndarray:
    if elem is None:
        return np.eye(4)
    return pose_matrix(*_floats(elem.text, 6))


def resolve_uri(uri: str, model_dirs: list[Path], base: Path) -> Path | None:
    uri = uri.strip()
    if uri.startswith("model://"):
        rel = uri[len("model://") :]
        for d in model_dirs:
            p = Path(d) / rel
            if p.exists():
                return p
        return None
    if uri.startswith("file://"):
        return Path(uri[len("file://") :])
    p = Path(uri)
    if p.is_absolute():
        return p
    return (base / uri).resolve()


def _model_file(model_dir: Path) -> Path | None:
    for cand in ("model.sdf",):
        if (model_dir / cand).exists():
            return model_dir / cand
    cfg = model_dir / "model.config"
    if cfg.exists():
        sdf = ET.parse(cfg).getroot().find("sdf")
        if sdf is not None and sdf.text and (model_dir / sdf.text.strip()).exists():
            return model_dir / sdf.text.strip()
    return None


class _Walker:
    def __init__(self, model_dirs: list[Path], base: Path):
        self.model_dirs = model_dirs
        self.base = base
        self.missing: set[str] = set()

    def model(self, elem: ET.Element, T_parent: np.ndarray, info: ModelInfo, base: Path) -> None:
        T = T_parent @ parse_pose(elem.find("pose"))
        for link in elem.findall("link"):
            T_link = T @ parse_pose(link.find("pose"))
            for role in ("collision", "visual"):
                for c in link.findall(role):
                    g = self._geom(c, T_link @ parse_pose(c.find("pose")), info.name, role, base)
                    if g:
                        info.geoms.append(g)
        for sub in elem.findall("model"):
            self.model(sub, T, info, base)
        for inc in elem.findall("include"):
            self.include(inc, T, info, base)

    def include(self, inc: ET.Element, T_parent: np.ndarray, info: ModelInfo, base: Path) -> None:
        uri = (inc.findtext("uri") or "").strip()
        path = resolve_uri(uri, self.model_dirs, base)
        sdf = _model_file(path) if path and path.is_dir() else (path if path and path.suffix == ".sdf" else None)
        if sdf is None:
            self.missing.add(uri)
            return
        T = T_parent @ parse_pose(inc.find("pose"))
        scale = inc.findtext("scale")  # not used by our worlds
        if scale:
            raise ValueError("include scale is not supported")
        root = ET.parse(sdf).getroot()
        m = root.find("model")
        if m is None:
            self.missing.add(uri)
            return
        sub = ModelInfo(info.name, T)
        # the included model's own <pose> is overridden by the include pose
        m_copy = ET.Element(m.tag, m.attrib)
        for ch in m:
            if ch.tag != "pose":
                m_copy.append(ch)
        self.model(m_copy, T, sub, sdf.parent)
        info.geoms.extend(sub.geoms)

    def _geom(self, elem: ET.Element, T: np.ndarray, model: str, role: str, base: Path) -> Geom | None:
        geo = elem.find("geometry")
        if geo is None:
            return None
        box = geo.find("box")
        if box is not None:
            return Geom("box", T, tuple(_floats(box.findtext("size"), 3, 1.0)), model=model, role=role)
        cyl = geo.find("cylinder")
        if cyl is not None:
            return Geom(
                "cylinder", T, (float(cyl.findtext("radius", "0.5")), float(cyl.findtext("length", "1"))),
                model=model, role=role,
            )
        sph = geo.find("sphere")
        if sph is not None:
            return Geom("sphere", T, (float(sph.findtext("radius", "0.5")),), model=model, role=role)
        mesh = geo.find("mesh")
        if mesh is not None:
            path = resolve_uri(mesh.findtext("uri", ""), self.model_dirs, base)
            if path is None or not path.exists():
                self.missing.add(mesh.findtext("uri", ""))
                return None
            scale = tuple(_floats(mesh.findtext("scale"), 3, 1.0))
            return Geom("mesh", T, mesh=path, scale=scale, model=model, role=role)
        if geo.find("plane") is not None:
            return Geom("plane", T, model=model, role=role)
        return None


def parse_world(sdf_path: Path, model_dirs: list[Path] | None = None) -> tuple[WorldGeometry, set[str]]:
    """Parse ``sdf_path``; returns (geometry, set of URIs that could not be resolved)."""
    sdf_path = Path(sdf_path)
    root = ET.parse(sdf_path).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"{sdf_path}: no <world>")
    dirs = [Path(d) for d in (model_dirs or [])]
    walker = _Walker(dirs, sdf_path.parent)
    models: dict[str, ModelInfo] = {}
    for i, elem in enumerate(world):
        if elem.tag == "model":
            name = elem.get("name", f"model_{i}")
            info = ModelInfo(name, parse_pose(elem.find("pose")))
            walker.model(elem, np.eye(4), info, sdf_path.parent)
            models[name] = info
        elif elem.tag == "include":
            name = elem.findtext("name") or f"include_{i}"
            info = ModelInfo(name, parse_pose(elem.find("pose")), uri=elem.findtext("uri", ""))
            walker.include(elem, np.eye(4), info, sdf_path.parent)
            models[name] = info
    return WorldGeometry(world.get("name", "world"), models), walker.missing


# -- geometry helpers ---------------------------------------------------------------------------

_MESH_CACHE: dict = {}


def load_mesh(path: Path):
    import trimesh

    key = str(path)
    if key not in _MESH_CACHE:
        m = trimesh.load(str(path), force="mesh", process=False)
        if Path(path).suffix.lower() == ".dae":  # Collada <unit meter=".."> is applied by Gazebo, not by trimesh
            head = Path(path).read_bytes()[:4000].decode("utf-8", "ignore")
            mt = re.search(r'<unit[^>]*meter="([0-9.eE+-]+)"', head)
            if mt and abs(float(mt.group(1)) - 1.0) > 1e-9:
                m = m.copy()
                m.apply_scale(float(mt.group(1)))
        _MESH_CACHE[key] = m
    return _MESH_CACHE[key]


def geom_points(g: Geom) -> np.ndarray:
    """A set of world-frame points whose hull/AABB covers the geometry (Nx3)."""
    if g.kind == "box":
        h = np.array(g.dims) / 2.0
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    elif g.kind == "cylinder":
        r, L = g.dims
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        c = np.concatenate(
            [np.c_[ring, np.full(len(a), L / 2)], np.c_[ring, np.full(len(a), -L / 2)]]
        )
    elif g.kind == "sphere":
        r = g.dims[0]
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * r
    elif g.kind == "mesh":
        m = load_mesh(g.mesh)
        c = np.asarray(m.vertices) * np.array(g.scale)
    else:
        return np.zeros((0, 3))
    return c @ g.T[:3, :3].T + g.T[:3, 3]


def model_aabb(info: ModelInfo, roles=("collision", "visual")) -> np.ndarray | None:
    """[xmin, ymin, zmin, xmax, ymax, zmax] of the model, preferring the first role that has geometry."""
    for role in roles:
        pts = [geom_points(g) for g in info.geoms if g.role == role and g.kind != "plane"]
        pts = [p for p in pts if len(p)]
        if pts:
            allp = np.concatenate(pts)
            return np.concatenate([allp.min(0), allp.max(0)])
    return None


# -- rasterisation ------------------------------------------------------------------------------


def _draw_poly(draw: ImageDraw.ImageDraw, pts_xy: np.ndarray, grid: GridMap) -> None:
    px = (pts_xy[:, 0] - grid.origin[0]) / grid.resolution - 0.5
    py = (pts_xy[:, 1] - grid.origin[1]) / grid.resolution - 0.5
    draw.polygon([(float(a), float(b)) for a, b in zip(px, py)], fill=1, outline=1)


def _hull_xy(pts: np.ndarray) -> np.ndarray:
    xy = pts[:, :2]
    try:
        return xy[ConvexHull(xy).vertices]
    except QhullError:
        lo, hi = xy.min(0), xy.max(0)
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def rasterize(
    geoms: list[Geom],
    bounds: tuple,
    resolution: float,
    zband: tuple = (0.08, 0.75),
) -> GridMap:
    """Occupancy grid of everything whose collision geometry intersects the height band ``zband``."""
    x0, y0, x1, y1 = bounds
    w = int(np.ceil((x1 - x0) / resolution))
    h = int(np.ceil((y1 - y0) / resolution))
    grid = GridMap(np.zeros((h, w), bool), resolution, (x0, y0))
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    zlo, zhi = zband
    for g in geoms:
        if g.kind == "plane":
            continue
        if g.kind == "mesh":
            _draw_mesh(draw, g, grid, zlo, zhi)
            continue
        pts = geom_points(g)
        if pts[:, 2].max() < zlo or pts[:, 2].min() > zhi:
            continue
        if g.kind in ("sphere",):
            r = g.dims[0]
            c = g.T[:3, 3]
            a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            _draw_poly(draw, np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)], grid)
        elif g.kind == "cylinder" and abs(g.T[2, 2]) > 0.95:
            r = g.dims[0]
            c = g.T[:3, 3]
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            _draw_poly(draw, np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)], grid)
        else:
            _draw_poly(draw, _hull_xy(pts), grid)
    grid.occ = np.asarray(img, dtype=bool)
    return grid


def _draw_mesh(draw: ImageDraw.ImageDraw, g: Geom, grid: GridMap, zlo: float, zhi: float) -> None:
    m = load_mesh(g.mesh)
    verts = np.asarray(m.vertices) * np.array(g.scale)
    verts = verts @ g.T[:3, :3].T + g.T[:3, 3]
    tris = verts[np.asarray(m.faces)]  # F x 3 x 3
    zmin, zmax = tris[:, :, 2].min(1), tris[:, :, 2].max(1)
    keep = (zmax >= zlo) & (zmin <= zhi)
    # skip horizontal triangles lying entirely above or below the band is already handled; additionally
    # skip near-horizontal triangles at floor level (treated as ground)
    flat = (zmax - zmin) < 1e-3
    keep &= ~(flat & (zmax < zlo + 0.02))
    for t in tris[keep]:
        _draw_poly(draw, t[:, :2], grid)
