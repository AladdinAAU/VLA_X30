"""Ground-truth objects of a world, read from its SDF (no Gazebo needed).

Label convention (documented in docs/requests/sim-to-worlds.md), first match wins:
  1. a child element <x30:label>chair</x30:label> of the <model> (optional <x30:instance>chair_3</x30:instance>),
  2. the class id of a gz Label plugin  <plugin filename="gz-sim-label-system"><label>10</label></plugin>  mapped through
     x30_worlds/config/labels.yaml,
  3. the model name (or the include URI) with a trailing instance suffix removed (chair_3 -> chair) when it is a known class.
Models without a label are not objects (walls and the ground plane are structure, listed only in the bounds).
Temperature: <plugin filename="gz-sim-thermal-system"><temperature>K</temperature></plugin> inside the model or its visuals/links
(Kelvin, as Gazebo expects) -> attribute "warm" (>= 305 K) or "hot" (>= 330 K).
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .geometry import pose_matrix, yaw_of
from .worlds import WorldSpec, parse_sdf_text, x30_child

STRUCTURE = {'wall', 'floor', 'ceiling', 'ground_plane', 'ground'}
WARM_K, HOT_K = 305.0, 330.0


@dataclass
class SceneObject:
    name: str
    label: str
    position: tuple
    yaw: float
    size: tuple
    attributes: list = field(default_factory=list)
    temperature_k: float | None = None


def _floats(text: str | None, n: int) -> list[float]:
    vals = [float(v) for v in (text or '').replace(',', ' ').split()]
    return (vals + [0.0] * n)[:n]


def _pose(el: ET.Element | None) -> np.ndarray:
    if el is None or not el.text:
        return np.eye(4)
    return pose_matrix(*_floats(el.text, 6))


def load_label_table(path: str | Path | None) -> tuple[dict[int, str], set[str]]:
    """(id -> class, set of all class names and synonyms) from labels.yaml."""
    if not path or not Path(path).is_file():
        return {}, set()
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    ids, names = {}, set()
    for cls, info in (data.get('classes') or {}).items():
        ids[int(info['id'])] = cls
        names.add(cls)
        names.update(s.replace(' ', '_') for s in info.get('synonyms', []))
    return ids, names


def _geometry_corners(geom: ET.Element) -> np.ndarray | None:
    """Corner points of the local axis-aligned bounding box of a <geometry>, (8,3) or None when unknown."""
    box = geom.find('box')
    if box is not None:
        sx, sy, sz = _floats(box.findtext('size'), 3)
    elif geom.find('cylinder') is not None:
        cyl = geom.find('cylinder')
        r, length = float(cyl.findtext('radius') or 0), float(cyl.findtext('length') or 0)
        sx, sy, sz = 2 * r, 2 * r, length
    elif geom.find('sphere') is not None:
        d = 2 * float(geom.find('sphere').findtext('radius') or 0)
        sx = sy = sz = d
    else:
        return None
    h = np.array([sx, sy, sz]) / 2
    return np.array([[sgn_x * h[0], sgn_y * h[1], sgn_z * h[2]] for sgn_x in (-1, 1) for sgn_y in (-1, 1) for sgn_z in (-1, 1)])


def _model_aabb(model: ET.Element, T_model: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    pts = []
    for tag in ('collision', 'visual'):
        for link in model.iter('link'):
            T_link = T_model @ _pose(link.find('pose'))
            for item in link.findall(tag):
                geom = item.find('geometry')
                if geom is None:
                    continue
                corners = _geometry_corners(geom)
                if corners is None:
                    continue
                T = T_link @ _pose(item.find('pose'))
                pts.append(corners @ T[:3, :3].T + T[:3, 3])
        if pts:
            break
    if not pts:
        return None
    allp = np.concatenate(pts)
    return allp.min(axis=0), allp.max(axis=0)


def _find_model_sdf(uri: str, model_dirs: list[Path]) -> ET.Element | None:
    if not uri.startswith('model://'):
        return None
    rel = uri[len('model://'):]
    for d in model_dirs:
        f = Path(d) / rel / 'model.sdf'
        if f.is_file():
            return parse_sdf_text(f.read_text(encoding='utf-8')).find('model')
    return None


def _thermal_k(model: ET.Element) -> float | None:
    for plugin in model.iter('plugin'):
        if 'thermal' in (plugin.get('filename', '') + plugin.get('name', '')).lower():
            t = plugin.findtext('temperature')
            if t:
                return float(t)
    return None


def _label_from(model: ET.Element, name: str, uri: str, ids: dict[int, str], known: set[str]) -> tuple[str | None, str]:
    inst_el, label_el = x30_child(model, 'instance'), x30_child(model, 'label')
    instance = ((inst_el.text if inst_el is not None and inst_el.text else name) or name).strip()
    if label_el is not None and label_el.text:
        return label_el.text.strip(), instance
    for plugin in model.iter('plugin'):
        if 'label' in plugin.get('filename', '').lower():
            lab = plugin.findtext('label')
            if lab and lab.strip().lstrip('-').isdigit() and int(lab) in ids:
                return ids[int(lab)], instance
    for cand in (name, uri.rsplit('/', 1)[-1] if uri else ''):
        base = re.sub(r'[_\-]?\d+$', '', cand.strip()).lower().replace(' ', '_')
        if base and (base in known or base in ids.values()):
            return base, instance
    return None, instance


def parse_scene(spec: WorldSpec, labels_path: str | Path | None) -> tuple[list[SceneObject], dict]:
    """(objects, scene_info dict) of a world SDF."""
    ids, known = load_label_table(labels_path)
    root = parse_sdf_text(spec.sdf.read_text(encoding='utf-8'))
    world = root.find('world')
    objects: list[SceneObject] = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    n_models = 0
    items = [(m, m.get('name', ''), '') for m in world.findall('model')]
    for inc in world.findall('include'):
        uri = (inc.findtext('uri') or '').strip()
        sub = _find_model_sdf(uri, spec.model_dirs)
        if sub is not None:
            name = (inc.findtext('name') or sub.get('name', '')).strip()
            sub.set('name', name)
            pose_el = inc.find('pose')
            if pose_el is not None:
                sub.insert(0, pose_el)
            for tag in ('label', 'instance'):          # an include may carry its own label
                if x30_child(inc, tag) is not None:
                    sub.append(x30_child(inc, tag))
            items.append((sub, name, uri))
    for model, name, uri in items:
        n_models += 1
        T = _pose(model.find('pose'))
        label, instance = _label_from(model, name, uri, ids, known)
        aabb = _model_aabb(model, T)
        if label is None or label in STRUCTURE:
            if label != 'ground_plane' and 'ground' not in name.lower() and aabb is not None:
                lo, hi = np.minimum(lo, aabb[0]), np.maximum(hi, aabb[1])
            continue
        size = tuple(float(v) for v in (aabb[1] - aabb[0])) if aabb else (0.5, 0.5, 0.5)
        centre = (aabb[0] + aabb[1]) / 2 if aabb else T[:3, 3]
        temp = _thermal_k(model)
        attrs = []
        if temp is not None:
            attrs = ['hot', 'warm'] if temp >= HOT_K else ['warm'] if temp >= WARM_K else []
        objects.append(SceneObject(instance, label, tuple(float(v) for v in centre), yaw_of(T[:3, :3]), size, attrs, temp))
        if aabb is not None:
            lo, hi = np.minimum(lo, aabb[0]), np.maximum(hi, aabb[1])
    bounds = None if not np.all(np.isfinite(lo)) else {'min': [float(v) for v in lo], 'max': [float(v) for v in hi]}
    info = {
        'scene': spec.world_id,
        'world_name': world.get('name', ''),
        'bounds': bounds,
        'spawn': spec.spawn,
        'n_models': n_models,
        'landmarks': [{'name': o.name, 'label': o.label, 'x': round(o.position[0], 3), 'y': round(o.position[1], 3),
                       'z': round(o.position[2], 3), 'attributes': o.attributes} for o in objects],
    }
    return objects, info


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)
