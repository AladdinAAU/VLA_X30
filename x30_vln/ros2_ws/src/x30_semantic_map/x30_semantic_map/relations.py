"""Spatial relations computed from object geometry (no ROS imports).

Objects are axis-aligned boxes centred on the estimated position (the lifted-point centroid) with the estimated size.
Relations: near / next to, on top of / under (vertical gap and footprint overlap), left / right / in front of /
behind (egocentric, relative to a robot pose, or object-to-object as seen from a viewer pose) and between.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import RelationsCfg
from .semantic_map import MapObject, SemanticMap, box_of


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def direction_word(bearing_deg: float, front_back_deg: float = 45.0) -> str:
    """Egocentric word for a bearing (+ = left of the heading)."""
    a = abs(bearing_deg)
    if a <= front_back_deg:
        return 'in front'
    if a >= 180.0 - front_back_deg:
        return 'behind'
    return 'left' if bearing_deg > 0 else 'right'


def fine_direction(bearing_deg: float) -> str:
    """Eight-sector description used in summaries: ahead, ahead-left, left, behind-left, behind, ..."""
    a = abs(bearing_deg)
    if a < 22.5:
        return 'ahead'
    if a > 157.5:
        return 'behind'
    side = 'left' if bearing_deg > 0 else 'right'
    if a < 67.5:
        return f'ahead-{side}'
    if a > 112.5:
        return f'behind-{side}'
    return side


def egocentric(pose: tuple[float, float, float], point: np.ndarray, cfg: RelationsCfg) -> dict[str, Any]:
    dx, dy = float(point[0]) - pose[0], float(point[1]) - pose[1]
    bearing = math.degrees(_wrap(math.atan2(dy, dx) - pose[2]))
    return {'range_m': round(math.hypot(dx, dy), 2), 'bearing_deg': round(bearing, 1),
            'direction': direction_word(bearing, cfg.front_back_deg), 'sector': fine_direction(bearing)}


def _footprint_overlap(a_lo, a_hi, b_lo, b_hi) -> float:
    """Overlap area of the XY footprints divided by the smaller footprint area."""
    ov = np.maximum(0.0, np.minimum(a_hi[:2], b_hi[:2]) - np.maximum(a_lo[:2], b_lo[:2]))
    inter = float(ov[0] * ov[1])
    small = min(float(np.prod(a_hi[:2] - a_lo[:2])), float(np.prod(b_hi[:2] - b_lo[:2])))
    return inter / small if small > 1e-9 else 0.0


def box_gap(a: MapObject, b: MapObject) -> float:
    """Euclidean gap between the two boxes (0 when they touch or overlap)."""
    a_lo, a_hi = box_of(a.position, a.size)
    b_lo, b_hi = box_of(b.position, b.size)
    gap = np.maximum(0.0, np.maximum(a_lo - b_hi, b_lo - a_hi))
    return float(np.linalg.norm(gap))


def pair_relations(a: MapObject, b: MapObject, cfg: RelationsCfg) -> list[str]:
    """How `a` relates to `b`: subset of ['on', 'under', 'near', 'next to']."""
    out: list[str] = []
    a_lo, a_hi = box_of(a.position, a.size)
    b_lo, b_hi = box_of(b.position, b.size)
    overlap = _footprint_overlap(a_lo, a_hi, b_lo, b_hi)
    if overlap >= cfg.on_overlap:
        if -cfg.on_tol_m <= a_lo[2] - b_hi[2] <= cfg.on_tol_m and a.position[2] > b.position[2]:
            out.append('on')
        elif -cfg.on_tol_m <= b_lo[2] - a_hi[2] <= cfg.on_tol_m and a.position[2] < b.position[2]:
            out.append('under')
    gap = box_gap(a, b)
    if gap <= cfg.near_m:
        out.append('near')
        if 'on' not in out and 'under' not in out and gap <= 0.5 * cfg.near_m:
            out.append('next to')
    return out


def viewer_relation(a_pos: np.ndarray, b_pos: np.ndarray, viewer: tuple[float, float, float], cfg: RelationsCfg) -> str:
    """Where `a` is relative to `b` as seen from the viewer: 'left of', 'right of', 'in front of' (closer to the
    viewer) or 'behind' (farther)."""
    fx, fy = b_pos[0] - viewer[0], b_pos[1] - viewer[1]
    n = math.hypot(fx, fy)
    if n < 1e-6:
        return 'near'
    fx, fy = fx / n, fy / n                        # unit vector viewer -> b
    vx, vy = a_pos[0] - b_pos[0], a_pos[1] - b_pos[1]
    along = vx * fx + vy * fy                      # + = farther from the viewer than b
    lateral = -vx * fy + vy * fx                   # + = left of the viewing direction
    ang = math.degrees(math.atan2(lateral, -along))      # 0 = in front of b (closer), 180 = behind b
    a = abs(ang)
    if a <= cfg.front_back_deg:
        return 'in front of'
    if a >= 180.0 - cfg.front_back_deg:
        return 'behind'
    return 'left of' if lateral > 0 else 'right of'


def between(a: MapObject, b: MapObject, c: MapObject, cfg: RelationsCfg) -> bool:
    """True if `a` lies between `b` and `c` (projection inside the segment, close to it in the floor plane)."""
    p, q, r = a.position[:2], b.position[:2], c.position[:2]
    seg = r - q
    length = float(np.linalg.norm(seg))
    if length < 0.3:
        return False
    t = float((p - q) @ seg) / (length * length)
    if not (0.1 <= t <= 0.9):
        return False
    perp = float(np.linalg.norm(p - (q + t * seg)))
    return perp <= cfg.between_tol_m + 0.5 * float(np.max(a.size[:2]))


def object_relations(smap: SemanticMap, oid: int, pose: tuple[float, float, float] | None = None,
                     radius: float | None = None, limit: int = 12) -> dict[str, Any]:
    cfg = smap.cfg.relations
    with smap.lock:
        o = smap.objects.get(oid)
        if o is None:
            raise KeyError(oid)
        radius = cfg.radius_m if radius is None else radius
        rows = []
        neigh: list[MapObject] = []
        for other in smap.visible_objects():
            if other.id == o.id:
                continue
            dist = float(np.linalg.norm(other.position[:2] - o.position[:2]))
            if dist > radius:
                continue
            neigh.append(other)
            rel = pair_relations(o, other, cfg)
            row: dict[str, Any] = {'id': other.id, 'label': smap.label(other), 'dist_m': round(dist, 2),
                                   'relations': rel}
            if pose is not None:
                row['seen_from_robot'] = viewer_relation(o.position, other.position, pose, cfg)
            rows.append(row)
        rows.sort(key=lambda r: (r['dist_m'], r['id']))
        rows = rows[:limit]
        bt = []
        near_objs = sorted(neigh, key=lambda x: np.linalg.norm(x.position[:2] - o.position[:2]))[:8]
        for i, b in enumerate(near_objs):
            for c in near_objs[i + 1:]:
                if between(o, b, c, cfg):
                    bt.append({'ids': [b.id, c.id], 'labels': [smap.label(b), smap.label(c)]})
        out: dict[str, Any] = {'id': o.id, 'label': smap.label(o), 'position': [round(float(x), 2) for x in o.position],
                               'room': smap.room_name_of(o), 'relations': rows, 'between': bt[:3]}
        if pose is not None:
            out['from_robot'] = egocentric(pose, o.position, cfg)
        out['text'] = relations_text(smap, o, rows, bt[:3], out.get('from_robot'))
        return out


def relations_text(smap: SemanticMap, o: MapObject, rows: list[dict[str, Any]], bt: list[dict[str, Any]],
                   ego: dict[str, Any] | None) -> str:
    parts = [f"#{o.id} {smap.label(o)}"]
    if ego:
        parts.append(f"{ego['range_m']} m {ego['sector']}")
    room = smap.room_name_of(o)
    if room:
        parts.append(f'in the {room}')
    clauses = []
    for r in rows[:6]:
        rel = [x for x in r['relations'] if x != 'near'] or (['near'] if 'near' in r['relations'] else [])
        word = ' / '.join(rel) if rel else f"{r['dist_m']} m from"
        side = f", {r['seen_from_robot']}" if r.get('seen_from_robot') else ''
        clauses.append(f"{word} #{r['id']} {r['label']} ({r['dist_m']} m{side})")
    for b in bt:
        clauses.append(f"between #{b['ids'][0]} {b['labels'][0]} and #{b['ids'][1]} {b['labels'][1]}")
    return '; '.join([', '.join(parts)] + clauses)
