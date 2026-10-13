"""Compact text description of the semantic map for the reasoner (no ROS imports)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from .config import SemanticMapConfig
from .gridtools import Frontier, GridView, find_frontiers
from .relations import fine_direction, pair_relations
from .rooms import region_label
from .semantic_map import MapObject, SemanticMap


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return int(math.ceil(len(text) / chars_per_token))


def _bearing(pose: tuple[float, float, float], p) -> tuple[float, float]:
    dx, dy = float(p[0]) - pose[0], float(p[1]) - pose[1]
    b = math.degrees((math.atan2(dy, dx) - pose[2] + math.pi) % (2 * math.pi) - math.pi)
    return math.hypot(dx, dy), b


def _angle_phrase(bearing: float) -> str:
    return f'{fine_direction(bearing)} ({round(bearing):+d} deg)'


def _object_line(smap: SemanticMap, o: MapObject, pose, neighbours: list[MapObject], cap_chars: int) -> str:
    rng, brg = _bearing(pose, o.position)
    attrs = [a for a in smap.attribute_list(o)]
    bits = [f'#{o.id} {smap.label(o)}']
    if attrs:
        bits[0] += f' [{", ".join(attrs)}]'
    where = f'{rng:.1f} m, {_angle_phrase(brg)}'
    room = smap.room_name_of(o)
    if room:
        where += f', {room}'
    if o.bearing_only:
        where += ', position approximate (seen from afar)'
    bits.append(where)
    rels = []
    for n in neighbours:
        if n.id == o.id or np.linalg.norm(n.position[:2] - o.position[:2]) > 2.0:
            continue
        r = [x for x in pair_relations(o, n, smap.cfg.relations) if x in ('on', 'under')]
        if r:
            rels.append(f'{r[0]} #{n.id} {smap.label(n)}')
        if len(rels) >= 2:
            break
    if rels:
        bits.append('; '.join(rels))
    if o.moving:
        bits.append('moving')
    if o.caption:
        c = o.caption if len(o.caption) <= cap_chars else o.caption[:cap_chars - 1].rstrip() + '...'
        bits.append(f'"{c}"')
    return ' | '.join(bits)


def build_summary(smap: SemanticMap, pose: tuple[float, float, float] | None, grid: GridView | None = None,
                  max_objects: int | None = None, token_budget: int | None = None,
                  include_tentative: bool | None = None, cfg: SemanticMapConfig | None = None) -> dict[str, Any]:
    cfg = cfg or smap.cfg
    sc = cfg.summary
    max_objects = sc.max_objects if max_objects is None else max_objects
    budget = sc.token_budget if token_budget is None else token_budget
    include_tentative = sc.include_tentative if include_tentative is None else include_tentative
    with smap.lock:
        objs = smap.visible_objects(include_tentative)
        n_total = len(objs)
        if pose is None:
            pose = (0.0, 0.0, 0.0)
            pose_known = False
        else:
            pose_known = True
        objs.sort(key=lambda o: (math.hypot(o.position[0] - pose[0], o.position[1] - pose[1]), o.id))
        header = [f'Robot at x={pose[0]:.1f}, y={pose[1]:.1f}, heading {math.degrees(pose[2]) % 360:.0f} deg (map frame; '
                  f'bearings are relative to the heading, + = left).' if pose_known else
                  'Robot pose unknown: distances and bearings are relative to the map origin.']
        counts = Counter(smap.canon(o) for o in objs)
        header.append(f'Map: {n_total} object(s)' + (': ' + ', '.join(f'{n} {k}' for k, n in counts.most_common(8))
                                                    if counts else '.'))
        room_lines = []
        in_room = None
        if smap.regions:
            rid = smap.room_id_at(np.array([pose[0], pose[1], 0.0])) if pose_known else None
            in_room = smap.regions[rid] if rid in smap.regions else None
            by_region: dict[int, list[MapObject]] = {}
            for o in objs:
                r = smap.room_id_at(o.position)
                if r is not None:
                    by_region.setdefault(r, []).append(o)
            for r in sorted(smap.regions.values(), key=lambda r: math.hypot(r.centroid[0] - pose[0], r.centroid[1] - pose[1])):
                rng, brg = _bearing(pose, r.centroid)
                members = Counter(smap.canon(o) for o in by_region.get(r.id, []))
                room_lines.append(f'- {region_label(r)} (id {r.id}, {r.area_m2:.0f} m2, at ({r.centroid[0]:.1f},{r.centroid[1]:.1f}), {rng:.0f} m '
                                  f'{fine_direction(brg)})' + (': ' + ', '.join(f'{n} {k}' if n > 1 else k
                                                                              for k, n in members.most_common(6))
                                                              if members else ': no objects yet'))
        if in_room is not None:
            header.append(f'The robot is in the {region_label(in_room)}.')
        front_lines = []
        if grid is not None:
            fr: list[Frontier] = find_frontiers(grid, (pose[0], pose[1]) if pose_known else None, cfg.frontiers,
                                                cfg.rooms.free_threshold)
            for f in fr[:cfg.frontiers.max_report]:
                rng, brg = _bearing(pose, f.center)
                note = '' if f.reachable else ', not connected to the robot free space'
                front_lines.append(f'- {rng:.1f} m {fine_direction(brg)} (opening ~{f.length_m:.1f} m{note})')
        top = objs[:max_objects]
        nb = [o for o in objs if o.dynamic is False]
        obj_lines = [_object_line(smap, o, pose, nb, sc.caption_chars) for o in top]

        def render(n_obj: int) -> str:
            parts = list(header)
            if room_lines:
                parts.append('Rooms:')
                parts += room_lines
            if smap.places:
                from .places import summary_lines
                parts.append('Places (taught by people; use the name or its owner):')
                parts += summary_lines(smap, pose if pose_known else None)
            if obj_lines:
                parts.append('Objects nearest first (id label [attributes] | distance, direction, room):')
                parts += obj_lines[:n_obj]
                if n_obj < n_total:
                    parts.append(f'(+{n_total - n_obj} more objects omitted; use the query or find tools)')
            else:
                parts.append('No objects mapped yet.')
            if grid is not None:
                parts.append('Unexplored directions:' if front_lines else 'Unexplored directions: none known.')
                parts += front_lines
            return '\n'.join(parts)

        n = len(obj_lines)
        text = render(n)
        while n > 0 and estimate_tokens(text, sc.chars_per_token) > budget:
            n = max(0, n - max(1, n // 8))
            text = render(n)
        return {'text': text, 'tokens_est': estimate_tokens(text, sc.chars_per_token), 'objects_shown': n,
                'objects_total': n_total, 'truncated': n < n_total, 'pose_known': pose_known,
                'objects': [smap.describe(o, pose if pose_known else None) for o in top[:n]]}
