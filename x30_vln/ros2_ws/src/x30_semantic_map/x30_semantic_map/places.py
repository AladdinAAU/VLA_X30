"""Taught places: named rooms and spots in the semantic map (no ROS imports).

A place is something a person told the robot during a tour ('this is Marcus's office', 'here we deliver supplies'). Two kinds:
  room  the whole room (region of the 2D map) the position lies in is named; the region keeps the name (name_source 'user') through every
        re-segmentation because `apply_places` re-applies it by position
  spot  a single point (a delivery spot, a corner) with an approach radius; used when the position is in a corridor/hall or no region is there
A place has an optional `owner` (a person the room belongs to) and `function`, aliases and the utterance it came from. Places are saved with the map.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

MAX_ROOM_AREA_M2 = 80.0                           # a bigger region is a hall / corridor system, not a room: the place becomes a spot


@dataclass
class Place:
    id: int
    name: str
    kind: str                                     # room | spot
    x: float
    y: float
    region_id: int | None = None
    area_m2: float = 0.0
    owner: str = ''
    function: str = ''
    aliases: list = field(default_factory=list)
    utterance: str = ''
    created: float = 0.0


def room_like(r) -> bool:
    return r is not None and r.name != 'corridor' and 2.0 <= r.area_m2 <= MAX_ROOM_AREA_M2


def room_region_at(smap, x: float, y: float, search_m: float = 1.5):
    """(region id, region) of the room the position belongs to. The region under the position if it is room-like; otherwise (hall, corridor, no region,
    e.g. a person standing next to a door or wall) the nearest room-like region whose cells lie within `search_m`; else the plain region or (None, None)."""
    with smap.lock:
        rid = smap.room_id_at(np.array([x, y, 0.0]))
        r = smap.regions.get(rid) if rid is not None else None
        if room_like(r) or smap.room_map is None:
            return (rid, r) if r is not None else (None, None)
        rm = smap.room_map
        g = rm.grid
        best = None
        for qid, q in smap.regions.items():
            if not room_like(q):
                continue
            rows, cols = np.nonzero(rm.labels == qid)
            if len(rows) == 0:
                continue
            dx, dy = g.origin_x + (cols + 0.5) * g.resolution - x, g.origin_y + (rows + 0.5) * g.resolution - y
            k = int(np.argmin(np.hypot(dx, dy)))
            d = float(np.hypot(dx[k], dy[k]))
            if d > search_m or (best is not None and d >= best[0]):
                continue
            los = getattr(smap, 'los_fn', None)             # a person next to a wall of the room is NOT in it: the room's cell must be visible without crossing a wall
            if los is not None and not los(np.array([x, y, 0.0]), np.array([x + dx[k], y + dy[k], 0.0])):
                continue
            best = (d, qid, q)
        if best is not None:
            return best[1], best[2]
        return (rid, r) if r is not None else (None, None)


def _key(name: str) -> str:
    return ' '.join(str(name).lower().split())


def find(smap, name_or_id) -> Place | None:
    with smap.lock:
        if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and name_or_id.isdigit()):
            return smap.places.get(int(name_or_id))
        k = _key(name_or_id)
        for p in smap.places.values():
            if _key(p.name) == k or k in {_key(a) for a in p.aliases}:
                return p
        return None


def add_place(smap, name: str, kind: str, x: float, y: float, owner: str = '', function: str = '', aliases=(), utterance: str = '') -> dict:
    """Add (or update, by name) a place at the map position (x, y). Returns {'ok', 'action': added|updated, 'place', 'note'}."""
    name = ' '.join(str(name).split())
    if not name:
        return {'ok': False, 'error': 'empty place name'}
    note = ''
    with smap.lock:
        region = None
        rid, region = room_region_at(smap, x, y) if kind == 'room' else (None, None)
        if kind == 'room':
            if region is None:
                kind, note = 'spot', 'no room region at this position: saved as a spot'
            elif region.area_m2 > MAX_ROOM_AREA_M2 or region.name == 'corridor':
                kind, note = 'spot', f'the region here is a {"corridor" if region.name == "corridor" else "large hall"} ({region.area_m2:.0f} m2), not a room: saved as a spot'
                region, rid = None, None
        old = find(smap, name)
        if old is not None:
            _release_region(smap, old)
            pid, action = old.id, 'updated'
        else:
            pid, action = smap.next_place_id, 'added'
            smap.next_place_id += 1
        p = Place(pid, name, kind, float(x), float(y), rid, float(region.area_m2) if region is not None else 0.0, str(owner or ''),
                  str(function or ''), [str(a) for a in aliases or []], str(utterance or ''), time.time())
        smap.places[pid] = p
        if region is not None:
            region.name, region.name_source = name, 'user'
    return {'ok': True, 'action': action, 'place': asdict(p), 'note': note}


def _release_region(smap, p: Place) -> None:
    """The region that carried the name of `p` loses it (name_source '' so the rules may name it again)."""
    r = smap.regions.get(p.region_id) if p.region_id is not None else None
    if r is not None and r.name_source == 'user' and _key(r.name) == _key(p.name):
        r.name, r.name_source = '', ''


def remove_place(smap, name_or_id) -> dict:
    with smap.lock:
        p = find(smap, name_or_id)
        if p is None:
            return {'ok': False, 'error': f'no place {name_or_id!r}'}
        _release_region(smap, p)
        del smap.places[p.id]
    return {'ok': True, 'removed': asdict(p)}


def apply_places(smap) -> None:
    """Re-apply the names of room places to the current regions (the segmentation may have changed ids): by position."""
    with smap.lock:
        for p in smap.places.values():
            if p.kind != 'room':
                continue
            rid, r = room_region_at(smap, p.x, p.y)
            if r is None or not room_like(r):
                continue
            p.region_id, p.area_m2 = rid, float(r.area_m2)
            r.name, r.name_source = p.name, 'user'


def list_places(smap) -> list[dict]:
    with smap.lock:
        return [asdict(p) for p in sorted(smap.places.values(), key=lambda q: q.id)]


def context_at(smap, x: float, y: float, radius: float = 4.0) -> dict:
    """What the teaching module tells the language model about the position: its region, taught places nearby, objects nearby."""
    with smap.lock:
        rid, r = room_region_at(smap, x, y)
        objs: dict[str, int] = {}
        for o in smap.visible_objects():
            if math.hypot(o.position[0] - x, o.position[1] - y) <= radius:
                lab = smap.canon(o)
                objs[lab] = objs.get(lab, 0) + 1
        near = sorted(((math.hypot(p.x - x, p.y - y), p) for p in smap.places.values()), key=lambda a: a[0])
        return {'position': [round(x, 2), round(y, 2)],
                'region': None if r is None else {'id': rid, 'name': r.name, 'name_source': r.name_source, 'area_m2': round(r.area_m2, 1),
                                                  'width_m': round(r.width_m, 1), 'length_m': round(r.length_m, 1)},
                'objects_nearby': objs, 'places': [{'name': p.name, 'kind': p.kind, 'owner': p.owner, 'distance_m': round(d, 1)} for d, p in near[:6]]}


def summary_lines(smap, pose, limit: int = 20) -> list[str]:
    """Lines for the reasoner's map summary: name, kind, owner/function, position, distance from the robot."""
    out = []
    with smap.lock:
        items = sorted(smap.places.values(), key=lambda p: math.hypot(p.x - pose[0], p.y - pose[1]) if pose else p.id)
        for p in items[:limit]:
            bits = [p.kind] + ([f'owner {p.owner}'] if p.owner else []) + ([p.function] if p.function else [])
            if p.kind == 'room' and p.area_m2:
                bits.append(f'{p.area_m2:.0f} m2')
            out.append(f'- {p.name} ({", ".join(bits)}) at ({p.x:.1f},{p.y:.1f})' + (f', {math.hypot(p.x - pose[0], p.y - pose[1]):.0f} m' if pose else ''))
    return out


def to_dicts(smap) -> list[dict]:
    return list_places(smap)


def from_dicts(smap, data: list[dict]) -> None:
    with smap.lock:
        smap.places = {int(d['id']): Place(**{k: d[k] for k in Place.__dataclass_fields__ if k in d}) for d in data or []}
        smap.next_place_id = (max(smap.places) + 1) if smap.places else 1
