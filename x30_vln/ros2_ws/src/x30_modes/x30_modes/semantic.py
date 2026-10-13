"""Semantic-map object records and candidate selection (pure Python, ROS messages are read duck-typed)."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class ObjectRecord:
    id: int
    label: str
    confidence: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sz: float = 0.0
    bearing_only: bool = False
    caption: str = ""
    attributes: list[str] = field(default_factory=list)
    room: str = ""
    last_seen: float = 0.0

    def to_dict(self, robot_xy: tuple[float, float] | None = None, robot_yaw: float | None = None) -> dict:
        d = {"id": self.id, "label": self.label, "confidence": round(self.confidence, 2), "x": round(self.x, 2),
             "y": round(self.y, 2), "room": self.room, "caption": self.caption, "bearing_only": self.bearing_only}
        if robot_xy is not None:
            d["distance_m"] = round(math.hypot(self.x - robot_xy[0], self.y - robot_xy[1]), 2)
            if robot_yaw is not None:
                d["bearing_deg"] = round(math.degrees(_wrap(math.atan2(self.y - robot_xy[1], self.x - robot_xy[0]) - robot_yaw)), 1)
        return d


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def record_from_msg(m: Any) -> ObjectRecord:
    """x30_msgs/SemanticObject (or any object with the same attributes)."""
    last = 0.0
    ls = getattr(m, "last_seen", None)
    if ls is not None:
        last = float(getattr(ls, "sec", 0)) + 1e-9 * float(getattr(ls, "nanosec", 0))
    return ObjectRecord(
        id=int(m.id), label=str(m.label), confidence=float(m.confidence), x=float(m.position.x), y=float(m.position.y),
        z=float(m.position.z), sx=float(m.size.x), sy=float(m.size.y), sz=float(m.size.z), bearing_only=bool(m.bearing_only),
        caption=str(m.caption), attributes=list(m.attributes), room=str(m.room), last_seen=last)


def _xyz(v: Any) -> tuple[float, float, float]:
    if isinstance(v, dict):
        return float(v.get("x", 0.0)), float(v.get("y", 0.0)), float(v.get("z", 0.0))
    v = list(v) + [0.0, 0.0, 0.0]
    return float(v[0]), float(v[1]), float(v[2])


def record_from_dict(d: dict) -> ObjectRecord:
    """Tolerant parser for the /semantic_map/query response (position as [x, y, z] or {x, y, z}; size likewise)."""
    px, py, pz = _xyz(d.get("position", d.get("centroid", [d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0)])))
    sx, sy, sz = _xyz(d.get("size", d.get("extent", [0, 0, 0])))
    return ObjectRecord(
        id=int(d["id"]), label=str(d.get("label", "")), confidence=float(d.get("confidence", 1.0)), x=px, y=py, z=pz,
        sx=sx, sy=sy, sz=sz, bearing_only=bool(d.get("bearing_only", False)), caption=str(d.get("caption", "")),
        attributes=list(d.get("attributes", []) or []), room=str(d.get("room", "")), last_seen=float(d.get("last_seen", 0.0) or 0.0))


def parse_query_response(text: str) -> list[ObjectRecord]:
    """JSON from /semantic_map/query -> records. Accepts {"objects": [...]}, {"results": [...]} or a bare list."""
    data = json.loads(text)
    items = data if isinstance(data, list) else (data.get("objects") or data.get("results") or data.get("matches") or [])
    return [record_from_dict(i) for i in items if isinstance(i, dict) and "id" in i]


def _norm(s: str) -> list[str]:
    toks = re.split(r"[\s_\-]+", s.strip().lower())
    out = []
    for t in toks:
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        if t:
            out.append(t)
    return out


def label_matches(query: str, label: str) -> bool:
    """Case/plural-insensitive match; 'extinguisher' matches 'fire extinguisher' and vice versa."""
    q, l = _norm(query), _norm(label)
    if not q or not l:
        return False
    return q == l or set(q) <= set(l) or set(l) <= set(q)


def select_candidates(objs: Iterable[ObjectRecord], label: str | None = None, exclude_ids: Iterable[int] = (),
                      min_confidence: float = 0.0, include_bearing_only: bool = False,
                      robot_xy: tuple[float, float] | None = None, attributes: Iterable[str] = ()) -> list[ObjectRecord]:
    """Filter by label/confidence/rejections and sort best first (attribute hits, then proximity and confidence)."""
    ex = set(int(i) for i in exclude_ids)
    attrs = [a.lower() for a in attributes]
    scored = []
    for o in objs:
        if o.id in ex or o.confidence < min_confidence or (o.bearing_only and not include_bearing_only):
            continue
        if label and not label_matches(label, o.label):
            continue
        hits = sum(1 for a in attrs if any(a in (x.lower()) for x in o.attributes) or a in o.caption.lower())
        dist = math.hypot(o.x - robot_xy[0], o.y - robot_xy[1]) if robot_xy else 0.0
        scored.append((-hits, dist - 4.0 * o.confidence, o))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [s[2] for s in scored]
