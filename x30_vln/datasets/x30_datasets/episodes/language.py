"""Template-based instructions from reference paths and ground-truth objects (several paraphrases)."""
from __future__ import annotations

import math
import random

from .. import labels
from ..transforms import wrap_angle
from .context import WorldContext

COLOR_WORDS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown", "beige", "gray"}
SKIP_LANDMARK = {"rug", "picture", "window", "mirror", "door", "clutter", "laptop", "monitor", "microwave", "book",
                 "radiator", "whiteboard", "box", "trash_can", "fire_extinguisher", "lamp"}
SKIP_REFERENCE = {"rug", "picture", "mirror", "door", "window", "clutter", "laptop", "monitor", "microwave"}


def an(phrase: str) -> str:
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


def the(label: str) -> str:
    return "the " + labels.spoken(label)


def dist_phrase(m: float) -> str:
    r = max(0.5, round(m * 2) / 2.0)
    if abs(r - round(r)) < 1e-6:
        r = int(r)
    return f"{r} meter" + ("" if r == 1 else "s")


def turn_word(deg: float) -> str:
    a = abs(deg)
    side = "left" if deg > 0 else "right"
    if a > 150:
        return "turn around"
    if a < 35:
        return f"turn slightly {side}"
    if a > 120:
        return f"make a sharp {side} turn"
    return f"turn {side}"


# -- path -> primitive actions ---------------------------------------------------------------------


def rdp(points: list, tol: float) -> list:
    if len(points) < 3:
        return list(points)
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy) or 1e-9
    dmax, idx = 0.0, 0
    for i, (x, y) in enumerate(points[1:-1], 1):
        d = abs(dy * (x - x0) - dx * (y - y0)) / n
        if d > dmax:
            dmax, idx = d, i
    if dmax > tol:
        return rdp(points[: idx + 1], tol)[:-1] + rdp(points[idx:], tol)
    return [points[0], points[-1]]


def simplify_path(points: list, tol: float = 0.3, min_seg: float = 0.6) -> list:
    pts = rdp(points, tol)
    out = [pts[0]]
    for p in pts[1:-1]:
        if math.dist(p, out[-1]) >= min_seg:
            out.append(p)
    if len(out) > 1 and math.dist(pts[-1], out[-1]) < min_seg and len(out) > 2:
        out.pop()
    out.append(pts[-1])
    return out


def path_to_actions(points: list, start_yaw: float) -> list[dict]:
    """Turn/forward primitives (degrees, metres) following the polyline ``points``."""
    acts: list[dict] = []
    yaw = start_yaw
    for a, b in zip(points, points[1:]):
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        d = wrap_angle(h - yaw)
        if abs(math.degrees(d)) >= 10:
            acts.append({"action": "turn_left" if d > 0 else "turn_right", "degrees": round(abs(math.degrees(d)))})
        acts.append({"action": "forward", "distance_m": round(math.dist(a, b), 2)})
        yaw = h
    acts.append({"action": "stop"})
    return acts


def count_turns(actions: list[dict], min_deg: float = 30.0) -> int:
    return sum(1 for a in actions if a["action"].startswith("turn") and a.get("degrees", 0) >= min_deg)


# -- landmarks ---------------------------------------------------------------------------------------


def _seg_dist(p, a, b) -> tuple[float, float, float]:
    """(distance from p to segment ab, projection parameter in metres, signed lateral offset, left positive)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy) or 1e-9
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / L
    tc = min(max(t, 0.0), L)
    cx, cy = ax + dx / L * tc, ay + dy / L * tc
    lat = (dx * (p[1] - ay) - dy * (p[0] - ax)) / L
    return math.hypot(p[0] - cx, p[1] - cy), t, lat


def landmarks_for_segment(ctx: WorldContext, a, b, max_dist: float = 2.0, skip: set | None = None) -> list[dict]:
    skip = SKIP_LANDMARK if skip is None else skip
    found = []
    for o in ctx.objects:
        if o["label"] in skip or not o.get("blocks_robot") or not labels.is_nav_target(o["label"]):
            continue
        d, t, lat = _seg_dist((o["pose"]["x"], o["pose"]["y"]), a, b)
        L = math.dist(a, b)
        if d <= max_dist and 0.0 <= t <= L and abs(lat) > 0.3:
            found.append({"label": o["label"], "instance": o["instance"], "side": "left" if lat > 0 else "right",
                          "t": t, "d": d})
    found.sort(key=lambda f: f["d"])
    # keep one landmark per label
    seen, out = set(), []
    for f in found:
        if f["label"] not in seen:
            seen.add(f["label"])
            out.append(f)
    return out


def room_sequence(ctx: WorldContext, points: list, step: float = 0.25) -> list[tuple[float, str]]:
    """[(distance along path, room name)] at each change of room."""
    seq: list[tuple[float, str]] = []
    s = 0.0
    last = None
    for a, b in zip(points, points[1:]):
        L = math.dist(a, b)
        n = max(1, int(L / step))
        for i in range(n):
            x = a[0] + (b[0] - a[0]) * i / n
            y = a[1] + (b[1] - a[1]) * i / n
            r = ctx.room_at(x, y)
            name = r["name"] if r else ""
            if r and r["type"] in ("warehouse_hall",):
                name = ""
            if name and name != last:
                seq.append((s + L * i / n, name))
                last = name
        s += L
    return seq


def nearest_landmark(ctx: WorldContext, x: float, y: float, max_dist: float = 1.8) -> dict | None:
    best = None
    for o in ctx.objects:
        if o["label"] in SKIP_REFERENCE or not o.get("blocks_robot") or not labels.is_nav_target(o["label"]):
            continue
        b = o["bbox"]
        d = math.hypot(max(b[0] - x, 0, x - b[3]), max(b[1] - y, 0, y - b[4]))
        if d <= max_dist and (best is None or d < best[0]):
            best = (d, o)
    return best[1] if best else None


# -- velocity_follow ---------------------------------------------------------------------------------


def path_instructions(ctx: WorldContext, points: list, start_yaw: float, rng: random.Random) -> list[str]:
    """Paraphrases (landmark-rich first) for the polyline ``points`` starting with heading ``start_yaw``."""
    acts = path_to_actions(points, start_yaw)
    rooms = room_sequence(ctx, points)
    goal_lm = nearest_landmark(ctx, *points[-1])
    start_room = rooms[0][1] if rooms else ""
    goal_room = rooms[-1][1] if rooms else ""

    # -- style B: landmark-rich
    sentences: list[str] = []
    yaw = start_yaw
    s_cum = 0.0
    used_rooms = {start_room}
    for i, (a, b) in enumerate(zip(points, points[1:])):
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        d = wrap_angle(h - yaw)
        L = math.dist(a, b)
        parts = []
        if abs(math.degrees(d)) >= 10:
            parts.append(turn_word(math.degrees(d)))
        walk = rng.choice(["walk forward", "go straight", "continue forward", "move ahead"]) if i else \
            rng.choice(["walk forward", "go straight", "move ahead"])
        walk += f" about {dist_phrase(L)}"
        lms = landmarks_for_segment(ctx, a, b)
        if lms:
            lm = lms[0]
            walk += f", passing {the(lm['label'])} on your {lm['side']}"
        entered = [name for (sd, name) in rooms if s_cum <= sd < s_cum + L and name not in used_rooms]
        if entered:
            walk += f" and enter the {entered[-1]}"
            used_rooms.add(entered[-1])
        parts.append(walk)
        sentences.append(" and ".join(parts) if len(parts) == 2 and parts[0].startswith("turn") else ", ".join(parts))
        yaw = h
        s_cum += L
    stop = f"Stop next to {the(goal_lm['label'])}." if goal_lm else "Stop there."
    b_text = ". ".join(s[0].upper() + s[1:] for s in sentences) + ". " + stop

    # -- style A: metric primitives only
    parts_a = []
    for a in acts:
        if a["action"] == "forward":
            parts_a.append(f"Move forward {dist_phrase(a['distance_m'])}.")
        elif a["action"].startswith("turn"):
            parts_a.append(f"Turn {a['action'][5:]} {int(round(a['degrees'] / 5.0) * 5)} degrees.")
        else:
            parts_a.append("Stop.")
    a_text = " ".join(parts_a)

    variants = [b_text, a_text]
    # -- style C: coarse, rooms and destination only
    if goal_room and start_room and goal_room != start_room:
        via = f" and stop next to {the(goal_lm['label'])}" if goal_lm else ""
        variants.append(f"Leave the {start_room}, go to the {goal_room}{via}.")
        variants.append(f"Walk from the {start_room} into the {goal_room}{via}.")
    elif goal_lm:
        variants.append(f"Walk over to {the(goal_lm['label'])} and stop.")
    # -- style D: R2R-like terse
    short = []
    for a in acts[:-1]:
        if a["action"] == "forward":
            short.append(f"go {rng.choice(['forward', 'straight'])} {dist_phrase(a['distance_m'])}")
        else:
            short.append(turn_word(a["degrees"] if a["action"] == "turn_left" else -a["degrees"]))
    variants.append((", then ".join(short) + (f", and stop by {the(goal_lm['label'])}" if goal_lm else ", and stop")).capitalize() + ".")
    return list(dict.fromkeys(variants))


# -- object_nav --------------------------------------------------------------------------------------


def _near_set(ctx: WorldContext, o: dict, thresh: float) -> set[str]:
    out = set()
    b = o["bbox"]
    for q in ctx.objects:
        if q["instance"] == o["instance"] or q["label"] == o["label"] or q["label"] in SKIP_REFERENCE:
            continue
        if not q.get("blocks_robot") or not labels.is_nav_target(q["label"]):
            continue
        c = q["bbox"]
        dx = max(b[0] - c[3], c[0] - b[3], 0.0)
        dy = max(b[1] - c[4], c[1] - b[4], 0.0)
        if math.hypot(dx, dy) <= thresh:
            out.add(q["label"])
    return out


def unique_descriptors(ctx: WorldContext, target: dict) -> list[dict]:
    """Ways to single out ``target`` among instances with the same label (empty if impossible)."""
    same = [o for o in ctx.objects if o["label"] == target["label"] and o["instance"] != target["instance"]
            and o.get("blocks_robot")]
    out = []
    for rel, th in (("next to", 1.0), ("near", 2.2)):
        mine = _near_set(ctx, target, th)
        others = set().union(*[_near_set(ctx, o, th) for o in same]) if same else set()
        for ref in sorted(mine - others):
            out.append({"kind": "relation", "relation": rel, "ref": ref,
                        "phrase": f"{rel} {the(ref)}"})
    if target.get("room") and all(o.get("room") != target["room"] for o in same):
        rn = next((r["name"] for r in ctx.rooms if r["id"] == target["room"]), "")
        if rn:
            out.append({"kind": "room", "phrase": f"in the {rn}", "room": rn})
    col = target.get("color")
    if col in COLOR_WORDS and all(o.get("color") != col for o in same):
        out.append({"kind": "color", "phrase": col, "color": col})
    return out


def object_nav_instructions(ctx: WorldContext, target: dict, descriptor: dict | None, rng: random.Random) -> list[str]:
    obj = labels.spoken(target["label"])
    n_same = sum(1 for o in ctx.objects if o["label"] == target["label"] and o.get("blocks_robot"))
    if descriptor is None:
        v = [
            f"Find {an(obj)} and stop next to it.",
            f"Go to the nearest {obj}.",
            f"Navigate to {an(obj)}.",
            f"Please find {an(obj)}; any one will do.",
        ]
        if n_same == 1:
            v.insert(0, f"Find the {obj} and stop next to it.")
        if "person-like" in target.get("attributes", []):
            v.append("Find a person and stop in front of them.")
        elif "hot" in target.get("attributes", []):
            v.append(f"Find the hot {obj} (it shows up bright on the thermal camera).")
        return v
    k = descriptor["kind"]
    if k == "relation":
        p = descriptor["phrase"]
        return [f"Find the {obj} {p} and stop next to it.", f"Go to the {obj} that is {p}.",
                f"Navigate to the {obj} {p}."]
    if k == "room":
        p = descriptor["phrase"]
        return [f"Find the {obj} {p} and stop next to it.", f"Go to the {obj} {p}.", f"Navigate to the {obj} {p}."]
    c = descriptor["color"]
    return [f"Find the {c} {obj} and stop next to it.", f"Go to the {c} {obj}.", f"Navigate to the {c} {obj}."]


def goal_pose_instructions(ctx: WorldContext, x: float, y: float, rng: random.Random) -> list[str]:
    lm = nearest_landmark(ctx, x, y)
    room = ctx.room_name(x, y)
    v = []
    if lm and room:
        v += [f"Go to the spot next to {the(lm['label'])} in the {room}.",
              f"Navigate to the {room} and stop near {the(lm['label'])}."]
    elif lm:
        v.append(f"Go to the spot next to {the(lm['label'])}.")
    elif room:
        v.append(f"Go to the middle of the {room}.")
    v.append(f"Navigate to the position x={x:.1f} m, y={y:.1f} m.")
    return v
