"""FAKE ENVIRONMENT for tests: a 2D three-room world with labelled objects, a skill server and a semantic map in pure python.

It is NOT a model of the robot. It only exists to run the agents end to end many times without ROS, to catch loops, bad
guard logic and crashes. Success rates measured here say nothing about the X30. Skill results copy the shapes of
configs/skills.yaml (`data` fields), so the same agent code runs unchanged against the real skill server.
"""
from __future__ import annotations

import heapq
import math
import random
import threading
from dataclasses import dataclass, field

import numpy as np

from x30_agent.core.types import AgentEnv, Pose2D, SkillResult

W, H, RES = 12.0, 8.0, 0.1
ROOMS = {"living room": (0, 0, 5, 8), "kitchen": (5, 0, 12, 4), "office": (5, 4, 12, 8)}
# (x0, y0, x1, y1) wall segments with door gaps left out
WALLS = [(5.0, 0.0, 5.0, 2.6), (5.0, 4.4, 5.0, 8.0), (5.8, 4.0, 8.0, 4.0), (9.2, 4.0, 12.0, 4.0)]
COLORS = ["red", "blue", "green", "white", "black"]
LABELS = ["chair", "fire extinguisher", "potted plant", "trash can", "couch", "refrigerator", "table", "tv"]


@dataclass
class Obj:
    id: int
    label: str
    x: float
    y: float
    color: str
    room: str
    size: float = 0.6
    seen: bool = False


class FakeWorld(AgentEnv):
    def __init__(self, seed: int = 0, noise: float = 0.0, start: tuple = (1.0, 1.0, 0.0)):
        self.rng = random.Random(seed)
        self.noise = noise                      # probability that verify answers wrongly
        self.pos = Pose2D(*start)
        self.t = 0.0
        self.path_len = 0.0
        self.objects: list[Obj] = []
        self.grid = self._make_grid()
        self.explored = np.zeros_like(self.grid, dtype=bool)
        self.skill_calls: list[tuple[str, dict]] = []
        self.rejected_seen: set = set()
        self.last_look = 0.0
        self._populate()
        self._reveal(self.pos, 360)

    # ------------------------------------------------------------------ world construction
    def _make_grid(self) -> np.ndarray:
        g = np.zeros((int(H / RES), int(W / RES)), dtype=np.uint8)
        g[0, :] = g[-1, :] = 1
        g[:, 0] = g[:, -1] = 1
        for x0, y0, x1, y1 in WALLS:
            for i in range(int(max(abs(x1 - x0), abs(y1 - y0)) / RES) + 1):
                f = i * RES / max(1e-6, max(abs(x1 - x0), abs(y1 - y0)))
                cx, cy = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
                g[min(int(cy / RES), g.shape[0] - 1), min(int(cx / RES), g.shape[1] - 1)] = 1
        return g

    def room_of(self, x: float, y: float) -> str:
        for name, (x0, y0, x1, y1) in ROOMS.items():
            if x0 <= x < x1 and y0 <= y < y1:
                return name
        return "hall"

    def _free(self, x: float, y: float, margin: float = 0.4) -> bool:
        r = int(margin / RES)
        j, i = int(y / RES), int(x / RES)
        if i < 0 or j < 0 or j >= self.grid.shape[0] or i >= self.grid.shape[1]:
            return False
        return not self.grid[max(0, j - r): j + r + 1, max(0, i - r): i + r + 1].any()

    def _populate(self) -> None:
        oid = 1
        n = self.rng.randint(9, 14)
        while len(self.objects) < n:
            x, y = self.rng.uniform(0.8, W - 0.8), self.rng.uniform(0.8, H - 0.8)
            if not self._free(x, y, 0.5) or any(math.hypot(o.x - x, o.y - y) < 1.2 for o in self.objects):
                continue
            self.objects.append(Obj(oid, self.rng.choice(LABELS), x, y, self.rng.choice(COLORS), self.room_of(x, y)))
            oid += 1
        self.next_id = oid

    def make_task(self) -> tuple[str, Obj]:
        """A task whose target is unique by (label, colour, room); distractors of the same label are ensured."""
        for _ in range(200):
            o = self.rng.choice(self.objects)
            same = [p for p in self.objects if p.label == o.label]
            if sum(1 for p in same if p.color == o.color and p.room == o.room) == 1:
                style = self.rng.random()
                if len(same) == 1:
                    return f"Find the {o.label} and stop next to it.", o
                if style < 0.5 and sum(1 for p in same if p.room == o.room) == 1:
                    return f"Go to the {o.label} in the {o.room}.", o
                if sum(1 for p in same if p.color == o.color) == 1:
                    return f"Find the {o.color} {o.label} and stop next to it.", o
                return f"Find the {o.color} {o.label} in the {o.room}.", o
        o = self.objects[0]
        return f"Find the {o.label}.", o

    # ------------------------------------------------------------------ geometry
    def _line_free(self, a: tuple, b: tuple) -> bool:
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / (RES / 2)) + 1
        for k in range(n + 1):
            x, y = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
            if self.grid[min(int(y / RES), self.grid.shape[0] - 1), min(int(x / RES), self.grid.shape[1] - 1)]:
                return False
        return True

    def _reveal(self, p: Pose2D, sweep_deg: float, rng_m: float = 5.0) -> list[Obj]:
        new = []
        half = math.radians(sweep_deg) / 2
        for o in self.objects:
            d = math.hypot(o.x - p.x, o.y - p.y)
            b = _wrap(math.atan2(o.y - p.y, o.x - p.x) - p.yaw)
            if d <= rng_m and abs(b) <= half + 1e-9 and self._line_free((p.x, p.y), (o.x, o.y)):
                if not o.seen:
                    o.seen = True
                    new.append(o)
        j, i = int(p.y / RES), int(p.x / RES)
        r = int(rng_m / RES)
        self.explored[max(0, j - r): j + r + 1, max(0, i - r): i + r + 1] |= True
        return new

    def _astar(self, a: tuple, b: tuple, want_path: bool = False):
        """Path length in metres on the 8-connected grid with a 0.35 m obstacle margin, or None."""
        sg = (int(a[1] / RES), int(a[0] / RES))
        gg = (int(b[1] / RES), int(b[0] / RES))
        margin = int(0.4 / RES)
        blocked = self.grid.copy()
        if margin:
            from itertools import product
            ys, xs = np.nonzero(self.grid)
            for dy, dx in product(range(-margin, margin + 1), repeat=2):
                blocked[np.clip(ys + dy, 0, blocked.shape[0] - 1), np.clip(xs + dx, 0, blocked.shape[1] - 1)] = 1
        blocked[sg] = blocked[gg] = 0
        openq = [(0.0, sg)]
        g = {sg: 0.0}
        came = {}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == gg:
                if want_path:
                    pts = [cur]
                    while pts[-1] in came:
                        pts.append(came[pts[-1]])
                    return [((c[1] + 0.5) * RES, (c[0] + 0.5) * RES) for c in reversed(pts)]
                return g[cur] * RES
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx = (cur[0] + dy, cur[1] + dx)
                if not (0 <= nx[0] < blocked.shape[0] and 0 <= nx[1] < blocked.shape[1]) or blocked[nx]:
                    continue
                ng = g[cur] + (1.414 if dy and dx else 1.0)
                if ng < g.get(nx, 1e9):
                    g[nx] = ng
                    came[nx] = cur
                    heapq.heappush(openq, (ng + math.hypot(gg[0] - nx[0], gg[1] - nx[1]), nx))
        return None

    def _teleport(self, x: float, y: float, yaw: float | None = None) -> bool:
        """Drive to (x, y) along the A* path: costs time and path length, updates exploration along the way."""
        length = self._astar((self.pos.x, self.pos.y), (x, y))
        if length is None:
            return False
        steps = max(1, int(length / 1.5))
        for k in range(1, steps + 1):                 # reveal along the way (straight-line approximation)
            f = k / steps
            self.pos = Pose2D(self.pos.x + (x - self.pos.x) / (steps - k + 1), self.pos.y + (y - self.pos.y) / (steps - k + 1), self.pos.yaw)
            self._reveal(Pose2D(self.pos.x, self.pos.y, math.atan2(y - self.pos.y, x - self.pos.x)), 120, 4.0)
        self.pos = Pose2D(x, y, self.pos.yaw if yaw is None else yaw)
        self.path_len += length
        self.t += 2.0 + length / 0.5
        return True

    # ------------------------------------------------------------------ AgentEnv
    def now(self) -> float:
        return self.t

    def pose(self) -> Pose2D:
        return Pose2D(self.pos.x, self.pos.y, self.pos.yaw)

    def _obj_line(self, o: Obj) -> str:
        d = math.hypot(o.x - self.pos.x, o.y - self.pos.y)
        b = math.degrees(_wrap(math.atan2(o.y - self.pos.y, o.x - self.pos.x) - self.pos.yaw))
        return f"#{o.id} {o.label} ({o.x:.1f}, {o.y:.1f}) {d:.1f} m, bearing {b:.0f}, {o.room}, {o.color}"

    def map_summary(self, pose, max_objects: int = 30) -> str:
        seen = sorted((o for o in self.objects if o.seen), key=lambda o: math.hypot(o.x - self.pos.x, o.y - self.pos.y))
        head = f"{len(seen)} objects mapped; {100 * self.explored.mean():.0f}% explored; robot in {self.room_of(self.pos.x, self.pos.y)}."
        return head + "".join("\n" + self._obj_line(o) for o in seen[:max_objects])

    def object_info(self, object_id: int):
        for o in self.objects:
            if o.id == object_id and o.seen:
                return {"id": o.id, "label": o.label, "x": o.x, "y": o.y, "size": [o.size, o.size]}
        return None

    def query_map(self, op: str, **kw) -> dict:
        if op == "list":
            lab = str(kw.get("label", "")).lower()
            hits = [o for o in self.objects if o.seen and (not lab or lab in o.label)]
            return {"ok": True, "text": f"{len(hits)} match(es): " + "; ".join(self._obj_line(o) for o in hits[:6]) if hits else "no match in the map"}
        if op == "get":
            o = self.object_info(int(kw.get("id", -1)))
            if not o:
                return {"ok": False, "error": "unknown object id"}
            return {"ok": True, "text": self._obj_line(next(p for p in self.objects if p.id == o["id"]))}
        return {"ok": False, "error": f"op {op} not supported in the fake world"}

    def map_image(self):
        from PIL import Image, ImageDraw
        s = 40
        img = Image.new("RGB", (int(W * s * 0.5), int(H * s * 0.5)), (30, 30, 30))
        d = ImageDraw.Draw(img)
        k = s * 0.5
        for (x0, y0, x1, y1) in ((a, b, c, e) for a, b, c, e in WALLS):
            d.line([x0 * k, y0 * k, x1 * k, y1 * k], fill=(200, 200, 200), width=2)
        d.rectangle([0, 0, W * k - 1, H * k - 1], outline=(200, 200, 200))
        for o in self.objects:
            if o.seen:
                d.ellipse([o.x * k - 3, o.y * k - 3, o.x * k + 3, o.y * k + 3], fill=(230, 160, 40))
                d.text((o.x * k + 4, o.y * k - 5), f"{o.id} {o.label}", fill=(255, 255, 255))
        d.polygon([(self.pos.x * k + 8 * math.cos(self.pos.yaw), self.pos.y * k + 8 * math.sin(self.pos.yaw)),
                   (self.pos.x * k + 5 * math.cos(self.pos.yaw + 2.5), self.pos.y * k + 5 * math.sin(self.pos.yaw + 2.5)),
                   (self.pos.x * k + 5 * math.cos(self.pos.yaw - 2.5), self.pos.y * k + 5 * math.sin(self.pos.yaw - 2.5))], fill=(60, 200, 90))
        return img

    def camera_image(self):
        """Column ray-cast view of the walls plus object sprites with their label printed on them."""
        from PIL import Image, ImageDraw
        w, h = 320, 200
        img = Image.new("RGB", (w, h), (90, 90, 100))
        d = ImageDraw.Draw(img)
        d.rectangle([0, h // 2, w, h], fill=(70, 60, 50))
        fov = math.radians(90)
        for c in range(w):
            a = self.pos.yaw + fov / 2 - fov * c / w
            dist = 8.0
            for k in range(1, 160):
                rr = k * 0.05
                x, y = self.pos.x + rr * math.cos(a), self.pos.y + rr * math.sin(a)
                if self.grid[min(int(y / RES), self.grid.shape[0] - 1), min(int(x / RES), self.grid.shape[1] - 1)]:
                    dist = rr
                    break
            hh = min(h, int(h / max(0.3, dist) * 0.7))
            shade = max(40, 200 - int(dist * 20))
            d.line([c, h // 2 - hh // 2, c, h // 2 + hh // 2], fill=(shade, shade, shade + 10))
        for o in sorted(self.objects, key=lambda o: -math.hypot(o.x - self.pos.x, o.y - self.pos.y)):
            dx, dy = o.x - self.pos.x, o.y - self.pos.y
            dist = math.hypot(dx, dy)
            b = _wrap(math.atan2(dy, dx) - self.pos.yaw)
            if dist < 0.2 or abs(b) > fov / 2 or not self._line_free((self.pos.x, self.pos.y), (o.x, o.y)):
                continue
            cx = w / 2 - b / fov * w
            sz = max(14, int(h / dist * 0.5))
            rgb = {"red": (210, 40, 40), "blue": (40, 70, 210), "green": (40, 170, 70), "white": (235, 235, 235), "black": (25, 25, 25)}[o.color]
            d.rectangle([cx - sz / 2, h / 2 - sz / 2 + sz / 4, cx + sz / 2, h / 2 + sz / 2 + sz / 4], fill=rgb)
            d.text((cx - sz / 2, h / 2 - sz / 2 - 8), o.label, fill=(255, 255, 0))
        return img

    # ------------------------------------------------------------------ skills
    def run_skill(self, name: str, args: dict, timeout_s: float, cancel: threading.Event) -> SkillResult:
        self.skill_calls.append((name, dict(args)))
        if cancel.is_set():
            return SkillResult(False, "cancelled", cancelled=True)
        fn = getattr(self, f"_sk_{name}", None)
        if fn is None:
            return SkillResult(False, f"unknown skill {name}")
        t0 = self.t
        res = fn(args)
        res.duration_s = self.t - t0
        return res

    def _find(self, args: dict, need_seen: bool = True) -> Obj | None:
        if args.get("object_id") is not None:
            return next((o for o in self.objects if o.id == int(args["object_id"]) and (o.seen or not need_seen)), None)
        lab = str(args.get("label", "")).lower()
        rej = set(args.get("rejected_ids") or [])
        c = [o for o in self.objects if o.seen and lab in o.label and o.id not in rej]
        return min(c, key=lambda o: math.hypot(o.x - self.pos.x, o.y - self.pos.y)) if c else None

    def _cands(self, label: str, rejected: set) -> list[dict]:
        lab = label.lower()
        out = []
        for o in self.objects:
            if o.seen and (lab in o.label or o.label in lab) and o.id not in rejected:
                d = math.hypot(o.x - self.pos.x, o.y - self.pos.y)
                b = math.degrees(_wrap(math.atan2(o.y - self.pos.y, o.x - self.pos.x) - self.pos.yaw))
                out.append({"id": o.id, "label": o.label, "confidence": 0.8, "x": round(o.x, 2), "y": round(o.y, 2), "distance_m": round(d, 1),
                            "bearing_deg": round(b), "room": o.room, "caption": o.color})
        return sorted(out, key=lambda c: c["distance_m"])

    def _frontier(self) -> tuple | None:
        best = None
        for _ in range(60):
            x, y = self.rng.uniform(0.6, W - 0.6), self.rng.uniform(0.6, H - 0.6)
            if not self._free(x, y, 0.4) or self.explored[int(y / RES), int(x / RES)]:
                continue
            d = self._astar((self.pos.x, self.pos.y), (x, y))
            if d is not None and (best is None or d < best[0]):
                best = (d, x, y)
        return best[1:] if best else None

    def _sk_explore(self, a: dict) -> SkillResult:
        fr = self._frontier()
        if fr is None:
            return SkillResult(False, "No frontier left: the reachable map is fully explored. Next: try search with another label or finish.",
                               {"visited": [], "new_objects": [], "remaining_frontiers": 0, "exhausted": True, "found": None})
        before = {o.id for o in self.objects if o.seen}
        self._teleport(*fr)
        self._reveal(self.pos, 360)
        self.t += 10
        new = [o for o in self.objects if o.seen and o.id not in before]
        found = None
        if a.get("until_label"):
            c = self._cands(a["until_label"], set())
            found = c[0] if c else None
        return SkillResult(True, f"Explored frontier at ({fr[0]:.1f}, {fr[1]:.1f}); {len(new)} new object(s).",
                           {"visited": [[round(fr[0], 1), round(fr[1], 1)]], "new_objects": [{"id": o.id, "label": o.label} for o in new],
                            "remaining_frontiers": 1, "exhausted": False, "found": found})

    def _sk_search(self, a: dict) -> SkillResult:
        rej = set(a.get("rejected_ids") or [])
        budget = float(a.get("budget_s", 120))
        t0 = self.t
        while True:
            c = self._cands(a["label"], rej)
            if c:
                return SkillResult(True, f"Found {len(c)} candidate(s) for '{a['label']}', best #{c[0]['id']} at {c[0]['distance_m']} m.",
                                   {"candidates": c, "explored": self.t > t0, "exhausted": False})
            if self.t - t0 > budget:
                return SkillResult(False, "Search budget ended without a candidate. Next: larger budget_s or another label.",
                                   {"candidates": [], "explored": True, "exhausted": False})
            r = self._sk_explore({})
            if r.data.get("exhausted"):
                return SkillResult(False, f"Nothing found for '{a['label']}'; the map is exhausted.", {"candidates": [], "explored": True, "exhausted": True})

    def _sk_goto_object(self, a: dict) -> SkillResult:
        o = self._find(a)
        if o is None:
            return SkillResult(False, "unknown object id: call search, or read the map summary.")
        stand = float(a.get("standoff_m", 1.0))
        ang = math.atan2(self.pos.y - o.y, self.pos.x - o.x)
        for da in (0, 0.6, -0.6, 1.2, -1.2, 2.0, -2.0):
            x, y = o.x + (stand + o.size / 2) * math.cos(ang + da), o.y + (stand + o.size / 2) * math.sin(ang + da)
            if self._free(x, y, 0.3) and self._teleport(x, y, math.atan2(o.y - y, o.x - x)):
                self._reveal(self.pos, 120)
                d = math.hypot(o.x - self.pos.x, o.y - self.pos.y)
                return SkillResult(True, f"Arrived {d:.1f} m from #{o.id} {o.label}.",
                                   {"object_id": o.id, "label": o.label, "distance_m": round(d, 2), "bearing_deg": 0.0,
                                    "robot_x": round(self.pos.x, 2), "robot_y": round(self.pos.y, 2), "robot_yaw_deg": round(math.degrees(self.pos.yaw))})
        return SkillResult(False, "no free standoff pose: try a smaller standoff_m or goto_pose to a nearby point.")

    def _sk_approach(self, a: dict) -> SkillResult:
        o = self._find(a)
        if o is None:
            return SkillResult(False, "object not visible to the lidar: goto_object first, or turn_to the object.")
        r = self._sk_goto_object({"object_id": o.id, "standoff_m": max(0.5, float(a.get("standoff_m", 0.8)))})
        if r.success:
            return SkillResult(True, f"In front of #{o.id}, range {r.data['distance_m']} m.",
                               {"object_id": o.id, "range_m": r.data["distance_m"], "standoff_m": a.get("standoff_m", 0.8), "bearing_error_deg": 2.0, "used_lidar": True})
        return r

    def _sk_goto_pose(self, a: dict) -> SkillResult:
        if not self._free(float(a["x"]), float(a["y"]), 0.3):
            return SkillResult(False, "no path: the target is not in free space. Next: pick another point or explore.")
        if self._teleport(float(a["x"]), float(a["y"]), None if "yaw_deg" not in a else math.radians(a["yaw_deg"])):
            return SkillResult(True, "Arrived.", {"final_x": round(self.pos.x, 2), "final_y": round(self.pos.y, 2), "remaining_m": 0.0})
        return SkillResult(False, "no path to the target.")

    def _sk_turn_to(self, a: dict) -> SkillResult:
        old = self.pos.yaw
        if a.get("object_id") is not None:
            o = self._find(a)
            if o is None:
                return SkillResult(False, "unknown object id")
            self.pos.yaw = math.atan2(o.y - self.pos.y, o.x - self.pos.x)
        elif a.get("degrees") is not None:
            self.pos.yaw = _wrap(self.pos.yaw + math.radians(float(a["degrees"])))
        elif a.get("heading_deg") is not None:
            self.pos.yaw = math.radians(float(a["heading_deg"]))
        else:
            self.pos.yaw = _wrap(self.pos.yaw + math.radians(_bearing(a.get("bearing", 0))))
        self.t += 2
        self._reveal(self.pos, 120)
        return SkillResult(True, "Turned.", {"achieved_deg": round(math.degrees(_wrap(self.pos.yaw - old)), 1), "heading_map_deg": round(math.degrees(self.pos.yaw))})

    def _sk_move_relative(self, a: dict) -> SkillResult:
        dist = float(a["distance_cm"]) / 100.0
        ang = {"forward": 0.0, "backward": math.pi, "left": math.pi / 2, "right": -math.pi / 2}[a.get("direction", "forward")]
        moved = 0.0
        while moved < dist - 1e-6:
            step = min(0.05, dist - moved)
            nx = self.pos.x + step * math.cos(self.pos.yaw + ang)
            ny = self.pos.y + step * math.sin(self.pos.yaw + ang)
            if not self._free(nx, ny, 0.2):
                break
            self.pos = Pose2D(nx, ny, self.pos.yaw)
            moved += step
        self.path_len += moved
        self.t += 1 + moved / 0.3
        self._reveal(self.pos, 120)
        blocked = moved < dist - 0.02
        msg = f"Moved {moved * 100:.0f} cm." if not blocked else f"Blocked after {moved * 100:.0f} cm of {dist * 100:.0f}: obstacle ahead (safety guard). Next: turn_to or strafe."
        return SkillResult(not blocked, msg, {"requested_cm": dist * 100, "achieved_cm": round(moved * 100, 1), "blocked": blocked,
                                              "obstacle_distance_cm": 20 if blocked else None})

    def _sk_scan(self, a: dict) -> SkillResult:
        new = self._reveal(Pose2D(self.pos.x, self.pos.y, self.pos.yaw), 360)
        self.t += 25
        return SkillResult(True, f"Scan done, {len(new)} new object(s).", {"new_objects": [{"id": o.id, "label": o.label} for o in new], "nearby": [], "stops": 6})

    def _sk_stop(self, a: dict) -> SkillResult:
        return SkillResult(True, "Stopped.", {"cancelled_skill": None})

    def _sk_verify(self, a: dict) -> SkillResult:
        o = self._find(a)
        question = (str(a.get("question", "")) + " " + str(a.get("label", ""))).lower()
        if o is None:
            return SkillResult(True, "No object in view.", {"answer": "no", "plausibility": 5, "reason": "nothing there", "object_id": None, "distance_m": None})
        if a.get("face_object", True):
            self.pos.yaw = math.atan2(o.y - self.pos.y, o.x - self.pos.x)
        d = math.hypot(o.x - self.pos.x, o.y - self.pos.y)
        self.t += 3
        words_ok = o.label in question or not any(l in question for l in LABELS)
        color_q = [c for c in COLORS if c in question]
        room_q = [r for r in ROOMS if r in question]
        match = words_ok and (not color_q or o.color in color_q) and (not room_q or o.room in room_q)
        if d > 4.0:
            return SkillResult(True, "Too far to tell.", {"answer": "unsure", "plausibility": 40, "reason": "far away", "object_id": o.id, "distance_m": round(d, 2)})
        if self.rng.random() < self.noise:
            match = not match
        return SkillResult(True, f"Looks {'like a match' if match else 'different'}.",
                           {"answer": "yes" if match else "no", "plausibility": 92 if match else 8, "reason": f"{o.color} {o.label} in the {o.room}",
                            "object_id": o.id, "distance_m": round(d, 2)})


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _bearing(b) -> float:
    words = {"ahead": 0, "left": 90, "right": -90, "behind": 180, "around": 180}
    return words.get(str(b).lower(), float(b) if str(b).lstrip("-").replace(".", "").isdigit() else 0.0)


def ground_truth_success(world: FakeWorld, target: Obj, radius: float = 1.5) -> bool:
    return math.hypot(world.pos.x - target.x, world.pos.y - target.y) <= radius
