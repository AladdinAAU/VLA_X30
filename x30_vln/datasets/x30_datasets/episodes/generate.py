"""Episode generators for registered worlds: object_nav, goal_pose, velocity_follow.

Sampling uses A* / geodesic distance fields on the free-space grid (robot radius inflated). Every
generator is a pure function of (world, seed): the same seed reproduces the same file.

Splits (documented default 60/40, ``dev_fraction``):
* ``object_nav``: ``specific_instance`` episodes by *target instance*, ``any_instance`` episodes by *class*
  (sha1 of "<seed>:<world>:<instance>" / "<seed>:<world>:class:<label>"), so a heldout target is never
  the answer of a dev episode of the same kind;
* ``goal_pose`` and ``velocity_follow``: by the *room of the goal* (sha1 of "<seed>:<world>:<room id>");
  worlds without rooms use 4 m x 4 m goal tiles.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from .. import gridmap as gm
from .. import labels
from ..paths import episodes_dir
from ..registry import register_episodes
from ..transforms import wrap_angle
from . import language as lang
from .context import WorldContext, load_context
from .io import assign_split, write_episodes

TIERS = {"easy": (3.0, 6.0), "medium": (6.0, 12.0), "hard": (12.0, 60.0)}
OBJECT_RADIUS = 1.5
POSE_RADIUS = 0.5
MIN_START_CLEARANCE = 0.45


def _time_limit(geodesic: float) -> int:
    return int(min(600, max(120, 90 + 3.5 * geodesic)))


def _tier(d: float) -> str:
    for name, (lo, hi) in TIERS.items():
        if lo <= d < hi:
            return name
    return "hard"


def _split_key_room(ctx: WorldContext, x: float, y: float) -> str:
    r = ctx.room_at(x, y)
    if r and r["type"] not in ("warehouse_hall", "corridor"):
        return r["id"]
    return f"tile_{int(x // 4)}_{int(y // 4)}"


def _clear_cells(ctx: WorldContext, min_clear: float) -> np.ndarray:
    """Reachable cells with at least ``min_clear`` clearance to obstacles (start/goal candidates)."""
    clear = ctx.grid.clearance()
    return ctx.reach & (clear >= min_clear)


def _pick_cells(rng: random.Random, mask: np.ndarray, k: int) -> list[tuple[int, int]]:
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return []
    sel = rng.sample(range(len(idx)), min(k, len(idx)))
    return [tuple(idx[i]) for i in sel]


def _start_yaw(rng: random.Random, ctx: WorldContext, start_xy, towards) -> float:
    base = math.atan2(towards[1] - start_xy[1], towards[0] - start_xy[0])
    return round(wrap_angle(base + rng.choice([0.0, 0.0, math.pi / 2, -math.pi / 2, math.pi])), 3)


def _wp(points: list) -> list[dict]:
    return [{"x": round(p[0], 2), "y": round(p[1], 2)} for p in points]


def object_nav_episodes(ctx: WorldContext, seed: int = 0, n: int = 40, dev_fraction: float = 0.6) -> list[dict]:
    rng = random.Random(f"{seed}:{ctx.world_id}:objnav")
    starts_ok = _clear_cells(ctx, MIN_START_CLEARANCE)
    candidates = [o for o in ctx.objects if labels.is_nav_target(o["label"]) and o.get("blocks_robot")
                  and o["label"] not in ("door", "window")]
    by_label: dict = {}
    for o in candidates:
        by_label.setdefault(o["label"], []).append(o)
    # reachability: target is usable when some reachable free cell lies within the success radius
    fields: dict = {}

    def field_for(o: dict) -> np.ndarray:
        if o["instance"] not in fields:
            d = ctx.bbox_distance_field(o)
            goal_cells = ctx.reach & (d <= OBJECT_RADIUS - 0.05)
            src = [tuple(c) for c in np.argwhere(goal_cells)]
            fields[o["instance"]] = gm.distance_field(ctx.blocked, src, ctx.grid.resolution) if src else None
        return fields[o["instance"]]

    jobs = []
    labels_order = sorted(by_label)
    rng.shuffle(labels_order)
    # round-robin over labels, then instances, so every class is covered before repeating
    pools = {lb: rng.sample(by_label[lb], len(by_label[lb])) for lb in labels_order}
    while len(jobs) < n * 2 and any(pools.values()):
        for lb in labels_order:
            if pools[lb]:
                jobs.append(pools[lb].pop())
    episodes: list[dict] = []
    tiers_cycle = ["easy", "medium", "hard", "medium", "easy", "hard"]
    counter = 0
    for target in jobs:
        if len(episodes) >= n:
            break
        same = by_label[target["label"]]
        descriptors = lang.unique_descriptors(ctx, target) if len(same) > 1 else []
        specific = bool(descriptors) and rng.random() < 0.5
        descriptor = rng.choice(descriptors) if specific else None
        if specific:
            f = field_for(target)
            success_ids = [target["instance"]]
        else:
            fs = [field_for(o) for o in same]
            fs = [x for x in fs if x is not None]
            if not fs:
                continue
            f = np.minimum.reduce(fs)
            success_ids = [o["instance"] for o in same]
        if f is None:
            continue
        tier_name = tiers_cycle[counter % len(tiers_cycle)]
        lo, hi = TIERS[tier_name]
        ok = starts_ok & np.isfinite(f) & (f >= lo) & (f < hi)
        if not ok.any():  # fall back to any tier with data
            ok = starts_ok & np.isfinite(f) & (f >= 2.5)
            if not ok.any():
                continue
        cell = _pick_cells(rng, ok, 1)[0]
        sx, sy = ctx.xy(cell)
        geo = float(f[cell])
        # reachable goal cells over all success instances; nearest instance (by geodesic) is the reference goal
        d_start = gm.distance_field(ctx.blocked, [cell], ctx.grid.resolution)
        best = None
        for iname in success_ids:
            cand = ctx.reach & (ctx.bbox_distance_field(ctx.by_name[iname]) <= OBJECT_RADIUS - 0.05)
            if not cand.any():
                continue
            dd = np.where(cand, d_start, np.inf)
            c = np.unravel_index(int(np.argmin(dd)), dd.shape)
            if best is None or dd[c] < best[0]:
                best = (dd[c], tuple(int(v) for v in c), ctx.by_name[iname])
        if best is None or not np.isfinite(best[0]):
            continue
        _, tc, nearest = best
        ref = ctx.path((sx, sy), ctx.xy(tc))
        if ref is None:
            continue
        counter += 1
        variants = lang.object_nav_instructions(ctx, target, descriptor, rng)
        yaw = _start_yaw(rng, ctx, (sx, sy), ref[1] if len(ref) > 1 else ref[0])
        acts = lang.path_to_actions(lang.simplify_path(ref), yaw)
        rooms_crossed = max(0, len(lang.room_sequence(ctx, ref)) - 1)
        goal = {
            "type": "object", "label": target["label"], "radius": OBJECT_RADIUS, "distance_to": "bbox",
            "any_instance": not specific, "instances": success_ids,
        }
        if specific:
            goal["instance"] = target["instance"]
            goal["other_instances"] = [o["instance"] for o in same if o["instance"] != target["instance"]]
            goal["description"] = descriptor["phrase"]
            goal["x"], goal["y"] = round(target["pose"]["x"], 2), round(target["pose"]["y"], 2)
        else:
            goal["x"], goal["y"] = round(nearest["pose"]["x"], 2), round(nearest["pose"]["y"], 2)
        episodes.append({
            "id": f"{ctx.world_id}_objnav_{len(episodes) + 1:03d}",
            "scene": ctx.world_id,
            "mode": "object_nav",
            "instruction": variants[0],
            "instruction_variants": variants,
            "start": {"x": round(sx, 2), "y": round(sy, 2), "yaw": yaw},
            "goal": goal,
            "time_limit_s": _time_limit(geo),
            "split": assign_split(f"{ctx.world_id}:{target['instance'] if specific else 'class:' + target['label']}",
                                  seed, dev_fraction),
            "reference_path": _wp(lang.simplify_path(ref)),
            "difficulty": {"tier": _tier(geo), "geodesic_m": round(geo, 2), "turns": lang.count_turns(acts),
                           "rooms_crossed": rooms_crossed, "distractors": len(same) - 1},
            "tags": ["specific_instance" if specific else "any_instance"] + ([f"descriptor_{descriptor['kind']}"] if specific else []),
            "seed": seed,
        })
    return episodes


def _sample_pairs(ctx: WorldContext, rng: random.Random, n: int, tiers: list[str], min_turns: int = 0,
                  clear: float = MIN_START_CLEARANCE, max_tries: int = 400):
    """Yield (start_cell, goal_cell, geodesic, path points) with distance in the requested tiers."""
    ok = _clear_cells(ctx, clear)
    cells = np.argwhere(ok)
    if len(cells) < 2:
        return
    produced = 0
    tries = 0
    ti = 0
    while produced < n and tries < max_tries:
        tries += 1
        s = tuple(cells[rng.randrange(len(cells))])
        lo, hi = TIERS[tiers[ti % len(tiers)]]
        d = gm.distance_field(ctx.blocked, [s], ctx.grid.resolution)
        cand = ok & np.isfinite(d) & (d >= lo) & (d < hi)
        if not cand.any():
            if tries % 8 == 0:  # the requested tier does not exist in this world (small reachable area): move on
                ti += 1
            continue
        g = _pick_cells(rng, cand, 1)[0]
        pts = ctx.path(ctx.xy(s), ctx.xy(g))
        if pts is None:
            continue
        simple = lang.simplify_path(pts)
        if len(simple) - 2 < min_turns:
            continue
        ti += 1
        produced += 1
        yield s, g, float(d[g]), pts, simple


def goal_pose_episodes(ctx: WorldContext, seed: int = 0, n: int = 20, dev_fraction: float = 0.6) -> list[dict]:
    rng = random.Random(f"{seed}:{ctx.world_id}:goalpose")
    eps = []
    for s, g, geo, pts, simple in _sample_pairs(ctx, rng, n, ["easy", "medium", "hard", "medium"]):
        sx, sy = ctx.xy(s)
        gx, gy = ctx.xy(g)
        yaw = _start_yaw(rng, ctx, (sx, sy), pts[1] if len(pts) > 1 else pts[0])
        variants = lang.goal_pose_instructions(ctx, gx, gy, rng)
        acts = lang.path_to_actions(simple, yaw)
        eps.append({
            "id": f"{ctx.world_id}_goalpose_{len(eps) + 1:03d}",
            "scene": ctx.world_id, "mode": "goal_pose",
            "instruction": variants[0], "instruction_variants": variants,
            "start": {"x": round(sx, 2), "y": round(sy, 2), "yaw": yaw},
            "goal": {"type": "pose", "x": round(gx, 2), "y": round(gy, 2), "radius": POSE_RADIUS},
            "time_limit_s": _time_limit(geo),
            "split": assign_split(f"{ctx.world_id}:{_split_key_room(ctx, gx, gy)}", seed, dev_fraction),
            "reference_path": _wp(simple),
            "difficulty": {"tier": _tier(geo), "geodesic_m": round(geo, 2), "turns": lang.count_turns(acts),
                           "rooms_crossed": max(0, len(lang.room_sequence(ctx, pts)) - 1)},
            "tags": [], "seed": seed,
        })
    return eps


def velocity_follow_episodes(ctx: WorldContext, seed: int = 0, n: int = 30, dev_fraction: float = 0.6) -> list[dict]:
    rng = random.Random(f"{seed}:{ctx.world_id}:velfollow")
    eps = []
    for s, g, geo, pts, simple in _sample_pairs(ctx, rng, n, ["easy", "medium", "medium", "hard"], min_turns=1):
        sx, sy = ctx.xy(s)
        gx, gy = ctx.xy(g)
        yaw = _start_yaw(rng, ctx, (sx, sy), simple[1])
        variants = lang.path_instructions(ctx, simple, yaw, rng)
        acts = lang.path_to_actions(simple, yaw)
        eps.append({
            "id": f"{ctx.world_id}_velfollow_{len(eps) + 1:03d}",
            "scene": ctx.world_id, "mode": "velocity_follow",
            "instruction": variants[0], "instruction_variants": variants,
            "start": {"x": round(sx, 2), "y": round(sy, 2), "yaw": yaw},
            "goal": {"type": "pose", "x": round(gx, 2), "y": round(gy, 2), "radius": POSE_RADIUS + 0.25},
            "time_limit_s": _time_limit(geo * 1.3),
            "split": assign_split(f"{ctx.world_id}:{_split_key_room(ctx, gx, gy)}", seed, dev_fraction),
            "reference_path": _wp(simple),
            "reference_actions": acts,
            "difficulty": {"tier": _tier(geo), "geodesic_m": round(geo, 2), "turns": lang.count_turns(acts),
                           "rooms_crossed": max(0, len(lang.room_sequence(ctx, pts)) - 1)},
            "tags": [], "seed": seed,
        })
    return eps


GENERATORS = {
    "object_nav": (object_nav_episodes, 40),
    "goal_pose": (goal_pose_episodes, 20),
    "velocity_follow": (velocity_follow_episodes, 30),
}


def generate_for_world(world_id: str, modes: list[str] | None = None, seed: int = 0, counts: dict | None = None,
                       dev_fraction: float = 0.6, out_dir: Path | None = None, register: bool = True,
                       ctx: WorldContext | None = None) -> dict[str, Path]:
    ctx = ctx or load_context(world_id)
    out = Path(out_dir) if out_dir else episodes_dir()
    written = {}
    for mode in modes or list(GENERATORS):
        if mode == "object_nav" and not any(o.get("blocks_robot") for o in ctx.objects):
            continue  # worlds without ground-truth objects only get pose / instruction episodes
        fn, default_n = GENERATORS[mode]
        n = (counts or {}).get(mode, default_n)
        eps = fn(ctx, seed=seed, n=n, dev_fraction=dev_fraction)
        path = out / f"{world_id}_{mode}.yaml"
        header = (f"# {mode} episodes for world '{world_id}' (generated by x30_datasets, seed {seed}, "
                  f"dev fraction {dev_fraction}). Schema: datasets/x30_datasets/schemas/episode.schema.json")
        write_episodes(path, eps, header)
        written[mode] = path
        if register:
            register_episodes(f"{world_id}_{mode}", f"episodes/{path.name}")
    return written
