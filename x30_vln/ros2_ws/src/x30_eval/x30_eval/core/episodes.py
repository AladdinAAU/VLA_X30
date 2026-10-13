"""Episode files (INTERFACES.md section 6): loading, validation and split selection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

MODES = ("object_nav", "velocity_follow", "goal_pose", "explore")
SPLITS = ("dev", "heldout")
GOAL_TYPES = ("object", "pose")


class EpisodeError(ValueError):
    """Raised for a malformed episode file."""


@dataclass
class Episode:
    id: str
    scene: str
    mode: str
    instruction: str
    start: dict
    goal: dict
    time_limit_s: float = 300.0
    split: str = "dev"
    reference_path: list = field(default_factory=list)
    difficulty: str = ""
    params: dict = field(default_factory=dict)       # extra Navigate params (params_json)
    shortest_path_m: float | None = None             # optional precomputed optimum (overrides grid A*)
    extra: dict = field(default_factory=dict)        # unknown keys, preserved (fake_policy, notes, ...)

    @property
    def goal_type(self) -> str:
        return self.goal.get("type", "pose")

    @property
    def target_label(self) -> str:
        return str(self.goal.get("label", ""))

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "scene": self.scene, "mode": self.mode, "instruction": self.instruction,
            "start": self.start, "goal": self.goal, "time_limit_s": self.time_limit_s, "split": self.split,
            "reference_path": self.reference_path,
        }
        if self.difficulty:
            d["difficulty"] = self.difficulty
        if self.params:
            d["params"] = self.params
        if self.shortest_path_m is not None:
            d["shortest_path_m"] = self.shortest_path_m
        d.update(self.extra)
        return d


_KNOWN = {"id", "scene", "mode", "instruction", "start", "goal", "time_limit_s", "split", "reference_path",
          "difficulty", "params", "shortest_path_m"}


def _validate(raw: dict, where: str) -> Episode:
    def need(key):
        if key not in raw:
            raise EpisodeError(f"{where}: missing '{key}'")
        return raw[key]

    eid = str(need("id"))
    where = f"{where} ({eid})"
    start = dict(need("start"))
    for k in ("x", "y"):
        if k not in start:
            raise EpisodeError(f"{where}: start needs '{k}'")
    start.setdefault("yaw", 0.0)
    start.setdefault("z", 0.0)
    goal = dict(need("goal"))
    gtype = goal.get("type")
    if gtype not in GOAL_TYPES:
        raise EpisodeError(f"{where}: goal.type must be one of {GOAL_TYPES}, got {gtype!r}")
    if gtype == "object" and not goal.get("label"):
        raise EpisodeError(f"{where}: object goal needs 'label'")
    if gtype == "pose" and not all(k in goal for k in ("x", "y")):
        raise EpisodeError(f"{where}: pose goal needs x and y")
    mode = str(raw.get("mode", "object_nav"))
    if mode not in MODES:
        raise EpisodeError(f"{where}: mode must be one of {MODES}, got {mode!r}")
    split = str(raw.get("split", "dev"))
    if split not in SPLITS:
        raise EpisodeError(f"{where}: split must be one of {SPLITS}, got {split!r}")
    limit = float(raw.get("time_limit_s", 300.0))
    if limit <= 0:
        raise EpisodeError(f"{where}: time_limit_s must be positive")
    sp = raw.get("shortest_path_m")
    diff = raw.get("difficulty", "")
    extra = {k: v for k, v in raw.items() if k not in _KNOWN}
    if isinstance(diff, dict):                               # datasets/ schema: {tier, geodesic_m, turns, ...}
        extra["difficulty_info"] = diff
        if diff.get("geodesic_m") is not None:
            extra["geodesic_hint_m"] = float(diff["geodesic_m"])
        diff = diff.get("tier", "")
    return Episode(
        id=eid, scene=str(need("scene")), mode=mode, instruction=str(raw.get("instruction", "")),
        start=start, goal=goal, time_limit_s=limit, split=split,
        reference_path=[[float(p["x"]), float(p["y"])] if isinstance(p, dict) else [float(p[0]), float(p[1])] for p in raw.get("reference_path") or []],
        difficulty=str(diff), params=dict(raw.get("params") or {}),
        shortest_path_m=None if sp is None else float(sp),
        extra=extra,
    )


def load_episodes(path: str | Path) -> list[Episode]:
    """Load a YAML (list, or {episodes: [...]}) or JSONL episode file and validate every entry."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        raws = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = yaml.safe_load(text)
        raws = data.get("episodes", []) if isinstance(data, dict) else data
    if not isinstance(raws, list) or not raws:
        raise EpisodeError(f"{p}: no episodes found")
    eps = [_validate(r, f"{p.name}[{i}]") for i, r in enumerate(raws)]
    ids = [e.id for e in eps]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise EpisodeError(f"{p}: duplicate episode ids {dup}")
    return eps


def select(episodes: Iterable[Episode], split: str = "dev", modes: Iterable[str] | None = None,
           ids: Iterable[str] | None = None, scenes: Iterable[str] | None = None) -> list[Episode]:
    """Filter by split ('dev' | 'heldout' | 'all'), mode, id and scene."""
    modes, ids, scenes = (set(x) if x else None for x in (modes, ids, scenes))
    out = []
    for e in episodes:
        if split != "all" and e.split != split:
            continue
        if modes and e.mode not in modes:
            continue
        if ids and e.id not in ids:
            continue
        if scenes and e.scene not in scenes:
            continue
        out.append(e)
    return out


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_key(episode_id: str, repeat: int) -> str:
    return f"{episode_id}#r{repeat}"


def unique_run_id(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
