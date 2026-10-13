"""Episode files: YAML read/write, JSON-schema validation, semantic validation, deterministic splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from jsonschema import Draft202012Validator

from ..paths import episode_schema_path


class _Dumper(yaml.SafeDumper):
    pass


def _is_leaf(value) -> bool:
    if isinstance(value, dict):
        return all(not isinstance(v, (dict, list)) for v in value.values())
    return False


def _dict_repr(dumper, data):
    style = True if _is_leaf(data) else None
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=style)


def _list_repr(dumper, data):
    flow = all(isinstance(v, (int, float, str)) and not isinstance(v, bool) for v in data) and len(data) <= 8 \
        and not any(isinstance(v, str) and len(v) > 24 for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True if flow else None)


_Dumper.add_representer(dict, _dict_repr)
_Dumper.add_representer(list, _list_repr)


def _round(obj, nd: int = 3):
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def write_episodes(path: Path, episodes: list[dict], header: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(_round(episodes), Dumper=_Dumper, sort_keys=False, width=4000, allow_unicode=True,
                     default_flow_style=None)
    path.write_text((header if header.endswith("\n") or not header else header + "\n") + text, encoding="utf-8")


def read_episodes(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{path}: episode file must be a YAML list")
    return data


_VALIDATOR: Draft202012Validator | None = None


def schema_validator() -> Draft202012Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = Draft202012Validator(json.loads(episode_schema_path().read_text(encoding="utf-8")))
    return _VALIDATOR


def schema_errors(episode: dict) -> list[str]:
    errs = []
    for e in sorted(schema_validator().iter_errors(episode), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in e.path) or "(episode)"
        errs.append(f"{loc}: {e.message[:200]}")
    return errs


def semantic_errors(episode: dict, objects: dict[str, dict] | None, known_scenes: set[str] | None) -> list[str]:
    """Checks that need context: scene exists, goal instances exist, ids unique within file are checked by the caller."""
    errs = []
    if known_scenes is not None and episode["scene"] not in known_scenes:
        errs.append(f"scene '{episode['scene']}' is not in the registry")
    g = episode.get("goal", {})
    if g.get("type") == "object" and objects is not None:
        names = list(g.get("instances", [])) + list(g.get("other_instances", [])) + ([g["instance"]] if g.get("instance") else [])
        for n in names:
            if n not in objects:
                errs.append(f"goal instance '{n}' not in objects_file")
            elif objects[n]["label"] != g["label"]:
                errs.append(f"goal instance '{n}' has label {objects[n]['label']}, expected {g['label']}")
        if g.get("any_instance") is False and not g.get("instance"):
            errs.append("any_instance=false requires 'instance'")
    if episode.get("mode") == "object_nav" and g.get("type") != "object":
        errs.append("object_nav episode needs an object goal")
    if episode.get("mode") in ("goal_pose", "velocity_follow") and g.get("type") != "pose":
        errs.append(f"{episode['mode']} episode needs a pose goal")
    return errs


def assign_split(key: str, seed: int = 0, dev_fraction: float = 0.6) -> str:
    """Deterministic dev/heldout assignment from a stable hash (independent of Python's hash seed)."""
    h = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    return "dev" if u < dev_fraction else "heldout"
