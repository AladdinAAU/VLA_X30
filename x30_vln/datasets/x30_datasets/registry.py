"""``datasets/registry.yaml``: worlds and episode sets known to the framework.

Format (paths of a world entry are relative to the share directory of the ``x30_worlds`` package,
i.e. ``ros2_ws/src/x30_worlds`` in the source tree; episode set paths are relative to ``datasets/``)::

    worlds:
      <id>: {sdf, spawn: {x,y,z,yaw}, bounds: [xmin,ymin,xmax,ymax], objects_file, license, source,
             source_dataset, rtf_note, [rooms_file, map_file, model_paths, scene_transform, stats]}
    episodes:
      <set id>: episodes/<file>.yaml
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .paths import datasets_dir, registry_path, worlds_pkg_dir

REQUIRED_WORLD_KEYS = ("sdf", "spawn", "bounds", "objects_file", "license", "source", "source_dataset", "rtf_note")


def load() -> dict:
    p = registry_path()
    if not p.exists():
        return {"worlds": {}, "episodes": {}}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("worlds", {})
    data.setdefault("episodes", {})
    return data


def save(reg: dict) -> None:
    reg = {"worlds": dict(sorted(reg["worlds"].items())), "episodes": dict(sorted(reg["episodes"].items()))}
    header = (
        "# World and episode registry (see x30_datasets/registry.py for the format).\n"
        "# World paths are relative to the x30_worlds package share dir; episode paths relative to datasets/.\n"
    )
    registry_path().write_text(header + yaml.safe_dump(reg, sort_keys=False, width=100), encoding="utf-8")


def register_world(world_id: str, entry: dict) -> None:
    reg = load()
    old = reg["worlds"].get(world_id, {})
    for key in ("rtf_note",):  # keep measured notes when a world is regenerated
        if old.get(key) and not old[key].startswith("not measured") and entry.get(key, "").startswith("not measured"):
            entry[key] = old[key]
    reg["worlds"][world_id] = entry
    save(reg)


def register_episodes(set_id: str, rel_path: str) -> None:
    reg = load()
    reg["episodes"][set_id] = rel_path
    save(reg)


def world_file(world_id: str, key: str = "sdf") -> Path:
    entry = load()["worlds"][world_id]
    return worlds_pkg_dir() / entry[key]


def world_entry(world_id: str) -> dict:
    reg = load()
    if world_id not in reg["worlds"]:
        raise KeyError(f"world '{world_id}' not in registry (known: {sorted(reg['worlds'])})")
    return reg["worlds"][world_id]


def episode_path(set_id: str) -> Path:
    return datasets_dir() / load()["episodes"][set_id]
