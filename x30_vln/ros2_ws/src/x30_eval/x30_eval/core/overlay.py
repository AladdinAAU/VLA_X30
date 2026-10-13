"""Config overlay mechanism: parameter overrides for sweeps, merged into config files at launch time.

An overlay is a YAML mapping keyed by the config file stem relative to ``configs/`` (``robot_sim``, ``models``,
``modes/object_nav``, ``eval`` ...). Each value is a (partial) dict deep-merged on top of that file's content:
dicts merge recursively, every other value (scalars, lists) replaces. ``X30_CONFIG_OVERLAY`` holds one or more overlay
file paths (os.pathsep separated, later files win). Nodes call ``apply_overlay(name, cfg)`` right after loading a
config file; ``docs/requests/eval-to-bringup.md`` asks x30_common.config to do so. This module is the reference
implementation (pure python + yaml) and is what the sweep harness uses to write and check overlays.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ENV_VAR = "X30_CONFIG_OVERLAY"


def deep_merge(base: Any, over: Any) -> Any:
    """Return ``base`` with ``over`` merged in (dicts recursively; everything else replaced). Inputs are not mutated."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = copy.deepcopy(base)
        for k, v in over.items():
            out[k] = deep_merge(base[k], v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


def set_dotted(overlay: dict, dotted: str, value: Any) -> None:
    """Set ``file.key.sub`` = value inside an overlay dict. The first dot-separated segment is the file stem
    (which may contain '/'), the rest are nested keys."""
    stem, _, rest = dotted.partition(".")
    if not rest:
        raise ValueError(f"overlay key {dotted!r} needs the form <file>.<key>[.<subkey>...]")
    node = overlay.setdefault(stem, {})
    keys = rest.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def dotted_to_overlay(params: dict[str, Any]) -> dict:
    out: dict = {}
    for k, v in params.items():
        set_dotted(out, k, v)
    return out


def load_overlay_files(paths: str | list[str] | None = None) -> dict:
    """Merge overlay files; ``paths`` defaults to $X30_CONFIG_OVERLAY."""
    if paths is None:
        paths = os.environ.get(ENV_VAR, "")
    if isinstance(paths, str):
        paths = [p for p in paths.split(os.pathsep) if p]
    merged: dict = {}
    for p in paths:
        data = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"overlay {p}: top level must be a mapping")
        merged = deep_merge(merged, data)
    return merged


def apply_overlay(name: str, cfg: dict, overlay: dict | None = None) -> dict:
    """Apply the part of the overlay that targets config file ``name`` (stem relative to configs/)."""
    ov = load_overlay_files() if overlay is None else overlay
    part = ov.get(name)
    return deep_merge(cfg, part) if isinstance(part, dict) else cfg


def write_overlay(path: str | Path, overlay: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(overlay, sort_keys=True), encoding="utf-8")
    return p


def check_overlay_against(configs_dir: str | Path, overlay: dict) -> list[str]:
    """Names of overlay entries that do not exist in the target config file (typos in a sweep would otherwise be
    silently ignored). Returns a list of 'file.key.path' strings; empty when everything resolves."""
    missing: list[str] = []

    def walk(base: Any, over: Any, prefix: str) -> None:
        if not isinstance(over, dict):
            return
        for k, v in over.items():
            if not isinstance(base, dict) or k not in base:
                missing.append(f"{prefix}.{k}")
            else:
                walk(base[k], v, f"{prefix}.{k}")

    for stem, part in overlay.items():
        f = Path(configs_dir) / f"{stem}.yaml"
        if not f.exists():
            missing.append(stem)
            continue
        walk(yaml.safe_load(f.read_text(encoding="utf-8")) or {}, part, stem)
    return missing
