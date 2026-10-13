"""Project paths and the evaluation config (configs/eval/eval.yaml + overlay + CLI overrides)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from .overlay import apply_overlay, deep_merge, load_overlay_files


def project_root(start: str | Path | None = None) -> Path:
    """The x30_vln directory: $X30_ROOT, else the nearest parent of ``start`` (default: this file, then cwd)
    that contains both ``configs`` and ``ros2_ws``."""
    env = os.environ.get("X30_ROOT")
    if env:
        return Path(env).resolve()
    for origin in ([Path(start)] if start else []) + [Path(__file__).resolve(), Path.cwd()]:
        for parent in [origin] + list(origin.parents):
            if (parent / "configs").is_dir() and (parent / "ros2_ws").is_dir():
                return parent
    return Path.cwd()


def default_config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / "configs" / "eval" / "eval.yaml"


def load_eval_config(path: str | Path | None = None, overlay: dict | None = None, root: Path | None = None) -> dict:
    """configs/eval/eval.yaml, then the overlay entry ``eval`` (from ``overlay`` or $X30_CONFIG_OVERLAY)."""
    p = Path(path) if path else default_config_path(root)
    if not p.exists():
        raise FileNotFoundError(f"evaluation config not found: {p} (set X30_ROOT or pass --config)")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return apply_overlay("eval", cfg, overlay if overlay is not None else load_overlay_files())


def merge_overrides(cfg: dict, overrides: dict | None) -> dict:
    return deep_merge(cfg, overrides or {})


def success_radius(cfg: dict, goal: dict) -> float:
    if goal.get("radius") is not None:
        return float(goal["radius"])
    key = "object_radius_m" if goal.get("type") == "object" else "pose_radius_m"
    return float(cfg["success"][key])
