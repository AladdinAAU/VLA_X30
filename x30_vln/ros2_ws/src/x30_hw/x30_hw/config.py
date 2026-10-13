"""Loading of configs/robot_real.yaml (no ROS)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml


def configs_dir() -> str:
    """configs/ directory: $X30_CONFIGS, else <repo>/configs relative to this file when running from source."""
    env = os.environ.get("X30_CONFIGS")
    if env and os.path.isdir(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(3, 7):
        cand = os.path.join(here, *([".."] * up), "configs")
        if os.path.isfile(os.path.join(cand, "robot_real.yaml")):
            return os.path.abspath(cand)
    raise FileNotFoundError("configs/robot_real.yaml not found; set X30_CONFIGS")


def load_robot_config(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or os.path.join(configs_dir(), "robot_real.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = os.path.abspath(path)
    return cfg


def stage_limits(cfg: Dict[str, Any], stage: Optional[str] = None) -> Dict[str, Any]:
    stage = stage or cfg.get("stage_default", "B")
    if stage not in cfg["stages"]:
        raise KeyError(f"unknown stage {stage}; known: {sorted(cfg['stages'])}")
    return cfg["stages"][stage]
