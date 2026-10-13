"""Locate the project `configs/` directory and load YAML config files for the behaviour layer."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def find_configs_dir() -> Path | None:
    """$X30_CONFIG_DIR or $X30_CONFIGS, else the nearest `configs/` containing skills.yaml above the cwd or this file."""
    for name in ("X30_CONFIG_DIR", "X30_CONFIGS"):
        env = os.environ.get(name)
        if env and Path(env).is_dir():
            return Path(env)
    for base in (Path.cwd(), Path(__file__).resolve()):
        for cand in [base, *base.parents]:
            if (cand / "configs" / "skills.yaml").is_file():
                return cand / "configs"
    return None


def load_yaml(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mode_config(mode: str, profile: str, configs_dir: Path | None = None) -> dict:
    """configs/modes/<mode>.yaml: defaults at top level, optional `sim:` / `real:` sections override them."""
    d = configs_dir or find_configs_dir()
    if d is None:
        return {}
    p = d / "modes" / f"{mode}.yaml"
    if not p.is_file():
        return {}
    data = load_yaml(p) or {}
    out = {k: v for k, v in data.items() if k not in ("sim", "real")}
    out.update(data.get(profile, {}) or {})
    return out
