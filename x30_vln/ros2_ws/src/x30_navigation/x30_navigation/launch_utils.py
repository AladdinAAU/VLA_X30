"""Helpers shared by the launch files: config directory resolution and parameter file lists."""
from __future__ import annotations

import os


def config_dir(override: str = "") -> str:
    """Directory with the navigation configs: explicit argument, else $X30_CONFIGS/navigation."""
    if override:
        return override
    root = os.environ.get("X30_CONFIGS", "")
    path = os.path.join(root, "navigation") if root else ""
    if not path or not os.path.isdir(path):
        raise RuntimeError("navigation configs not found: set X30_CONFIGS (…/configs) or pass config_dir:=<path to configs/navigation>")
    return path


def node_param_files(cfg: str, profile: str) -> list[str]:
    files = [os.path.join(cfg, "navigation.yaml")]
    prof = os.path.join(cfg, "profiles", f"{profile}.yaml")
    if os.path.isfile(prof):
        files.append(prof)
    return files
