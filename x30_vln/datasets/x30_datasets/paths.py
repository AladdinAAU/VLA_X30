"""Locations of the datasets folder, the x30_worlds data package and the registry.

Everything is derived from this file's location so nothing is hard-coded.
Environment overrides: ``X30_DATASETS_DIR``, ``X30_WORLDS_DIR``.
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve()


def datasets_dir() -> Path:
    """``x30_vln/datasets`` (holds registry.yaml, episodes/, external/)."""
    env = os.environ.get("X30_DATASETS_DIR")
    return Path(env) if env else _HERE.parents[1]


def project_root() -> Path:
    return datasets_dir().parent


def worlds_pkg_dir() -> Path:
    """Source directory of the ``x30_worlds`` ament package (worlds/, models/, config/)."""
    env = os.environ.get("X30_WORLDS_DIR")
    return Path(env) if env else project_root() / "ros2_ws" / "src" / "x30_worlds"


def registry_path() -> Path:
    return datasets_dir() / "registry.yaml"


def episodes_dir() -> Path:
    return datasets_dir() / "episodes"


def external_dir() -> Path:
    return datasets_dir() / "external"


def labels_path() -> Path:
    return worlds_pkg_dir() / "config" / "labels.yaml"


def episode_schema_path() -> Path:
    return _HERE.parent / "schemas" / "episode.schema.json"


def world_dir(world_id: str) -> Path:
    return worlds_pkg_dir() / "worlds" / world_id
