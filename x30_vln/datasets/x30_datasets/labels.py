"""Semantic label table (master file: x30_worlds/config/labels.yaml)."""
from __future__ import annotations

from functools import lru_cache

import yaml

from .paths import labels_path


@lru_cache(maxsize=1)
def _table() -> dict:
    with open(labels_path(), encoding="utf-8") as f:
        return yaml.safe_load(f)


def classes() -> dict[str, dict]:
    return _table()["classes"]


def label_id(label: str) -> int:
    return int(classes().get(label, classes()["other"])["id"])


def synonyms(label: str) -> list[str]:
    entry = classes().get(label)
    return list(entry["synonyms"]) if entry else [label.replace("_", " ")]


def spoken(label: str) -> str:
    """Primary human-readable phrase for a label ("fire_extinguisher" -> "fire extinguisher")."""
    return synonyms(label)[0]


def is_nav_target(label: str) -> bool:
    entry = classes().get(label)
    return bool(entry and entry.get("nav_target", False))


def canonical(name: str) -> str:
    """Map an external category name (HM3D / MP3D / Habitat) to a canonical label."""
    key = name.strip().lower().replace("_", " ")
    table = _table()
    alias = table.get("external_aliases", {})
    if key in alias:
        return alias[key]
    if key.replace(" ", "_") in table["classes"]:
        return key.replace(" ", "_")
    for label, entry in table["classes"].items():
        if key in entry.get("synonyms", []):
            return label
    return "other"
