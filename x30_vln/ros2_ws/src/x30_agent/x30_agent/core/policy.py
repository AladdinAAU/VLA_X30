"""Agent policy files (configs/agent/*.yaml): loading, override merging and prompt hashing."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from x30_common.config import find_config_dir


class Section:
    """Read-only attribute view of a dict section. Unknown names raise, so a typo in code or in an override is loud."""

    def __init__(self, data: dict, path: str = ""):
        object.__setattr__(self, "_d", data)
        object.__setattr__(self, "_p", path)

    def __getattr__(self, name: str) -> Any:
        d = object.__getattribute__(self, "_d")
        if name not in d:
            raise AttributeError(f"policy has no key '{object.__getattribute__(self, '_p')}{name}'")
        v = d[name]
        return Section(v, f"{object.__getattribute__(self, '_p')}{name}.") if isinstance(v, dict) and name not in ("hints", "tool_descriptions") else v

    def get(self, name: str, default: Any = None) -> Any:
        d = object.__getattribute__(self, "_d")
        return getattr(self, name) if name in d else default

    def as_dict(self) -> dict:
        return copy.deepcopy(object.__getattribute__(self, "_d"))


def deep_merge(base: dict, over: dict, path: str = "") -> dict:
    """Merge `over` into a copy of `base`; keys that do not exist in `base` are an error (typo protection)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise KeyError(f"unknown policy key '{path}{k}'")
        if isinstance(out[k], dict) and isinstance(v, dict) and k not in ("hints", "tool_descriptions"):
            out[k] = deep_merge(out[k], v, f"{path}{k}.")
        elif isinstance(out[k], dict) and isinstance(v, dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def agent_config_dir(explicit: str | os.PathLike | None = None) -> Path:
    if explicit:
        return Path(explicit)
    d = find_config_dir()
    if d is None:
        raise FileNotFoundError("configs/ directory not found; set X30_CONFIG_DIR or pass config_dir")
    return d / "agent"


class Policy(Section):
    def __init__(self, data: dict, files: dict[str, Path], hash_: str, name: str):
        super().__init__(data)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "hash", hash_)
        object.__setattr__(self, "name", name)

    def text(self, key: str) -> str:
        """Content of a prompt file named in `prompt_files`."""
        return Path(object.__getattribute__(self, "files")[key]).read_text(encoding="utf-8")


def load_policy(name: str, config_dir: str | os.PathLike | None = None, overrides: dict | None = None) -> Policy:
    """Load configs/agent/<name>.yaml (+ overrides). `hash` covers the yaml, every prompt file it names, and the overrides."""
    d = agent_config_dir(config_dir)
    path = d / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    data = deep_merge(base, overrides or {})
    files = {k: d / v for k, v in (data.get("prompt_files") or {}).items()}
    h = hashlib.sha256()
    h.update(path.read_bytes())
    for k in sorted(files):
        h.update(k.encode())
        try:
            h.update(files[k].read_bytes())
        except OSError:
            h.update(b"<missing>")
    h.update(repr(sorted((overrides or {}).items(), key=lambda kv: kv[0])).encode())
    return Policy(data, files, h.hexdigest()[:12], name)
