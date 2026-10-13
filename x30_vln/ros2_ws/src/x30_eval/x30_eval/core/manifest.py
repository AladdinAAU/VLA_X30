"""Run manifest: everything needed to know what was evaluated (code, configs, prompts, models, containers, seeds)."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from .. import __version__
from .episodes import file_sha256


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(root: Path) -> dict | None:
    if not shutil.which("git") or _run(["git", "rev-parse", "--is-inside-work-tree"], root) != "true":
        return None
    status = _run(["git", "status", "--porcelain"], root) or ""
    return {"commit": _run(["git", "rev-parse", "HEAD"], root), "dirty": bool(status), "n_changed": len(status.splitlines())}


def glob_files(root: Path, patterns: list[str]) -> list[Path]:
    out: set[Path] = set()
    for pat in patterns:
        out.update(p for p in root.glob(pat) if p.is_file())
    return sorted(out)


def hash_files(root: Path, patterns: list[str]) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): file_sha256(p)[:16] for p in glob_files(root, patterns)}


def tree_hash(file_hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(file_hashes.items())).encode()).hexdigest()[:16]


def docker_image_ids(containers: list[str], images: list[str]) -> dict:
    if not shutil.which("docker"):
        return {"available": False}
    out: dict = {"available": True, "containers": {}, "images": {}}
    for c in containers:
        out["containers"][c] = _run(["docker", "inspect", "--format", "{{.Image}} {{.Config.Image}}", c])
    for i in images:
        out["images"][i] = _run(["docker", "image", "inspect", "--format", "{{.Id}}", i])
    return out


def models_summary(root: Path, rel: str) -> dict | None:
    """Model names/endpoints from configs/models.yaml with anything that looks like a secret removed."""
    p = root / rel
    if not p.exists():
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    def scrub(x):
        if isinstance(x, dict):
            return {k: ("<redacted>" if any(s in k.lower() for s in ("key", "token", "secret", "password")) else scrub(v))
                    for k, v in x.items()}
        return x

    return scrub(data)


def build_manifest(root: Path, cfg: dict, run_name: str, args: dict, episodes_file: str | Path, episode_ids: list[str],
                   overlay: dict | None = None, extra: dict | None = None) -> dict:
    hashes = hash_files(root, cfg["manifest"]["hash_globs"])
    prompt_hashes = {k: v for k, v in hashes.items() if "prompt" in k.lower()}
    src_hashes = hash_files(root, ["ros2_ws/src/**/*.py", "ros2_ws/src/**/*.launch.py"])
    man = {
        "run_name": run_name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "x30_eval_version": __version__, "python": platform.python_version(), "host": platform.node(),
        "git": git_state(root),
        "config_hash": tree_hash(hashes), "prompts_hash": tree_hash(prompt_hashes), "source_hash": tree_hash(src_hashes),
        "config_files": hashes, "n_source_files": len(src_hashes),
        "eval_config": cfg, "overlay": overlay or {}, "overlay_hash": hashlib.sha256(
            json.dumps(overlay or {}, sort_keys=True).encode()).hexdigest()[:16],
        "models": models_summary(root, cfg["manifest"]["models_config"]),
        "episodes_file": str(episodes_file), "episodes_sha256": file_sha256(episodes_file)[:16],
        "episode_ids": episode_ids, "args": args,
        "docker": docker_image_ids(cfg["manifest"]["docker_containers"], cfg["manifest"]["docker_images"]),
        "worlds": [],
    }
    man.update(extra or {})
    return man


def comparable(old: dict, new: dict) -> list[str]:
    """Differences between two manifests that make a resumed run non-comparable (empty = safe to resume)."""
    diffs = []
    for k in ("config_hash", "prompts_hash", "episodes_sha256", "overlay_hash"):
        if old.get(k) != new.get(k):
            diffs.append(f"{k}: {old.get(k)} -> {new.get(k)}")
    for k in ("repeats", "split", "seed", "force_mode"):
        if old.get("args", {}).get(k) != new.get("args", {}).get(k):
            diffs.append(f"args.{k}: {old.get('args', {}).get(k)} -> {new.get('args', {}).get(k)}")
    return diffs

