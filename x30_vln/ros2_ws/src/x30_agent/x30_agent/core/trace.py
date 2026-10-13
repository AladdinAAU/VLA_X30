"""Per-episode trace: results/<run>/traces/<episode>.jsonl plus the images each turn used, saved as files.

One JSON object per line. `kind` is one of: header, turn, finish_guard, event, summary. The x30_eval trace reader
(x30_eval.core.traces) understands `kind=turn` rows with `turn, tool, args, latency_s, usage/prompt_tokens, parse_ok`.
Writing never raises: a full disk must not kill a navigation episode (the failure is counted and logged once).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80] or "episode"


class TraceWriter:
    def __init__(self, directory: str | Path | None, episode_id: str, enabled: bool = True):
        self.enabled = bool(directory) and enabled
        self.episode = safe_name(episode_id)
        self.errors = 0
        self.t0 = time.time()
        self.path: Path | None = None
        self.image_dir: Path | None = None
        self._f = None
        if self.enabled:
            try:
                d = Path(directory)
                d.mkdir(parents=True, exist_ok=True)
                self.path = d / f"{self.episode}.jsonl"
                self.image_dir = d / f"{self.episode}_img"
                self._f = open(self.path, "w", encoding="utf-8")
            except OSError:
                self.enabled = False
                self.errors += 1

    def write(self, kind: str, **fields: Any) -> None:
        if not self.enabled or self._f is None:
            return
        row = {"kind": kind, "t": round(time.time() - self.t0, 3), **fields}
        try:
            self._f.write(json.dumps(row, default=_default, ensure_ascii=False) + "\n")
            self._f.flush()
        except (OSError, ValueError, TypeError):
            self.errors += 1

    def save_image(self, name: str, data: bytes) -> str:
        """Store JPEG bytes next to the trace; returns a path relative to the trace directory ('' if not stored)."""
        if not self.enabled or self.image_dir is None:
            return ""
        try:
            self.image_dir.mkdir(parents=True, exist_ok=True)
            p = self.image_dir / f"{safe_name(name)}.jpg"
            p.write_bytes(data)
            return f"{self.image_dir.name}/{p.name}"
        except OSError:
            self.errors += 1
            return ""

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
            self._f = None
