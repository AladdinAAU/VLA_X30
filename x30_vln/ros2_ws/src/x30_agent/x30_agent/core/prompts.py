"""Prompt assembly and the token budget for an ~8-16k context model server.

Layout of every request (stable prefix first so llama.cpp can reuse its KV cache):
  system   : role + method + rules (object_nav_system.md) + few-shot transcripts (object_nav_examples.yaml)
  history  : the last N (assistant tool call, tool result) pairs
  user     : this turn's state (task, pose, map summary, harness notes, optional images)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from x30_common.images import estimate_image_tokens
from x30_common.toolschema import tool_name


def render_examples(path: str | Path, ids: list[str]) -> str:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    by_id = {e["id"]: e for e in data.get("examples", [])}
    out = []
    for i, eid in enumerate(ids, 1):
        e = by_id[eid]
        out.append(f"Example {i} (invented map, shows the format only). Task: {e['task']}")
        for s in e["steps"]:
            if s.get("map"):
                out.append("  map: " + s["map"].replace("\n", " | "))
            c = s["call"]
            args = ", ".join(f"{k}={json.dumps(v)}" for k, v in c["args"].items())
            out.append(f"  -> {c['tool']}({args})")
            out.append(f"     result: {s['result']}")
    return "\n".join(out)


def build_system_prompt(policy) -> str:
    base = policy.text("system").strip()
    ids = list(policy.examples)
    if ids and "examples" in policy.prompt_files.as_dict():
        base += "\n\n" + render_examples(policy.files["examples"], ids)
    return base


def example_calls(path: str | Path) -> list[tuple[str, str, dict]]:
    """(example id, tool, args) for every call in the examples file (tests replay them against the schemas)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [(e["id"], s["call"]["tool"], s["call"]["args"]) for e in data.get("examples", []) for s in e["steps"]]


class TokenEstimator:
    """Rough token counts, recalibrated from the server's reported prompt_tokens (an EMA of actual/estimated)."""

    def __init__(self, chars_per_token: float = 3.4):
        self.cpt = chars_per_token
        self.scale = 1.0

    def text(self, s: str) -> int:
        return math.ceil(len(s) / self.cpt)

    def message(self, m: dict, image_tokens: int = 0) -> int:
        c = m.get("content")
        n = 6
        if isinstance(c, str):
            n += self.text(c)
        elif isinstance(c, list):
            for p in c:
                if p.get("type") == "text":
                    n += self.text(p.get("text", ""))
                else:
                    n += image_tokens
        for tc in m.get("tool_calls") or []:
            n += self.text(json.dumps(tc.get("function", {})))
        return n

    def messages(self, messages: list[dict], tools: list[dict], image_tokens: int = 0) -> int:
        raw = sum(self.message(m, image_tokens) for m in messages) + self.text(json.dumps(tools, separators=(",", ":"))) // 1
        return int(raw * self.scale)

    def calibrate(self, estimated: int, actual: int) -> None:
        if estimated > 0 and actual > 0:
            self.scale = min(1.8, max(0.5, 0.6 * self.scale + 0.4 * (actual / (estimated / self.scale))))


def image_tokens_for(max_side: int, aspect: float = 1.0) -> int:
    w = max_side
    h = int(max_side / aspect) if aspect >= 1 else max_side
    return estimate_image_tokens(w, h)


def tools_chars(tools: list[dict]) -> int:
    return len(json.dumps(tools, separators=(",", ":")))


def tool_names(tools: list[dict]) -> list[str]:
    return [tool_name(t) for t in tools]
