"""Per-turn trace files (``traces/<episode>.jsonl``, one JSON object per line) and statistics derived from them.

The trace schema is owned by the agent package; this reader accepts the fields documented in docs/EVALUATION.md
section 'Trace format' and a few common aliases so it keeps working when the writer changes slightly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_trace(path: str | Path) -> list[dict]:
    """Read a JSONL trace; unreadable lines are skipped (a crashed episode may leave a truncated last line)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _latency_s(rec: dict) -> float | None:
    for k, scale in (("latency_s", 1.0), ("latency", 1.0), ("latency_ms", 1e-3), ("vlm_latency_s", 1.0)):
        if rec.get(k) is not None:
            return float(rec[k]) * scale
    return None


def _tokens(rec: dict) -> tuple[int, int]:
    u = rec.get("usage") or {}
    pt = rec.get("prompt_tokens", u.get("prompt_tokens", 0)) or 0
    ct = rec.get("completion_tokens", u.get("completion_tokens", 0)) or 0
    return int(pt), int(ct)


def _is_turn(rec: dict) -> bool:
    return rec.get("kind") in ("turn", "reasoner_turn", "llm_turn")


def trace_stats(records: list[dict]) -> dict:
    """Agent turns, tokens, VLM latency, tool usage, parse errors and loop indicators.

    One reasoner turn is one ``kind: turn`` row (rows of re-asked replies share the turn number)."""
    turns = [r for r in records if _is_turn(r)]
    lat = np.array([x for x in (_latency_s(r) for r in turns) if x is not None], float)
    ptok = ctok = 0
    for r in turns:
        a, b = _tokens(r)
        ptok += a
        ctok += b
    calls = [r for r in turns if r.get("tool")]
    sig = [(r["tool"], json.dumps(r.get("args", {}), sort_keys=True, default=str)) for r in calls]
    run = best = 0
    prev = None
    for s_ in sig:
        run = run + 1 if s_ == prev else 1
        best = max(best, run)
        prev = s_
    counts: dict[str, int] = {}
    for r in calls:
        counts[r["tool"]] = counts.get(r["tool"], 0) + 1
    return {
        "turns": len({r.get("turn", i) for i, r in enumerate(turns)}),
        "vlm_calls": int(len(lat)),
        "prompt_tokens": ptok, "completion_tokens": ctok,
        "vlm_latency_mean_s": float(lat.mean()) if len(lat) else None,
        "vlm_latency_p95_s": float(np.percentile(lat, 95)) if len(lat) else None,
        "vlm_latency_total_s": float(lat.sum()) if len(lat) else 0.0,
        "parse_errors": sum(1 for r in turns if r.get("parse_ok") is False) + sum(1 for r in records if r.get("kind") == "parse_error"),
        "refused_calls": sum(1 for r in calls if r.get("refused")),
        "tool_counts": counts,
        "max_identical_tool_repeat": best,
        "skill_failures": sum(1 for r in calls if r.get("success") is False),
        "finish_refused": sum(1 for r in records if r.get("kind") == "finish_guard" and r.get("allowed") is False),
        "last_tool": calls[-1]["tool"] if calls else None,
    }
