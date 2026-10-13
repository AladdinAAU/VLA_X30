"""Harness-held memory. The model's own bookkeeping is unreliable (the drone project saw it re-check rejected objects
and repeat one call 19 times), so facts that matter are kept here and injected into every prompt and every skill call."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .types import Pose2D


def call_key(name: str, args: dict) -> str:
    return name + json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))


def fmt_args(args: dict, limit: int = 60) -> str:
    s = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}" for k, v in args.items())
    return s if len(s) <= limit else s[: limit - 1] + "…"


@dataclass
class VerifyRecord:
    object_id: int | None
    answer: str
    plausibility: float
    turn: int
    pose: Pose2D | None
    distance_m: float | None = None
    label: str = ""

    def positive(self, accept_unsure_plausibility: float = 0.0) -> bool:
        if self.answer == "yes":
            return True
        return bool(accept_unsure_plausibility > 0 and self.answer == "unsure" and self.plausibility >= accept_unsure_plausibility)


@dataclass
class HarnessMemory:
    rejected: set = field(default_factory=set)
    last_object: int | None = None
    verifies: list = field(default_factory=list)
    scans: list = field(default_factory=list)            # Pose2D where a scan completed
    visited: list = field(default_factory=list)          # (Pose2D, skill)
    fail_counts: dict = field(default_factory=dict)      # call_key -> failures
    failed_notes: list = field(default_factory=list)     # short strings for the prompt
    action_log: list = field(default_factory=list)       # one line per executed or refused call
    candidates: dict = field(default_factory=dict)       # id -> label seen in search results
    recent_calls: list = field(default_factory=list)     # call_key of the last calls (executed or refused)
    consecutive_failures: int = 0
    refusals: int = 0
    finish_refusals: int = 0

    # ---------------------------------------------------------------- queries
    def latest_positive_verify(self, accept_unsure_plausibility: float = 0.0) -> VerifyRecord | None:
        for v in reversed(self.verifies):
            if v.positive(accept_unsure_plausibility):
                return v
        return None

    def near_previous_scan(self, pose: Pose2D | None, min_sep: float) -> float | None:
        """Distance to the closest earlier scan if it is closer than min_sep, else None."""
        if pose is None:
            return None
        ds = [pose.dist(p) for p in self.scans]
        return min(ds) if ds and min(ds) < min_sep else None

    def repeat_run(self, key: str) -> int:
        """How many times in a row `key` was already the latest call."""
        n = 0
        for k in reversed(self.recent_calls):
            if k != key:
                break
            n += 1
        return n

    def alternating(self, key: str) -> bool:
        """True if appending `key` would continue x,y,x,y,x (period-2 cycle over the last 4 calls)."""
        r = self.recent_calls
        return len(r) >= 4 and r[-2] == key and r[-4] == key and r[-1] == r[-3] and r[-1] != key

    # ---------------------------------------------------------------- prompt text
    def notes(self, max_log: int = 10) -> str:
        lines = []
        if self.rejected:
            lines.append(f"Rejected by verify (never use these ids): {', '.join('#' + str(i) for i in sorted(self.rejected))}")
        if self.verifies:
            v = self.verifies[-1]
            where = f"#{v.object_id}" if v.object_id is not None else "current view"
            dist = f", {v.distance_m:.1f} m away then" if v.distance_m is not None else ""
            lines.append(f"Last verify: {where} -> {v.answer}{dist}")
        if self.scans:
            lines.append("Scanned at: " + "; ".join(f"({p.x:.1f},{p.y:.1f})" for p in self.scans[-4:]))
        if self.failed_notes:
            lines.append("Failed: " + "; ".join(self.failed_notes[-4:]))
        if self.action_log:
            lines.append("Recent actions:")
            lines += [f"  {s}" for s in self.action_log[-max_log:]]
        return "\n".join(lines)
