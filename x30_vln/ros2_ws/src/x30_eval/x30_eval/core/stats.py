"""Aggregation over repeats, bootstrap confidence intervals and paired comparisons (numpy only).

Unit of resampling is the EPISODE: repeats of the same episode are averaged first, then episodes are resampled.
Treating every repeat as independent would make the intervals too narrow (repeats share the scene and the goal).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np

# headline metrics: (key in the flat record, label, higher_is_better)
METRICS: list[tuple[str, str, bool | None]] = [
    ("success", "SR", True), ("oracle_success", "OSR", True), ("spl", "SPL", True),
    ("final_distance_m", "DTG [m]", False), ("path_length_m", "path [m]", None), ("duration_s", "time [s]", False),
    ("collisions", "collisions", False), ("safety_stops", "guard stops", False), ("interventions", "interventions", False),
    ("agent_turns", "agent turns", None), ("ndtw", "nDTW", True), ("sdtw", "SDTW", True),
    ("ate_rmse_m", "ATE rmse [m]", False), ("map_recall_covered", "map recall (covered)", True),
    ("map_duplicate_rate", "map duplicates", False), ("map_pos_err_mean_m", "map pos err [m]", False),
    ("prompt_tokens", "prompt tokens", None), ("completion_tokens", "completion tokens", None),
    ("vlm_latency_mean_s", "VLM latency [s]", False), ("parse_errors", "parse errors", False),
]


def flatten(res: dict) -> dict:
    """Flat view of a result record: nested ate/map/trace entries get prefixes (ate_rmse_m, map_recall, prompt_tokens ...)."""
    out = {k: v for k, v in res.items() if not isinstance(v, dict)}
    for k, v in (res.get("ate") or {}).items():
        out[k if k.startswith("ate_") else f"ate_{k}"] = v
    for k, v in (res.get("odom_ate") or {}).items():
        out[f"odom_{k if k.startswith('ate_') else 'ate_' + k}"] = v
    for k, v in (res.get("map") or {}).items():
        out[f"map_{k}"] = v
    tr = res.get("trace") or {}
    for k, v in tr.items():
        if not isinstance(v, dict):
            out[k] = v
    for k in ("success", "oracle_success", "success_instance"):
        if k in out and out[k] is not None:
            out[k] = float(bool(out[k]))
    return out


def per_episode_means(records: Iterable[dict], key: str) -> dict[str, float]:
    """{episode_id: mean over repeats} ignoring missing/None/NaN values."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in records:
        v = r.get(key)
        if v is not None and isinstance(v, (int, float, np.floating)) and math.isfinite(float(v)):
            acc[r["episode_id"]].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap of the mean; (nan, nan, nan) for empty input."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return (math.nan,) * 3
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def repeat_std(records: Sequence[dict], key: str) -> float | None:
    """Std (ddof=1) over repeats of the per-repeat mean: the seed-to-seed variability of the whole suite."""
    by_rep: dict[int, list[float]] = defaultdict(list)
    for r in records:
        v = r.get(key)
        if v is not None and math.isfinite(float(v)):
            by_rep[int(r.get("repeat", 0))].append(float(v))
    means = [float(np.mean(v)) for v in by_rep.values()]
    return float(np.std(means, ddof=1)) if len(means) > 1 else None


def summarise(records: Sequence[dict], key: str, n_boot: int = 4000, seed: int = 0) -> dict:
    ep = per_episode_means(records, key)
    mean, lo, hi = bootstrap_mean_ci(list(ep.values()), n_boot, seed=seed)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_episodes": len(ep),
            "n_values": sum(1 for r in records if r.get(key) is not None), "std_over_repeats": repeat_std(records, key)}


def summarise_all(records: Sequence[dict], n_boot: int = 4000, seed: int = 0) -> dict[str, dict]:
    return {key: summarise(records, key, n_boot, seed) for key, _, _ in METRICS
            if any(r.get(key) is not None for r in records)}


def group_by(records: Sequence[dict], key_fn: Callable[[dict], str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[str(key_fn(r))].append(r)
    return dict(sorted(out.items()))


def paired_bootstrap(a: Sequence[dict], b: Sequence[dict], key: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of run B against run A on the episodes both ran (mean of B - A).

    Returns n_pairs, mean_a, mean_b, diff, CI of the diff and a two-sided bootstrap p-value (probability mass of the
    resampled differences on the other side of zero, doubled). Few episodes give wide intervals by design."""
    ma, mb = per_episode_means(a, key), per_episode_means(b, key)
    common = sorted(set(ma) & set(mb))
    if not common:
        return {"n_pairs": 0}
    da = np.array([ma[e] for e in common])
    db = np.array([mb[e] for e in common])
    d = db - da
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    p = 2.0 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {"n_pairs": len(common), "mean_a": float(da.mean()), "mean_b": float(db.mean()), "diff": float(d.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "p_value": min(p, 1.0), "episodes_b_better": int((d > 0).sum()), "episodes_a_better": int((d < 0).sum())}
