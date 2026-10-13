"""Run summary (summary.json): headline metrics with bootstrap CIs, breakdowns, failure categories."""
from __future__ import annotations

from collections import Counter, defaultdict

from . import stats as S


def derive_difficulty(optimal_m: float | None, cfg: dict) -> str:
    if optimal_m is None:
        return "unknown"
    d = cfg["difficulty"]
    return "easy" if optimal_m <= d["easy_max_m"] else "medium" if optimal_m <= d["medium_max_m"] else "hard"


def scored(results: list[dict]) -> list[dict]:
    """Records that count toward the metrics: final ('done') episodes, flattened. Infrastructure failures are excluded."""
    return [S.flatten(r) for r in results if r.get("status", "done") == "done" and not r.get("infrastructure_failure")]


def failure_table(flat: list[dict], examples: int = 3) -> dict:
    counts = Counter(r.get("failure_category", "other") for r in flat)
    ex: dict[str, list[dict]] = defaultdict(list)
    for r in flat:
        cat = r.get("failure_category", "other")
        if cat != "success" and len(ex[cat]) < examples:
            ex[cat].append({"episode_id": r["episode_id"], "repeat": r.get("repeat"),
                            "evidence": r.get("failure_evidence")})
    return {"counts": dict(counts), "examples": dict(ex)}


def breakdown(flat: list[dict], key: str, n_boot: int = 2000) -> dict:
    out = {}
    for name, recs in S.group_by(flat, lambda r: r.get(key) or "unknown").items():
        sm = S.summarise_all(recs, n_boot)
        out[name] = {"n_episodes": len({r["episode_id"] for r in recs}), "n_runs": len(recs),
                     **{k: sm[k] for k in ("success", "oracle_success", "spl", "final_distance_m", "duration_s") if k in sm}}
    return out


def build_summary(results: list[dict], infra: list[dict], manifest: dict | None, planned: int, n_boot: int = 4000) -> dict:
    flat = scored(results)
    exhausted = [r for r in results if r.get("status") == "infra_failed"]
    out = {
        "run_name": (manifest or {}).get("run_name"),
        "planned_runs": planned, "scored_runs": len(flat), "infra_failed_runs": len(exhausted),
        "infra_incidents": len(infra),
        "n_episodes": len({r["episode_id"] for r in flat}),
        "repeats": len({r.get("repeat") for r in flat}),
        "split": (manifest or {}).get("args", {}).get("split"),
        "config_hash": (manifest or {}).get("config_hash"),
        "metrics": S.summarise_all(flat, n_boot),
        "by_scene": breakdown(flat, "scene"), "by_difficulty": breakdown(flat, "difficulty"),
        "by_mode": breakdown(flat, "mode"), "by_goal_type": breakdown(flat, "goal_type"),
        "failures": failure_table(flat),
        "spl_basis": dict(Counter(r.get("spl_basis") for r in flat)),
        "complete": len(flat) + len(exhausted) >= planned,
    }
    if exhausted:
        out["warning"] = (f"{len(exhausted)} run(s) failed for infrastructure reasons after all retries and are "
                          f"EXCLUDED from the metrics; the numbers describe fewer episodes than planned")
    return out
