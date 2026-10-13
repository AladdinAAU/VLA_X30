"""Heuristic failure categorisation of a scored episode. One category per failed episode, with the evidence used.

Order matters (first matching rule wins); the rules and thresholds are documented in docs/EVALUATION.md and the
thresholds live in configs/eval/eval.yaml (``failure``). This is a triage aid for reading traces, not ground truth:
the report lists example episodes per category so the labels can be spot-checked.
"""
from __future__ import annotations

CATEGORIES = (
    "success", "infrastructure", "eval_error", "collision", "safety_stop", "agent_loop", "wrong_object",
    "not_stopped_at_goal", "never_found", "timeout", "agent_error", "other",
)


def classify(res: dict, cfg: dict) -> tuple[str, list[str]]:
    """Return (category, evidence list) for a scored episode record ``res``."""
    fc = cfg["failure"]
    if res.get("success"):
        return "success", []
    ev: list[str] = []
    if res.get("infrastructure_failure"):
        return "infrastructure", [str(res.get("infra_reason", ""))]
    if res.get("eval_error"):
        return "eval_error", [str(res["eval_error"])]
    coll = res.get("collisions")
    if coll is not None and (coll >= fc["collision_min"] or (coll >= 1 and res.get("stop_reason") == "aborted")):
        return "collision", [f"{coll} collisions"]
    stops = res.get("safety_stops", 0) or 0
    if res.get("stop_reason") == "safety" or stops >= fc["safety_stops_min"]:
        return "safety_stop", [f"stop_reason={res.get('stop_reason')}", f"{stops} guard stops"]
    if res.get("stop_issued"):
        d = res.get("final_distance_m")
        near = res.get("final_nearest_gt_label")
        ev = [f"stopped {d:.2f} m from the goal (radius {res.get('radius_m')})" if d is not None else "stopped outside radius"]
        if near:
            ev.append(f"nearest GT object at the stop: {near}")
        return "wrong_object", ev
    rep = (res.get("trace") or {}).get("max_identical_tool_repeat", 0) or 0
    if res.get("oracle_success") and not res.get("stop_issued") and res.get("stop_reason") in ("timeout", "evaluator_timeout"):
        return "not_stopped_at_goal", [f"was within the radius at t={res.get('time_to_oracle_s')}s but never stopped",
                                       f"max identical consecutive tool calls={rep}"]
    if rep >= fc["loop_repeat"] or res.get("stop_reason") == "no_progress":
        return "agent_loop", [f"max identical consecutive tool calls={rep}", f"stop_reason={res.get('stop_reason')}"]
    if res.get("stop_reason") in ("timeout", "evaluator_timeout"):
        if res.get("goal_type") == "object" and res.get("target_in_map") is False:
            return "never_found", ["no map object of the goal label near any GT instance at the end"]
        return "timeout", [f"min distance {res.get('min_distance_m')}"]
    if res.get("goal_type") == "object" and res.get("target_in_map") is False:
        return "never_found", [f"stop_reason={res.get('stop_reason')}"]
    if res.get("stop_reason") in ("error", "aborted"):
        return "agent_error", [str(res.get("message", ""))[:200]]
    return "other", [f"stop_reason={res.get('stop_reason')}"]
