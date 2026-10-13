"""Score one episode: ground truth recording + task result -> flat-ish result dict (one JSON line)."""
from __future__ import annotations

import math

import numpy as np

from . import metrics as M
from .config import success_radius
from .episodes import Episode
from .failure import classify
from .geometry import SE2
from .grid import OccGrid
from .optimum import OptimumProvider
from .recording import Recording, TaskResult
from .traces import trace_stats


def _round(x, n=4):
    return None if x is None else round(float(x), n)


def _at_time(gt: np.ndarray, t_end: float) -> int:
    """Index of the last GT sample not later than ``t_end`` (+ tolerance); last sample if none qualifies."""
    idx = np.nonzero(gt[:, 0] <= t_end + 0.25)[0]
    return int(idx[-1]) if len(idx) else len(gt) - 1


def score_episode(ep: Episode, rec: Recording, task: TaskResult, cfg: dict, optimum: OptimumProvider | None = None,
                  grid: OccGrid | None = None, trace: list[dict] | None = None) -> dict:
    """Compute every metric of docs/EVALUATION.md for one finished attempt."""
    sc = cfg["success"]
    syn = sc.get("synonyms")
    goal = ep.goal
    mode = M.goal_distance_mode(goal, sc["object_distance_mode"])
    radius = success_radius(cfg, goal)
    res: dict = {
        "episode_id": ep.id, "scene": ep.scene, "mode": ep.mode, "split": ep.split, "goal_type": ep.goal_type,
        "goal_label": ep.target_label or None, "radius_m": radius, "time_limit_s": ep.time_limit_s, "stop_reason": task.stop_reason,
        "message": task.message[:300], "agent_claimed_success": bool(task.success), "stop_issued": task.stop_issued,
        "agent_turns": int(task.agent_turns), "instruction": ep.instruction,
    }
    gt = rec.gt_array()
    if len(gt) < 2:
        res.update(success=False, oracle_success=False, eval_error="fewer than 2 ground-truth pose samples recorded")
        res["failure_category"], res["failure_evidence"] = classify(res, cfg)
        return res

    gt_objects = rec.gt_objects
    try:
        dist = M.distance_series(gt[:, 1:3], goal, gt_objects, syn, mode)
    except ValueError as exc:
        res.update(success=False, oracle_success=False, eval_error=str(exc))
        res["failure_category"], res["failure_evidence"] = classify(res, cfg)
        return res

    i_end = _at_time(gt, rec.t_end)
    xy = gt[: i_end + 1, 1:3]
    final_d, min_d = float(dist[i_end]), float(dist[: i_end + 1].min())
    within = np.nonzero(dist[: i_end + 1] <= radius)[0]
    oracle = bool(len(within))
    stop_ok = (task.stop_issued or not sc["require_stop"]) and (task.success or not sc["require_agent_claim"])
    success = bool(final_d <= radius and stop_ok)
    path_m = M.path_length_deadband(xy)

    res.update(
        success=success, oracle_success=oracle, final_distance_m=_round(final_d), min_distance_m=_round(min_d),
        time_to_oracle_s=_round(gt[within[0], 0], 2) if oracle else None, path_length_m=_round(path_m),
        duration_s=_round(rec.t_end, 2), final_xy=[_round(v) for v in xy[-1]],
    )
    inst_d = M.instance_distance_series(gt[: i_end + 1, 1:3], goal, gt_objects, mode)
    if inst_d is not None:
        res["success_instance"] = bool(inst_d[-1] <= radius and stop_ok)
        res["final_distance_instance_m"] = _round(inst_d[-1])
    near = [o for o in gt_objects if M.object_distance(xy[-1], o, mode) <= radius]
    near.sort(key=lambda o: M.object_distance(xy[-1], o, mode))
    res["final_nearest_gt_label"] = near[0]["label"] if near else None

    # shortest path / SPL
    opt, basis = (optimum.optimum(ep, gt_objects) if optimum else (None, "unavailable"))
    res.update(optimal_path_m=_round(opt), spl_basis=basis, spl=_round(M.spl(success, opt, path_m)))
    if opt is not None and opt <= 1e-6:
        res["trivial_start"] = True
    # reference-path metrics
    if ep.reference_path:
        nd = M.ndtw(xy, np.asarray(ep.reference_path, float), cfg["ndtw"]["d_th_m"], cfg["ndtw"]["spacing_m"])
        res.update(ndtw=_round(nd), sdtw=_round(M.sdtw(success, nd)))

    # collisions, safety, interventions
    csrc = cfg["collisions"]["source"]
    ref_s = cfg["collisions"]["refractory_s"]
    coll, basis_c = None, "none"
    if csrc in ("auto", "contacts") and rec.contacts_available:
        coll, basis_c = len(M.merge_events([t for t in rec.contacts if t <= rec.t_end + 0.25], ref_s)), "contacts"
    elif csrc in ("auto", "grid") and grid is not None:
        coll = len(M.grid_collision_times(gt[: i_end + 1], grid, cfg["robot"]["footprint_length_m"],
                                          cfg["robot"]["footprint_width_m"], ref_s))
        basis_c = "grid"
    res.update(collisions=coll, collision_basis=basis_c)
    stops, prev = 0, False
    for s in rec.safety:
        if s.get("stopped") and not prev:
            stops += 1
        prev = bool(s.get("stopped"))
    res.update(safety_stops=stops, interventions=len(rec.interventions))

    # localisation
    align = None
    for key, pairs in (("ate", rec.pairs_array()), ("odom_ate", rec.odom_pairs_array())):
        out = {m: M.ate(pairs, m, cfg["ate"]["min_pairs"]) for m in ("yaw", "se2")}
        if out["yaw"] is not None:
            head = out[cfg["ate"]["method"]]
            res[key] = {k: _round(v) for k, v in head.items() if isinstance(v, float)}
            res[key]["n_pairs"] = head["n_pairs"]
            res[key]["se2_fit_rmse_m"] = _round(out["se2"]["ate_rmse_m"])
            if key == "ate":
                a = out["yaw"]["align"]
                align = SE2(a["x"], a["y"], a["yaw"])
    if align is None and rec.map_to_world:
        align = SE2(**rec.map_to_world)
        res["map_alignment"] = "start_tf"
    elif align is not None:
        res["map_alignment"] = "trajectory_yaw"

    # semantic map
    if rec.map_objects:
        smc = cfg["semantic_map"]
        sm = M.semantic_map_metrics(rec.map_objects, gt_objects, align, xy, smc["match_radius_m"], smc["cover_radius_m"],
                                    syn, smc["vocab"], smc["ignore_labels"])
        res["map"] = {k: (_round(v) if isinstance(v, float) else v) for k, v in sm.items()}
        if goal["type"] == "object":
            inst = M.goal_instances(goal, gt_objects, syn)
            mp = np.array([[o["x"], o["y"]] for o in rec.map_objects if not o.get("bearing_only")]).reshape(-1, 2)
            pts = align.apply(mp) if align is not None else mp
            labs = [o["label"] for o in rec.map_objects if not o.get("bearing_only")]
            found = any(M.label_match(labs[k], goal["label"], syn) and any(
                math.hypot(pts[k][0] - g["x"], pts[k][1] - g["y"]) <= cfg["failure"]["never_found_map_radius_m"] for g in inst)
                for k in range(len(labs)))
            res["target_in_map"] = found

    # trace statistics
    if trace is not None:
        ts = trace_stats(trace)
        res["trace"] = ts
        res["agent_turns"] = max(res["agent_turns"], ts["turns"])
    res["failure_category"], res["failure_evidence"] = classify(res, cfg)
    if rec.notes:
        res["notes"] = rec.notes
    return res
