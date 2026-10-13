import numpy as np
import pytest

from x30_eval.core.recording import Recording, TaskResult
from x30_eval.core.scoring import score_episode

OBJS = [{"id": 1, "name": "chair_1", "label": "chair", "x": 3.0, "y": 1.0, "sx": 0.5, "sy": 0.5},
        {"id": 2, "name": "chair_2", "label": "chair", "x": 9.0, "y": 6.0, "sx": 0.5, "sy": 0.5},
        {"id": 3, "name": "table_1", "label": "table", "x": 3.0, "y": 6.0, "sx": 1.2, "sy": 0.8}]


def straight(x0, y0, x1, y1, n=100, dur=40.0):
    t = np.linspace(0, dur, n)
    return [[float(a), float(x0 + (x1 - x0) * a / dur), float(y0 + (y1 - y0) * a / dur), 0.0] for a in t]


def rec_for(path, t_end=40.0):
    r = Recording(gt=path, gt_objects=[dict(o) for o in OBJS], t_end=t_end)
    return r


FIN = TaskResult(success=True, stop_reason="finished", agent_turns=3)


def test_object_success_needs_stop_claim_and_radius(cfg, mk, optimum):
    ep = mk(goal={"type": "object", "label": "chair", "instance": "chair_1", "radius": 1.5})
    near = rec_for(straight(1, 1, 2.0, 1.0))                         # ends 0.75 m from chair_1's surface
    assert score_episode(ep, near, FIN, cfg, optimum)["success"] is True
    # reached but the system never stopped: oracle success only
    r = score_episode(ep, near, TaskResult(success=False, stop_reason="timeout"), cfg, optimum)
    assert r["success"] is False and r["oracle_success"] is True and r["failure_category"] == "not_stopped_at_goal"
    # stop issued but no success claim
    assert score_episode(ep, near, TaskResult(success=False, stop_reason="finished"), cfg, optimum)["success"] is False
    # stop issued far away: wrong object / false stop
    far = rec_for(straight(1, 1, 1.0, 4.0))
    r = score_episode(ep, far, FIN, cfg, optimum)
    assert r["success"] is False and r["failure_category"] == "wrong_object"


def test_any_instance_counts_but_named_instance_is_reported(cfg, mk, optimum):
    ep = mk(goal={"type": "object", "label": "chair", "instance": "chair_2", "radius": 1.5})
    r = score_episode(ep, rec_for(straight(1, 1, 2.0, 1.0)), FIN, cfg, optimum)
    assert r["success"] is True                    # chair_1 is a chair: any instance satisfies the goal
    assert r["success_instance"] is False          # but it is not the named one
    assert r["final_distance_instance_m"] > 5


def test_pose_goal_default_radius_and_spl(cfg, mk, optimum):
    ep = mk(goal={"type": "pose", "x": 3.0, "y": 1.0}, mode="goal_pose")
    ok = score_episode(ep, rec_for(straight(1, 1, 2.8, 1.0)), FIN, cfg, optimum)
    assert ok["radius_m"] == 0.5 and ok["success"] and ok["spl"] > 0.8 and ok["spl_basis"] == "grid"
    miss = score_episode(ep, rec_for(straight(1, 1, 2.0, 1.0)), FIN, cfg, optimum)
    assert miss["success"] is False and miss["spl"] == 0.0 and miss["oracle_success"] is False


def test_time_to_oracle_and_final_pose_taken_at_stop_time(cfg, mk, optimum):
    ep = mk(goal={"type": "pose", "x": 3.0, "y": 1.0}, mode="goal_pose")
    path = straight(1, 1, 3.0, 1.0, dur=20.0) + [[30.0, 8.0, 1.0, 0.0]]       # drifts away AFTER the stop at t=20
    r = score_episode(ep, rec_for(path, t_end=20.0), FIN, cfg, optimum)
    assert r["success"] and r["final_distance_m"] < 0.1
    assert 14.5 < r["time_to_oracle_s"] < 16.0


def test_missing_gt_instance_is_an_eval_error_not_a_failure_of_the_system(cfg, mk, optimum):
    ep = mk(goal={"type": "object", "label": "piano", "radius": 1.5})
    r = score_episode(ep, rec_for(straight(1, 1, 2, 1)), FIN, cfg, optimum)
    assert r["failure_category"] == "eval_error" and "piano" in r["eval_error"]


def test_euclidean_fallback_is_flagged(cfg, mk):
    from x30_eval.core.optimum import OptimumProvider
    ep = mk(goal={"type": "pose", "x": 3.0, "y": 1.0}, mode="goal_pose", start=(1.0, 1.0, 0.0))
    ep.scene = "unknown_scene"
    r = score_episode(ep, rec_for(straight(1, 1, 2.8, 1.0)), FIN, cfg, OptimumProvider(cfg, grids={}))
    assert r["spl_basis"] == "euclidean"
    ep.shortest_path_m = 2.0
    assert score_episode(ep, rec_for(straight(1, 1, 2.8, 1.0)), FIN, cfg, OptimumProvider(cfg, grids={}))["spl_basis"] == "episode"


def test_reference_path_gives_ndtw_and_sdtw(cfg, mk, optimum):
    ep = mk(goal={"type": "pose", "x": 3.0, "y": 1.0}, mode="goal_pose", reference_path=[[1, 1], [3, 1]])
    r = score_episode(ep, rec_for(straight(1, 1, 2.8, 1.0)), FIN, cfg, optimum)
    assert r["ndtw"] > 0.9 and r["sdtw"] == r["ndtw"]
    r2 = score_episode(ep, rec_for(straight(1, 1, 1.0, 5.0)), FIN, cfg, optimum)
    assert r2["sdtw"] == 0.0 and r2["ndtw"] < r["ndtw"]


def test_collisions_contacts_beat_grid_and_interventions_counted(cfg, mk, optimum):
    ep = mk(goal={"type": "pose", "x": 3.0, "y": 1.0}, mode="goal_pose")
    rec = rec_for(straight(1, 1, 2.8, 1.0))
    rec.contacts, rec.contacts_available = [5.0, 5.3, 20.0], True
    rec.interventions = [{"t": 3.0, "kind": "teleop"}]
    rec.safety = [{"t": 1, "stopped": True}, {"t": 2, "stopped": True}, {"t": 3, "stopped": False}, {"t": 4, "stopped": True}]
    r = score_episode(ep, rec, FIN, cfg, optimum, grid=optimum.grid_for(ep.scene))
    assert r["collisions"] == 2 and r["collision_basis"] == "contacts"
    assert r["interventions"] == 1 and r["safety_stops"] == 2
    rec.contacts_available = False
    r = score_episode(ep, rec, FIN, cfg, optimum, grid=optimum.grid_for(ep.scene))
    assert r["collision_basis"] == "grid" and r["collisions"] == 0


def test_trace_stats_flow_into_the_record(cfg, mk, optimum):
    ep = mk(goal={"type": "pose", "x": 3.0, "y": 1.0}, mode="goal_pose")
    trace = [{"kind": "turn", "turn": i, "tool": "scan", "args": {"d": 360}, "latency_s": 1.0 + i, "prompt_tokens": 100, "completion_tokens": 10}
             for i in range(1, 5)] + [{"kind": "parse_error", "turn": 5}]
    r = score_episode(ep, rec_for(straight(1, 1, 1.5, 1.0)), TaskResult(success=False, stop_reason="timeout"), cfg, optimum, trace=trace)
    t = r["trace"]
    assert t["turns"] == 4 and t["prompt_tokens"] == 400 and t["parse_errors"] == 1 and t["max_identical_tool_repeat"] == 4
    assert t["vlm_latency_mean_s"] == pytest.approx(3.5)
    assert r["failure_category"] == "agent_loop"
