import json

import numpy as np
import pytest
import yaml

from x30_eval import report
from x30_eval.core import overlay as O
from x30_eval import run_episodes
from x30_eval.core import stats as S
from x30_eval.core.episodes import EpisodeError, load_episodes, select
from x30_eval.core.failure import classify
from x30_eval.core.infra import looks_like_infra
from x30_eval.core.recording import Recording, TaskResult
from x30_eval.core.runlog import RunDir
from x30_eval.core.traces import read_trace, trace_stats


# ------------------------------------------------------------------ episodes
def test_episode_loading_validation_and_split_selection(tmp_path):
    good = [{"id": "a", "scene": "s", "mode": "object_nav", "instruction": "x", "start": {"x": 0, "y": 0},
             "goal": {"type": "object", "label": "chair", "radius": 1.5}, "split": "dev", "custom": 5},
            {"id": "b", "scene": "s", "mode": "goal_pose", "start": {"x": 0, "y": 0}, "goal": {"type": "pose", "x": 1, "y": 2}, "split": "heldout"}]
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump(good))
    eps = load_episodes(p)
    assert eps[0].extra == {"custom": 5} and eps[0].start["yaw"] == 0.0 and eps[1].time_limit_s == 300.0
    assert [e.id for e in select(eps, "dev")] == ["a"] and [e.id for e in select(eps, "all")] == ["a", "b"]
    j = tmp_path / "e.jsonl"
    j.write_text("\n".join(json.dumps(g) for g in good))
    assert len(load_episodes(j)) == 2
    for bad in ({**good[0], "goal": {"type": "object"}}, {**good[0], "mode": "fly"}, {**good[0], "split": "test"}, {**good[0], "time_limit_s": 0}):
        p.write_text(yaml.safe_dump([bad]))
        with pytest.raises(EpisodeError):
            load_episodes(p)
    p.write_text(yaml.safe_dump([good[0], good[0]]))
    with pytest.raises(EpisodeError, match="duplicate"):
        load_episodes(p)


# ------------------------------------------------------------------ overlay
def test_overlay_merge_dotted_and_env(tmp_path, monkeypatch):
    base = {"a": {"x": 1, "y": {"z": 2}}, "l": [1, 2]}
    assert O.deep_merge(base, {"a": {"y": {"z": 3}}, "l": [9]}) == {"a": {"x": 1, "y": {"z": 3}}, "l": [9]}
    assert base["a"]["y"]["z"] == 2                                   # inputs are not mutated
    ov = O.dotted_to_overlay({"modes/object_nav.standoff_m": 1.2, "modes/object_nav.limits.max": 3, "models.reasoner_vlm.temperature": 0.1})
    assert ov == {"modes/object_nav": {"standoff_m": 1.2, "limits": {"max": 3}}, "models": {"reasoner_vlm": {"temperature": 0.1}}}
    f1, f2 = tmp_path / "1.yaml", tmp_path / "2.yaml"
    f1.write_text(yaml.safe_dump({"models": {"a": 1, "b": 1}}))
    f2.write_text(yaml.safe_dump({"models": {"b": 2}}))
    monkeypatch.setenv(O.ENV_VAR, f"{f1}{__import__('os').pathsep}{f2}")
    assert O.apply_overlay("models", {"a": 0, "c": 0}) == {"a": 1, "c": 0, "b": 2}
    assert O.apply_overlay("other", {"q": 1}) == {"q": 1}
    with pytest.raises(ValueError):
        O.set_dotted({}, "nodots", 1)


def test_overlay_typo_detection(tmp_path):
    (tmp_path / "models.yaml").write_text(yaml.safe_dump({"reasoner_vlm": {"temperature": 0.2}}))
    assert O.check_overlay_against(tmp_path, {"models": {"reasoner_vlm": {"temperature": 0.5}}}) == []
    assert O.check_overlay_against(tmp_path, {"models": {"reasoner_vlm": {"temprature": 0.5}}, "nofile": {"a": 1}}) == [
        "models.reasoner_vlm.temprature", "nofile"]


# ------------------------------------------------------------------ stats
def test_bootstrap_paired_and_repeat_std():
    lo, hi = S.bootstrap_mean_ci([1, 0, 1, 1, 0, 1, 1, 1], 2000)[1:]
    assert 0.2 < lo < 0.75 < hi <= 1.0
    assert S.bootstrap_mean_ci([])[0] != S.bootstrap_mean_ci([])[0]               # NaN for empty input
    a = [{"episode_id": f"e{i}", "repeat": r, "success": 0.0} for i in range(8) for r in range(2)]
    b = [{"episode_id": f"e{i}", "repeat": r, "success": 1.0 if i < 6 else 0.0} for i in range(8) for r in range(2)]
    pb = S.paired_bootstrap(a, b, "success", 2000)
    assert pb["n_pairs"] == 8 and pb["diff"] == pytest.approx(0.75) and pb["ci_lo"] > 0.4 and pb["p_value"] < 0.05
    same = S.paired_bootstrap(a, a, "success", 500)
    assert same["diff"] == 0 and same["p_value"] == 1.0
    reps = [{"episode_id": "e", "repeat": 0, "x": 1.0}, {"episode_id": "e", "repeat": 1, "x": 3.0}]
    assert S.repeat_std(reps, "x") == pytest.approx(np.sqrt(2.0))
    assert S.paired_bootstrap(a, [{"episode_id": "zz", "repeat": 0, "success": 1.0}], "success")["n_pairs"] == 0


def test_repeats_are_averaged_before_resampling():
    recs = [{"episode_id": "a", "repeat": r, "success": 1.0} for r in range(10)] + [{"episode_id": "b", "repeat": 0, "success": 0.0}]
    sm = S.summarise(recs, "success", 500)
    assert sm["n_episodes"] == 2 and sm["mean"] == pytest.approx(0.5)       # 10 repeats of one episode do not outweigh the other episode


# ------------------------------------------------------------------ traces, infra, failure classifier
def test_trace_reader_tolerates_truncation_and_counts(tmp_path):
    rows = [{"kind": "header"}, {"kind": "turn", "turn": 1, "tool": "search", "args": {"l": "a"}, "prompt_tokens": 100, "completion_tokens": 5, "latency_s": 1.0, "parse_ok": True, "success": True},
            {"kind": "turn", "turn": 2, "tool": None, "parse_ok": False, "prompt_tokens": 100, "completion_tokens": 0, "latency_s": 2.0},
            {"kind": "turn", "turn": 3, "tool": "search", "args": {"l": "a"}, "prompt_tokens": 50, "completion_tokens": 5, "latency_s": 3.0, "parse_ok": True, "success": False, "refused": True},
            {"kind": "finish_guard", "turn": 4, "allowed": False}]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"kind": "tur')
    recs = read_trace(p)
    assert len(recs) == 5 and read_trace(tmp_path / "missing.jsonl") == []
    t = trace_stats(recs)
    assert t["turns"] == 3 and t["prompt_tokens"] == 250 and t["parse_errors"] == 1 and t["refused_calls"] == 1
    assert t["vlm_latency_mean_s"] == pytest.approx(2.0) and t["finish_refused"] == 1 and t["skill_failures"] == 1 and t["max_identical_tool_repeat"] == 2


def test_infra_pattern_detection(cfg):
    pats = cfg["infra"]["error_patterns"]
    assert looks_like_infra(TaskResult(stop_reason="error", message="LLM error: Connection refused"), pats)
    assert looks_like_infra(TaskResult(stop_reason="error", message="HTTPConnectionPool: Max retries exceeded"), pats)
    assert looks_like_infra(TaskResult(stop_reason="error", message="agent exception: KeyError"), pats) is None
    assert looks_like_infra(TaskResult(stop_reason="finished", message="connection refused"), pats) is None      # a finished run is never reclassified
    assert looks_like_infra(TaskResult(stop_reason="timeout", message="connection reset"), pats) is None


def test_failure_classifier_rules(cfg):
    base = {"success": False, "stop_issued": False, "stop_reason": "timeout", "radius_m": 1.5, "goal_type": "object", "collisions": 0, "safety_stops": 0}
    c = lambda **kw: classify({**base, **kw}, cfg)[0]  # noqa: E731
    assert c(success=True) == "success"
    assert c(infrastructure_failure=True) == "infrastructure"
    assert c(eval_error="x") == "eval_error"
    assert c(collisions=3) == "collision" and c(collisions=1, stop_reason="aborted") == "collision" and c(collisions=1) == "timeout"
    assert c(safety_stops=9) == "safety_stop" and c(stop_reason="safety") == "safety_stop"
    assert c(stop_issued=True, stop_reason="finished", final_distance_m=4.0) == "wrong_object"
    assert c(oracle_success=True) == "not_stopped_at_goal"
    assert c(trace={"max_identical_tool_repeat": 4}) == "agent_loop" and c(stop_reason="no_progress") == "agent_loop"
    assert c(target_in_map=False) == "never_found" and c(target_in_map=True) == "timeout"
    assert c(stop_reason="error", message="boom") == "agent_error"


def test_recording_roundtrip(tmp_path):
    r = Recording(gt=[[0, 1, 2, 0.1]], pairs=[[0, 1, 2, 0.1, 1, 2, 0.1]], contacts=[1.0], contacts_available=True, t_end=3.0, notes=["n"])
    r.save(tmp_path / "x" / "r.json.gz")
    r2 = Recording.load(tmp_path / "x" / "r.json.gz")
    assert r2.gt == r.gt and r2.contacts_available and r2.t_end == 3.0 and r2.pairs_array().shape == (1, 7)


# ------------------------------------------------------------------ report
def test_report_with_two_runs_writes_tables_plots_and_flags(tmp_path, mk):
    from conftest import make_episode
    eps = [make_episode(f"e{i}", {"type": "object", "label": "plant", "instance": "plant_4", "radius": 1.5}).to_dict() for i in range(4)]
    eps.append(make_episode("loop", {"type": "object", "label": "plant", "radius": 1.5}, policy="loop").to_dict())
    ep = tmp_path / "eps.yaml"
    ep.write_text(yaml.safe_dump({"episodes": eps}))
    res = str(tmp_path / "res")
    for name, standoff in (("A", "1.0"), ("B", "2.5")):
        assert run_episodes.main(["--episodes", str(ep), "--env", "fake", "--results-dir", res, "--run-name", name, "--repeats", "2",
                                  "--set", f"modes/object_nav.standoff_m={standoff}"]) == 0
    out = tmp_path / "rep"
    md = report.main([f"{res}/A", f"{res}/B", "--out", str(out)])
    assert md == 0
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "Run `A`" in text and "### `B` vs `A`" in text and "Failure categories" in text and "agent_loop" in text
    for f in ("metrics.csv", "comparison.csv", "headline.png", "sr_by_scene.png", "failure_categories.png", "dtg_cdf.png", "trajectories_A.png", "episodes_A.csv"):
        assert (out / f).exists() and (out / f).stat().st_size > 0, f
    comp = (out / "comparison.csv").read_text()
    assert "success" in comp
    # infrastructure bookkeeping appears in the report
    RunDir(f"{res}/A").append_infra({"episode_id": "e0", "repeat": 0, "attempt": 1, "reason": "x"})
    report.main([f"{res}/A", "--out", str(tmp_path / "rep2")])
    assert "infrastructure incidents 1" in (tmp_path / "rep2" / "report.md").read_text()


def test_dataset_schema_variants_and_real_episode_files(tmp_path):
    """Episode files written by the datasets package: dict reference points, difficulty dict, distance_to, instances."""
    from pathlib import Path

    from x30_eval.core import metrics as M
    ep = {"id": "x", "scene": "s", "mode": "object_nav", "instruction": "i", "start": {"x": 0, "y": 0, "yaw": 0},
          "goal": {"type": "object", "label": "refrigerator", "radius": 1.5, "distance_to": "bbox", "any_instance": False, "instances": ["refrigerator_01"]},
          "reference_path": [{"x": 0, "y": 0}, {"x": 1, "y": 1}], "difficulty": {"tier": "easy", "geodesic_m": 5.66}, "tags": [], "seed": 0, "split": "dev"}
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump([ep]))
    e = load_episodes(p)[0]
    assert e.reference_path == [[0.0, 0.0], [1.0, 1.0]] and e.difficulty == "easy" and e.extra["geodesic_hint_m"] == 5.66
    objs = [{"label": "refrigerator", "name": "refrigerator_01", "id": 1, "x": 5, "y": 0}, {"label": "refrigerator", "name": "refrigerator_02", "id": 2, "x": 0, "y": 5}]
    assert [o["name"] for o in M.goal_instances(e.goal, objs)] == ["refrigerator_01"]               # any_instance: false restricts
    assert M.goal_distance_mode(e.goal, "centroid") == "surface" and M.goal_distance_mode({}, "centroid") == "centroid"
    for f in sorted(Path(__file__).resolve().parents[4].glob("datasets/episodes/*.yaml")):          # whatever the datasets agent produced must load
        assert load_episodes(f)
