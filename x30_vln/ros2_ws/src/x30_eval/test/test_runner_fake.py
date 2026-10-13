"""End-to-end chain on the FAKE environment: runner -> scoring -> jsonl/summary/manifest -> report."""
import json
from pathlib import Path

import pytest
import yaml

from x30_eval import run_episodes
from x30_eval.core.episodes import episode_key
from x30_eval.core.runlog import HELDOUT_LOCK, RunDir
from x30_eval.core.runner import SuiteAborted, SuiteOptions, run_suite
from x30_eval.testing.fake_env import SCENE, FakeEnv

PLANT = {"type": "object", "label": "plant", "instance": "plant_4", "radius": 1.5}
MANIFEST = {"args": {"split": "dev", "repeats": 1}, "config_hash": "x", "prompts_hash": "p", "episodes_sha256": "e", "overlay_hash": "o"}


def episodes(mk):
    return [
        mk("oracle_plant", PLANT),
        mk("oracle_pose", {"type": "pose", "x": 9.0, "y": 6.0}, mode="goal_pose"),
        mk("wrong", PLANT, policy="wrong_object"),
        mk("nostop", PLANT, policy="never_stop", limit=100),
        mk("wander", PLANT, policy="wander", limit=40),
        mk("loop", PLANT, policy="loop"),
        mk("collide", PLANT, policy="collider", start=(3.0, 4.0, 0.0)),
        mk("hang", PLANT, policy="hang", limit=20),
    ]


def run(tmp_path, cfg, optimum, eps, name="r", repeats=1, env=None, **kw):
    rd = RunDir(tmp_path / name)
    opts = SuiteOptions(run_name=name, repeats=repeats, log=lambda m: None, sleep=lambda s: None, **kw)
    summary = run_suite(env or FakeEnv(), eps, cfg, rd, opts, optimum, {"run_name": name, **MANIFEST})
    return rd, summary


def test_all_policies_get_the_expected_outcome(tmp_path, cfg, optimum, mk):
    rd, summary = run(tmp_path, cfg, optimum, episodes(mk), repeats=2)
    by = {(r["episode_id"], r["repeat"]): r for r in rd.results()}
    assert len(by) == 16 and all(r["status"] == "done" for r in by.values())
    names = ("oracle_plant", "oracle_pose", "wrong", "nostop", "wander", "loop", "collide", "hang")
    cat = {e: by[(e, 0)]["failure_category"] for e in names}
    assert cat["oracle_plant"] == "success" and cat["oracle_pose"] == "success"
    assert cat["wrong"] == "wrong_object"
    assert cat["nostop"] == "not_stopped_at_goal"
    assert cat["wander"] in ("never_found", "timeout")
    assert cat["loop"] == "agent_loop"
    assert cat["collide"] == "collision" and by[("collide", 0)]["collisions"] >= 1
    assert cat["hang"] in ("timeout", "never_found") and by[("hang", 0)]["stop_reason"] == "evaluator_timeout"
    ok = by[("oracle_plant", 0)]
    assert ok["spl"] > 0.6 and ok["ate"]["ate_rmse_m"] < 0.2 and ok["map"]["recall_covered"] is not None
    assert ok["trace"]["turns"] >= 1 and ok["trace"]["prompt_tokens"] > 0
    assert 0 < ok["duration_s"] < 120
    assert summary["metrics"]["success"]["mean"] == pytest.approx(2 / 8)
    assert summary["failures"]["counts"]["agent_loop"] == 2 and summary["complete"]
    assert (rd.traces / "oracle_plant_r0.jsonl").exists() and (rd.tracks / "oracle_plant_r0.json.gz").exists()


def test_repeats_use_different_seeds_and_runs_are_reproducible(tmp_path, cfg, optimum, mk):
    eps = [mk("w", PLANT, policy="wander", limit=30)]
    a, _ = run(tmp_path, cfg, optimum, eps, "a", repeats=2)
    b, _ = run(tmp_path, cfg, optimum, eps, "b", repeats=2)
    ra, rb = a.results(), b.results()
    assert ra[0]["seed"] != ra[1]["seed"] and ra[0]["path_length_m"] != ra[1]["path_length_m"]
    assert [r["path_length_m"] for r in ra] == [r["path_length_m"] for r in rb]


def test_infrastructure_failure_is_retried_and_not_counted(tmp_path, cfg, optimum, mk):
    eps = [mk("a", PLANT), mk("b", PLANT)]
    env = FakeEnv(inject={"a": ["task_error", "reset_error"], "b": []})
    rd, summary = run(tmp_path, cfg, optimum, eps, env=env)
    res = {r["episode_id"]: r for r in rd.results()}
    assert res["a"]["attempts"] == 3 and res["a"]["status"] == "done" and res["a"]["success"] is True
    assert res["b"]["attempts"] == 1
    inc = rd.infra_failures()
    assert len(inc) == 2 and "Connection refused" in inc[0]["reason"] and "not available" in inc[1]["reason"]
    assert summary["infra_incidents"] == 2 and summary["metrics"]["success"]["mean"] == 1.0 and summary["scored_runs"] == 2


def test_exhausted_retries_are_excluded_reported_and_retried_on_resume(tmp_path, cfg, optimum, mk):
    eps = [mk("a", PLANT), mk("b", PLANT)]
    env = FakeEnv(inject={"a": ["reset_error"] * 4})
    rd, summary = run(tmp_path, cfg, optimum, eps, env=env)
    a = next(r for r in rd.results() if r["episode_id"] == "a")
    assert a["status"] == "infra_failed" and a["attempts"] == 4
    assert summary["scored_runs"] == 1 and summary["infra_failed_runs"] == 1 and summary["metrics"]["success"]["mean"] == 1.0
    assert "EXCLUDED" in summary["warning"]
    rd2, s2 = run(tmp_path, cfg, optimum, eps, env=FakeEnv())          # resume: the lost episode is run again
    assert s2["scored_runs"] == 1 + 1 and s2["complete"]


def test_bench_down_aborts_the_run(tmp_path, cfg, optimum, mk):
    env = FakeEnv(inject={"a": ["health_error"]})
    with pytest.raises(SuiteAborted):
        run(tmp_path, cfg, optimum, [mk("a", PLANT), mk("b", PLANT), mk("c", PLANT)], env=env)
    rd = RunDir(tmp_path / "r")
    assert len(rd.results()) == 2 and all(r["status"] == "infra_failed" for r in rd.results())
    assert rd.summary.exists()                  # the summary is still written for what was done


def test_resume_skips_done_tolerates_truncated_line_and_checks_manifest(tmp_path, cfg, optimum, mk):
    eps = [mk("a", PLANT), mk("b", PLANT, policy="loop"), mk("c", PLANT)]
    rd, _ = run(tmp_path, cfg, optimum, eps[:2])
    with open(rd.episodes, "a", encoding="utf-8") as f:
        f.write('{"episode_id": "c", "repeat": 0, "sta')          # crash in the middle of a write
    rd, _ = run(tmp_path, cfg, optimum, eps)
    done = sorted((r["episode_id"], r["repeat"]) for r in rd.results())
    assert done == [("a", 0), ("b", 0), ("c", 0)]
    bad = {"run_name": "r", **MANIFEST, "config_hash": "CHANGED"}
    with pytest.raises(RuntimeError, match="different setup"):
        run_suite(FakeEnv(), eps, cfg, RunDir(tmp_path / "r"), SuiteOptions(run_name="r", log=lambda m: None), optimum, bad)


def test_start_jitter_is_seeded_and_deterministic(cfg, optimum, mk):
    from x30_eval.core.runner import jitter_start
    cfg["run"]["start_jitter_xy_m"] = 0.5
    ep = mk("a", PLANT)
    s1, s2 = jitter_start(ep, 1, cfg, optimum), jitter_start(ep, 1, cfg, optimum)
    assert s1 == s2 and s1 != jitter_start(ep, 2, cfg, optimum) and s1["x"] != ep.start["x"]


# ------------------------------------------------------------------------------------------- CLI, overlay, held-out lock
def write_episodes(tmp_path, mk) -> Path:
    eps = [mk("dev_a", PLANT).to_dict(), mk("dev_b", {"type": "pose", "x": 9.0, "y": 6.0}, mode="goal_pose").to_dict(),
           mk("held_a", PLANT, split="heldout").to_dict()]
    p = tmp_path / "episodes.yaml"
    p.write_text(yaml.safe_dump({"episodes": eps}), encoding="utf-8")
    return p


def cli(tmp_path, ep_file, *extra):
    return run_episodes.main(["--episodes", str(ep_file), "--env", "fake", "--results-dir", str(tmp_path / "results"), *extra])


def test_cli_dev_run_and_heldout_guard(tmp_path, mk):
    ep = write_episodes(tmp_path, mk)
    lock_path = tmp_path / "results" / HELDOUT_LOCK
    assert cli(tmp_path, ep, "--run-name", "dev1", "--repeats", "2") == 0
    rd = RunDir(tmp_path / "results" / "dev1")
    assert len(rd.results()) == 4 and rd.read_manifest()["config_hash"] and not lock_path.exists()
    assert {r["episode_id"] for r in rd.results()} == {"dev_a", "dev_b"}           # dev never touches held-out episodes
    assert cli(tmp_path, ep, "--run-name", "h1", "--split", "heldout") == 2
    assert cli(tmp_path, ep, "--run-name", "all1", "--split", "all") == 2
    assert not (tmp_path / "results" / "h1").exists() and not lock_path.exists()
    assert cli(tmp_path, ep, "--run-name", "h1", "--split", "heldout", "--final") == 0
    lock = json.loads(lock_path.read_text())
    assert [r["run"] for r in lock["runs"]] == ["h1"] and lock["runs"][0]["evaluation_number"] == 1
    assert cli(tmp_path, ep, "--run-name", "h2", "--split", "heldout", "--final") == 2     # second look refused
    assert not (tmp_path / "results" / "h2").exists()
    assert cli(tmp_path, ep, "--run-name", "h1", "--split", "heldout", "--final") == 0     # resume of the same run is fine
    assert cli(tmp_path, ep, "--run-name", "h2", "--split", "heldout", "--final", "--allow-repeat-heldout") == 0
    assert [r["evaluation_number"] for r in json.loads(lock_path.read_text())["runs"]] == [1, 2]


def test_cli_overlay_changes_behaviour_and_is_recorded(tmp_path, mk):
    ep = write_episodes(tmp_path, mk)
    assert cli(tmp_path, ep, "--run-name", "near", "--episode-ids", "dev_a", "--set", "modes/object_nav.standoff_m=1.0") == 0
    assert cli(tmp_path, ep, "--run-name", "far", "--episode-ids", "dev_a", "--set", "modes/object_nav.standoff_m=2.5") == 0
    near, far = (RunDir(tmp_path / "results" / n) for n in ("near", "far"))
    assert near.results()[0]["success"] is True and far.results()[0]["success"] is False
    assert far.results()[0]["failure_category"] == "wrong_object" and far.results()[0]["final_distance_m"] > 1.5
    mn, mf = near.read_manifest(), far.read_manifest()
    assert mn["overlay_hash"] != mf["overlay_hash"] and mf["overlay"]["modes/object_nav"]["standoff_m"] == 2.5
    assert yaml.safe_load((far.path / "overlay.yaml").read_text())["modes/object_nav"]["standoff_m"] == 2.5


def test_cli_dry_run_mode_filter_and_forced_mode(tmp_path, mk, capsys):
    ep = write_episodes(tmp_path, mk)
    assert cli(tmp_path, ep, "--run-name", "x", "--dry-run", "--mode", "goal_pose") == 0
    out = capsys.readouterr().out
    assert "dev_b" in out and "dev_a" not in out and not (tmp_path / "results" / "x").exists()
    assert cli(tmp_path, ep, "--run-name", "forced", "--force-mode", "goal_pose", "--episode-ids", "dev_b") == 0
    assert RunDir(tmp_path / "results" / "forced").results()[0]["mode"] == "goal_pose"


def test_episode_key():
    assert episode_key("a", 2) == "a#r2"
    assert SCENE == "fake_house"
