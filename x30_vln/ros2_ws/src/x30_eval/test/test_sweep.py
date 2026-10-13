"""Sweep harness on the FAKE environment: search -> confirm -> finalize, pruning, overfitting guards."""
import json

import pytest
import yaml

from x30_eval.core.episodes import load_episodes
from x30_eval.core.runlog import HELDOUT_LOCK
from x30_eval.sweep.__main__ import main as sweep_main
from x30_eval.sweep.harness import objective_score, rung_sizes
from x30_eval.sweep.space import load_spec

OBJ = {"type": "object", "label": "plant", "instance": "plant_4", "radius": 1.5}


def write_files(tmp_path, mk, n_dev=6, n_held=4, spec_extra=None):
    eps = [mk(f"dev{i}", OBJ, limit=60).to_dict() for i in range(n_dev)]
    eps += [mk(f"held{i}", OBJ, split="heldout", limit=60).to_dict() for i in range(n_held)]
    ep = tmp_path / "eps.yaml"
    ep.write_text(yaml.safe_dump({"episodes": eps}), encoding="utf-8")
    spec = {"study": "t", "seed": 1, "n_trials": 6, "episodes_per_trial": 4, "rungs": [0.5, 1.0], "pruner": "sha",
            "confirm_top_k": 2, "confirm_repeats": 2, "final_repeats": 2,
            "baseline_overlay": {"modes/object_nav": {"standoff_m": 2.5}},     # a bad baseline: stops too far away (fails)
            "params": {"modes/object_nav.standoff_m": {"type": "float", "low": 0.6, "high": 2.6},
                       "robot_sim.limits.vx": {"type": "float", "low": 0.2, "high": 0.8}}, **(spec_extra or {})}
    sp = tmp_path / "spec.yaml"
    sp.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return ep, sp


def cli(cmd, ep, sp, tmp_path, *extra):
    return sweep_main([cmd, "--spec", str(sp), "--episodes", str(ep), "--env", "fake", "--results-dir", str(tmp_path / "results"), *extra])


def test_spec_validation(tmp_path):
    with pytest.raises(ValueError, match="scoring"):
        load_spec({"study": "x", "params": {"eval.success.pose_radius_m": {"type": "float", "low": 0.1, "high": 2}}})
    with pytest.raises(ValueError, match="rungs"):
        load_spec({"study": "x", "rungs": [0.5], "params": {"a.b": {"type": "float", "low": 0, "high": 1}}})
    assert rung_sizes(load_spec({"study": "x", "rungs": [0.25, 0.5, 1.0], "params": {"a.b": {"type": "int", "low": 0, "high": 1}}}), 8) == [2, 4, 8]


def test_objective_score_weights():
    flat = [{"success": 1.0, "spl": 0.8, "collisions": 2, "duration_s": 30.0, "time_limit_s": 60.0},
            {"success": 0.0, "spl": 0.0, "collisions": 0, "duration_s": 60.0, "time_limit_s": 60.0}]
    assert objective_score(flat, {"success": 1.0, "spl": 0.25, "collisions": 0.05, "time": 0.1}) == pytest.approx(0.5 + 0.1 - 0.05 - 0.075)


def test_search_confirm_finalize_flow_and_guards(tmp_path, mk):
    ep, sp = write_files(tmp_path, mk)
    results = tmp_path / "results"
    # finalize before confirm, and without --final, are refused
    with pytest.raises(RuntimeError, match="confirm"):
        cli("finalize", ep, sp, tmp_path, "--final")
    assert cli("run", ep, sp, tmp_path) == 0
    study = results / "sweeps" / "t"
    search = json.loads((study / "search.json").read_text())
    assert search["n_trials"] == 6 and search["n_complete"] >= 1
    assert search["best"]["params"]["modes/object_nav.standoff_m"] < 1.6          # the search found the working region
    used = {json.loads(line)["trial"] for line in (study / "trials.jsonl").read_text().splitlines()}
    assert len(used) == 6
    # no run directory of the search contains held-out episodes, and the lock does not exist yet
    for run in (study / "runs").iterdir():
        for line in (run / "episodes.jsonl").read_text().splitlines():
            assert not json.loads(line)["episode_id"].startswith("held")
    assert not (results / HELDOUT_LOCK).exists()
    assert cli("run", ep, sp, tmp_path) == 0                                      # resumable: nothing left to do, no new trials
    assert json.loads((study / "search.json").read_text())["n_trials"] == 6
    assert cli("confirm", ep, sp, tmp_path) == 0
    conf = json.loads((study / "confirmation.json").read_text())
    assert conf["baseline"]["score"] < conf["chosen"]["confirm_score"] and conf["chosen"]["beats_baseline_on_dev"]
    assert conf["n_dev_episodes"] == 6 and (study / "chosen_overlay.yaml").exists()
    assert cli("finalize", ep, sp, tmp_path) == 2                                 # without --final the guard refuses
    assert not (results / HELDOUT_LOCK).exists()
    assert cli("finalize", ep, sp, tmp_path, "--final") == 0
    final = json.loads((study / "final.json").read_text())
    assert final["tuned"]["metrics"]["success"]["mean"] > final["baseline"]["metrics"]["success"]["mean"]
    lock = json.loads((results / HELDOUT_LOCK).read_text())
    assert {r["evaluation_number"] for r in lock["runs"]} == {1} and len(lock["runs"]) == 2        # baseline + tuned = ONE held-out look
    assert "held-out" in (study / "final_report.md").read_text()
    assert cli("finalize", ep, sp, tmp_path, "--final") == 0                      # same group: resume, not a second look
    sp2 = tmp_path / "spec2.yaml"
    sp2.write_text(sp.read_text().replace("study: t", "study: other"), encoding="utf-8")
    assert cli("run", ep, sp2, tmp_path) == 0 and cli("confirm", ep, sp2, tmp_path) == 0
    assert cli("finalize", ep, sp2, tmp_path, "--final") == 2                     # another sweep may not look again


def test_evaluator_refuses_heldout_outside_finalize(tmp_path, mk, cfg):
    from x30_eval.sweep.evaluator import RunnerEvaluator
    from x30_eval.testing.fake_env import FakeEnv, fake_optimum
    ev = RunnerEvaluator(cfg, fake_optimum(cfg), tmp_path, lambda o: FakeEnv(overlay=o))
    held = [mk("h", OBJ, split="heldout")]
    with pytest.raises(PermissionError):
        ev.evaluate({}, held, seed=0, tag="x")


def test_pruning_happens_with_aggressive_budget(tmp_path, mk):
    ep, sp = write_files(tmp_path, mk, n_dev=8, spec_extra={"n_trials": 10, "episodes_per_trial": 8, "rungs": [0.25, 0.5, 1.0]})
    assert cli("run", ep, sp, tmp_path) == 0
    search = json.loads((tmp_path / "results" / "sweeps" / "t" / "search.json").read_text())
    assert search["n_pruned"] >= 1 and search["n_complete"] >= 1
    assert search["episodes_evaluated_in_search"] < 10 * 8                       # pruned trials cost fewer episodes
