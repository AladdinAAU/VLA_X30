"""Optuna-based parameter search over config overlays, with guards against overfitting to the evaluation episodes.

Protocol (docs/EVALUATION.md section 'Sweeps'):
  1. ``run``      search on the DEV split with a small per-trial budget and successive halving (pruning).
  2. ``confirm``  re-evaluate the top-k trials and the baseline on the full dev split with fresh seeds and repeats.
  3. ``finalize`` (needs --final) evaluate the confirmed configuration and the baseline once on the HELD-OUT split.
Only step 3 touches held-out episodes; it counts as one held-out evaluation in results/.heldout_lock.json.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from ..core import stats as S
from ..core.episodes import Episode, select
from ..core.overlay import write_overlay
from ..core.runlog import HeldoutGuard
from .evaluator import Evaluator
from .space import SweepSpec


def objective_score(flat: list[dict], weights: dict) -> float:
    """Weighted score of scored runs: w_sr*SR + w_spl*SPL - w_coll*collisions/run - w_time*duration/time-limit."""
    if not flat:
        return math.nan

    def mean(key, f=lambda x: x):
        v = [f(r[key]) for r in flat if r.get(key) is not None]
        return float(np.mean(v)) if v else 0.0
    time_frac = float(np.mean([r["duration_s"] / r["time_limit_s"] for r in flat if r.get("duration_s") is not None and r.get("time_limit_s")] or [0.0]))
    return (weights.get("success", 0.0) * mean("success", float) + weights.get("spl", 0.0) * mean("spl")
            - weights.get("collisions", 0.0) * mean("collisions") - weights.get("time", 0.0) * time_frac)


def trial_episodes(spec: SweepSpec, dev: list[Episode]) -> list[Episode]:
    """The fixed episode subset every trial uses (same order, so trials are compared on identical episodes)."""
    order = list(np.random.default_rng(spec.seed).permutation(len(dev)))
    return [dev[i] for i in order[: spec.episodes_per_trial]]


def rung_sizes(spec: SweepSpec, n: int) -> list[int]:
    sizes = sorted({max(1, math.ceil(f * n)) for f in spec.rungs})
    return sizes if sizes[-1] == n else sizes + [n]


def _pruner(spec: SweepSpec):
    import optuna
    if spec.pruner == "sha":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=spec.reduction_factor)
    if spec.pruner == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    return optuna.pruners.NopPruner()


class Sweep:
    def __init__(self, spec: SweepSpec, evaluator: Evaluator, episodes: list[Episode], study_dir: Path, results_root: Path,
                 log: Callable[[str], None] = print):
        self.spec, self.evaluator, self.all_episodes = spec, evaluator, episodes
        self.dev = select(episodes, "dev")
        self.dir = Path(study_dir)
        self.results_root = Path(results_root)
        self.log = log
        if not self.dev:
            raise ValueError("no dev episodes")
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ search
    def study(self):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna.create_study(study_name=self.spec.study, storage=f"sqlite:///{(self.dir / 'study.db').as_posix()}", load_if_exists=True,
                                   direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.spec.seed),
                                   pruner=_pruner(self.spec))

    def run(self, n_trials: int | None = None) -> dict:
        import optuna
        study = self.study()
        eps = trial_episodes(self.spec, self.dev)
        sizes = rung_sizes(self.spec, len(eps))
        (self.dir / "spec.yaml").write_text(yaml.safe_dump({"study": self.spec.study, "seed": self.spec.seed, "objective": self.spec.objective,
                                                            "episodes": [e.id for e in eps], "rung_sizes": sizes,
                                                            "params": [p.__dict__ for p in self.spec.params]}, sort_keys=False), encoding="utf-8")
        todo = (n_trials or self.spec.n_trials) - len([t for t in study.trials if t.state.is_finished()])

        def objective(trial):
            values = self.spec.suggest(trial)
            overlay = self.spec.overlay_for(values)
            results: list[dict] = []
            start = 0
            score = math.nan
            for step, size in enumerate(sizes, 1):
                chunk = eps[start:size]
                start = size
                got = self.evaluator.evaluate(overlay, chunk, seed=self.spec.seed, tag=f"trial{trial.number:04d}_s{step}")
                results += got
                if len(results) < max(1, math.ceil(0.5 * size)):
                    trial.set_user_attr("aborted", "too many infrastructure failures")
                    raise optuna.TrialPruned()
                score = objective_score(results, self.spec.objective)
                trial.report(score, step)
                self._log_trial(trial.number, values, overlay, step, size, score, len(results))
                if step < len(sizes) and trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("n_scored", len(results))
            return score

        if todo > 0:
            study.optimize(objective, n_trials=todo)
        best = self.best_trials(study, 1)
        out = {"n_trials": len(study.trials), "n_complete": len([t for t in study.trials if t.state.name == "COMPLETE"]),
               "n_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
               "best": best[0] if best else None,
               "episodes_evaluated_in_search": self._episodes_evaluated()}
        (self.dir / "search.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        return out

    def _episodes_evaluated(self) -> int:
        """Episode runs spent in the search (pruned trials stop early, so this is below trials x budget)."""
        last: dict[int, int] = {}
        path = self.dir / "trials.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                last[d["trial"]] = d["n_scored"]
        return sum(last.values())

    def _log_trial(self, number, values, overlay, step, size, score, n):
        with open(self.dir / "trials.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"trial": number, "step": step, "episodes": size, "n_scored": n, "score": score, "params": values,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, default=str) + "\n")

    def best_trials(self, study, k: int) -> list[dict]:
        done = sorted([t for t in study.trials if t.state.name == "COMPLETE"], key=lambda t: -t.value)
        return [{"number": t.number, "search_score": t.value, "params": t.params, "overlay": self.spec.overlay_for(t.params)} for t in done[:k]]

    # ------------------------------------------------------------------ confirmation on the full dev split
    def confirm(self) -> dict:
        """Top-k trials and the baseline on ALL dev episodes with repeats and seeds never used in the search.

        The search score of the best trial is optimistic (it was selected for being high on a small, noisy budget);
        this step gives the unbiased-by-selection dev estimate for the candidates. The final pick among the k
        candidates is again a (small) selection, so the held-out step remains the only honest generalisation number."""
        study = self.study()
        cands = self.best_trials(study, self.spec.confirm_top_k)
        if not cands:
            raise RuntimeError("no completed trials to confirm")
        seed = self.spec.seed + 10_000
        base = self.evaluator.evaluate(self.spec.baseline_overlay, self.dev, seed=seed, tag="confirm_baseline", repeats=self.spec.confirm_repeats)
        out = {"seed": seed, "repeats": self.spec.confirm_repeats, "n_dev_episodes": len(self.dev), "candidates": [],
               "baseline": {"overlay": self.spec.baseline_overlay, "score": objective_score(base, self.spec.objective),
                            "metrics": S.summarise_all(base, 1000)}}
        best = None
        for c in cands:
            res = self.evaluator.evaluate(c["overlay"], self.dev, seed=seed, tag=f"confirm_trial{c['number']:04d}", repeats=self.spec.confirm_repeats)
            score = objective_score(res, self.spec.objective)
            entry = {**c, "confirm_score": score, "metrics": S.summarise_all(res, 1000),
                     "paired_vs_baseline": {k: S.paired_bootstrap(base, res, k, 2000) for k in ("success", "spl", "duration_s")}}
            out["candidates"].append(entry)
            if best is None or score > best["confirm_score"]:
                best = entry
        out["chosen"] = {"number": best["number"], "overlay": best["overlay"], "params": best["params"], "confirm_score": best["confirm_score"],
                         "beats_baseline_on_dev": best["confirm_score"] > out["baseline"]["score"]}
        write_overlay(self.dir / "chosen_overlay.yaml", best["overlay"])
        (self.dir / "confirmation.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        return out

    # ------------------------------------------------------------------ one look at the held-out split
    def finalize(self, final: bool) -> dict:
        conf_path = self.dir / "confirmation.json"
        if not conf_path.exists():
            raise RuntimeError("run `confirm` first: the held-out evaluation compares a configuration already confirmed on dev")
        held = select(self.all_episodes, "heldout")
        if not held:
            raise RuntimeError("no heldout episodes in the episode file")
        guard = HeldoutGuard(self.results_root)
        name = {k: f"{self.spec.study}-heldout-{k}" for k in ("baseline", "tuned")}
        for n in name.values():
            guard.check_and_register(n, "heldout", final, group=self.spec.study, info={"sweep": self.spec.study, "n_episodes": len(held)})
        conf = json.loads(conf_path.read_text(encoding="utf-8"))
        seed = self.spec.seed + 20_000
        base = self.evaluator.evaluate(self.spec.baseline_overlay, held, seed=seed, tag=name["baseline"], repeats=self.spec.final_repeats, allow_heldout=True)
        tuned = self.evaluator.evaluate(conf["chosen"]["overlay"], held, seed=seed, tag=name["tuned"], repeats=self.spec.final_repeats, allow_heldout=True)
        out = {"seed": seed, "repeats": self.spec.final_repeats, "n_heldout_episodes": len(held),
               "baseline": {"score": objective_score(base, self.spec.objective), "metrics": S.summarise_all(base, 2000)},
               "tuned": {"score": objective_score(tuned, self.spec.objective), "metrics": S.summarise_all(tuned, 2000), "params": conf["chosen"]["params"]},
               "paired_tuned_vs_baseline": {k: S.paired_bootstrap(base, tuned, k, 4000) for k in ("success", "oracle_success", "spl", "duration_s")},
               "dev_confirm_score_tuned": conf["chosen"]["confirm_score"], "dev_confirm_score_baseline": conf["baseline"]["score"],
               "search": json.loads((self.dir / "search.json").read_text(encoding="utf-8")) if (self.dir / "search.json").exists() else None}
        out["generalisation_gap"] = out["dev_confirm_score_tuned"] - out["tuned"]["score"]
        (self.dir / "final.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        (self.dir / "final_report.md").write_text(final_markdown(self.spec, out), encoding="utf-8")
        return out


def final_markdown(spec: SweepSpec, out: dict) -> str:
    def m(block, k):
        s = (block["metrics"] or {}).get(k)
        return f"{s['mean']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]" if s else "n/a"
    pb = out["paired_tuned_vs_baseline"].get("success", {})
    lines = [f"# Sweep `{spec.study}`: final report", "",
             f"Held-out episodes: {out['n_heldout_episodes']}, repeats {out['repeats']}. The held-out split was evaluated ONCE for this sweep "
             "(baseline and tuned configuration together). Do not tune further on these numbers.", "",
             "| | dev (confirmation, fresh seeds) | held-out |", "|---|---|---|",
             f"| baseline score | {out['dev_confirm_score_baseline']:.3f} | {out['baseline']['score']:.3f} |",
             f"| tuned score | {out['dev_confirm_score_tuned']:.3f} | {out['tuned']['score']:.3f} |",
             f"| tuned SR [95% CI] | | {m(out['tuned'], 'success')} |", f"| baseline SR [95% CI] | | {m(out['baseline'], 'success')} |",
             f"| tuned SPL [95% CI] | | {m(out['tuned'], 'spl')} |", f"| baseline SPL [95% CI] | | {m(out['baseline'], 'spl')} |", "",
             f"Generalisation gap (dev confirmation minus held-out, tuned): {out['generalisation_gap']:+.3f}.",
             f"Paired SR difference tuned - baseline on held-out: {pb.get('diff', float('nan')):+.3f} "
             f"[{pb.get('ci_lo', float('nan')):+.3f}, {pb.get('ci_hi', float('nan')):+.3f}], p={pb.get('p_value', float('nan')):.3f}, "
             f"{pb.get('n_pairs', 0)} episodes.", "",
             f"Tuned parameters: `{json.dumps(out['tuned']['params'])}`.",
             f"Search budget: {out['search']['n_trials'] if out.get('search') else '?'} trials, "
             f"{out['search']['episodes_evaluated_in_search'] if out.get('search') else '?'} dev episode evaluations (multiple comparisons: "
             "the more trials, the more the best dev score overstates real performance)."]
    return "\n".join(lines) + "\n"
