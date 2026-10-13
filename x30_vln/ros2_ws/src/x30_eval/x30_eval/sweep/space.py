"""Sweep specification: which config values to search, the objective, the budget."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..core.overlay import dotted_to_overlay

DEFAULT_OBJECTIVE = {"success": 1.0, "spl": 0.25, "collisions": 0.05, "time": 0.0}


@dataclass
class Param:
    name: str                       # "<config file stem>.<key>[.<subkey>]", e.g. modes/object_nav.standoff_m
    type: str                       # float | int | categorical | bool
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: list = field(default_factory=list)

    def suggest(self, trial):
        if self.type == "float":
            return trial.suggest_float(self.name, self.low, self.high, log=self.log, step=self.step)
        if self.type == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log, step=int(self.step or 1))
        if self.type == "bool":
            return trial.suggest_categorical(self.name, [False, True])
        if self.type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        raise ValueError(f"param {self.name}: unknown type {self.type!r}")


@dataclass
class SweepSpec:
    study: str
    params: list[Param]
    seed: int = 0
    n_trials: int = 20
    episodes_per_trial: int = 8             # budget of one trial (episodes of the DEV split, one repeat each)
    rungs: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])   # fractions of the budget at which a trial may be pruned
    pruner: str = "sha"                     # sha (successive halving) | median | none
    reduction_factor: int = 2
    objective: dict = field(default_factory=lambda: dict(DEFAULT_OBJECTIVE))
    baseline_overlay: dict = field(default_factory=dict)   # applied to every trial and to the baseline (fixed non-searched settings)
    confirm_top_k: int = 3
    confirm_repeats: int = 3
    final_repeats: int = 3
    stack_restart_cmd: str | None = None    # ROS runs: command that (re)starts the stack with $X30_CONFIG_OVERLAY={overlay}
    stack_wait_s: float = 90.0

    def overlay_for(self, values: dict) -> dict:
        from ..core.overlay import deep_merge
        return deep_merge(self.baseline_overlay, dotted_to_overlay(values))

    def suggest(self, trial) -> dict:
        return {p.name: p.suggest(trial) for p in self.params}


def load_spec(path: str | Path | dict) -> SweepSpec:
    raw = path if isinstance(path, dict) else yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    params = []
    for name, d in (raw.get("params") or {}).items():
        if name.split(".")[0] == "eval":
            raise ValueError(f"param {name}: scoring parameters (eval.*) cannot be swept; that would tune the metric, not the system")
        params.append(Param(name=name, **d))
    if not params:
        raise ValueError("sweep spec has no params")
    spec = SweepSpec(study=raw["study"], params=params, **{k: raw[k] for k in (
        "seed", "n_trials", "episodes_per_trial", "rungs", "pruner", "reduction_factor", "objective", "baseline_overlay",
        "confirm_top_k", "confirm_repeats", "final_repeats", "stack_restart_cmd", "stack_wait_s") if k in raw})
    if "eval" in spec.baseline_overlay:
        raise ValueError("baseline_overlay must not change the eval scoring config")
    if sorted(spec.rungs) != spec.rungs or spec.rungs[-1] != 1.0:
        raise ValueError("rungs must be increasing fractions ending in 1.0")
    return spec
