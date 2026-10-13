"""Evaluators used by the sweep harness: run a set of episodes under a parameter overlay and return scored records."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Protocol

from ..core.episodes import Episode
from ..core.optimum import OptimumProvider
from ..core.overlay import ENV_VAR, write_overlay
from ..core.runlog import RunDir
from ..core.runner import SuiteOptions, run_suite
from ..core.summary import scored
from ..env.base import Env


class Evaluator(Protocol):
    def evaluate(self, overlay: dict, episodes: list[Episode], *, seed: int, tag: str, repeats: int = 1,
                 allow_heldout: bool = False) -> list[dict]:
        """Flat result records of the scored runs (infrastructure failures excluded)."""


class RunnerEvaluator:
    """Runs the episode runner for every call; each call is a normal run directory under ``<study>/runs/<tag>``
    (so reports and exports work on it). ``env_factory(overlay)`` returns the environment for that overlay:
    FakeEnv(overlay=...) in tests, a stack restart plus RosEnv for real runs."""

    def __init__(self, cfg: dict, optimum: OptimumProvider, study_dir: Path, env_factory: Callable[[dict], Env],
                 log: Callable[[str], None] = print):
        self.cfg, self.optimum, self.study_dir, self.env_factory, self.log = cfg, optimum, Path(study_dir), env_factory, log

    def evaluate(self, overlay: dict, episodes: list[Episode], *, seed: int, tag: str, repeats: int = 1,
                 allow_heldout: bool = False) -> list[dict]:
        if not allow_heldout and any(e.split != "dev" for e in episodes):
            raise PermissionError("sweeps may only evaluate the dev split; the held-out split is reserved for `finalize --final`")
        run = RunDir(self.study_dir / "runs" / tag).create()
        write_overlay(run.path / "overlay.yaml", overlay)
        env = self.env_factory(overlay)
        opts = SuiteOptions(run_name=tag, repeats=repeats, seed=seed, resume=True, log=lambda m: None)
        manifest = {"run_name": tag, "args": {"split": episodes[0].split if episodes else "dev", "repeats": repeats, "seed": seed},
                    "overlay": overlay, "config_hash": "sweep", "prompts_hash": "", "episodes_sha256": ",".join(e.id for e in episodes)[:64],
                    "overlay_hash": str(hash(str(sorted(overlay.items())))), "episode_ids": [e.id for e in episodes]}
        try:
            run_suite(env, episodes, self.cfg, run, opts, self.optimum, manifest)
        finally:
            env.close()
        return scored(run.results())


class StackEnvFactory:
    """Real-stack factory: writes the overlay file, restarts the stack with ``stack_restart_cmd`` (``{overlay}`` is
    replaced by the overlay path and X30_CONFIG_OVERLAY is exported), waits, then returns one shared RosEnv.
    Nodes read the overlay when they start (docs/requests/eval-to-bringup.md), hence the restart per trial."""

    def __init__(self, cfg: dict, restart_cmd: str | None, wait_s: float, overlay_dir: Path):
        from ..env.ros_env import RosEnv
        self.cfg, self.cmd, self.wait_s, self.dir = cfg, restart_cmd, wait_s, Path(overlay_dir)
        self.env = RosEnv(cfg)
        self._n = 0

    def __call__(self, overlay: dict) -> Env:
        self._n += 1
        path = write_overlay(self.dir / f"overlay_{self._n:04d}.yaml", overlay)
        if self.cmd:
            subprocess.run(self.cmd.format(overlay=path), shell=True, check=True, timeout=900, env={**os.environ, ENV_VAR: str(path)})
            time.sleep(self.wait_s)
        elif overlay:
            raise RuntimeError("the real stack needs sweep.stack_restart_cmd to apply a parameter overlay; refusing to run "
                               "trials whose parameters would silently be ignored")
        self.env.recover(self.cfg)
        self.env.close = lambda: None        # keep the node alive between trials
        return self.env
