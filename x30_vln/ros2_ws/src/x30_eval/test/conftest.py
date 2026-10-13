"""Shared fixtures. Everything here runs without ROS, Gazebo or Docker (the fake env is labelled as such)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x30_eval.core.config import load_eval_config  # noqa: E402
from x30_eval.core.episodes import Episode  # noqa: E402
from x30_eval.core.optimum import OptimumProvider  # noqa: E402
from x30_eval.testing.fake_env import SCENE, default_grid  # noqa: E402


@pytest.fixture()
def cfg():
    c = load_eval_config()
    c["run"]["infra_backoff_s"] = 0.0
    c["report"]["n_boot"] = 300
    return c


@pytest.fixture()
def optimum(cfg):
    return OptimumProvider(cfg, grids={SCENE: default_grid()})


def make_episode(eid="e", goal=None, policy="oracle", split="dev", start=(1.0, 1.0, 0.0), limit=120.0, mode="object_nav", **kw):
    goal = goal or {"type": "object", "label": "chair", "instance": "chair_2", "radius": 1.5}
    return Episode(id=eid, scene=SCENE, mode=mode, instruction=f"go to {goal.get('label', 'the pose')}",
                   start={"x": start[0], "y": start[1], "yaw": start[2], "z": 0.0}, goal=goal, time_limit_s=limit,
                   split=split, extra={"fake_policy": policy}, **kw)


@pytest.fixture()
def mk():
    return make_episode
