import math

import pytest
from conftest import CONFIGS
from fake_world import FakeWorld, ground_truth_success
from mock_models import ObjNavMock
from x30_common.toolschema import coerce_arguments, find_tool

from x30_agent.core.baselines import DeadReckon, ScriptedSearch, target_label_from, world_to_map_path
from x30_agent.core.policy import load_policy
from x30_agent.core.prompts import build_system_prompt, example_calls
from x30_agent.core.tools import build_toolset
from x30_agent.core.types import Pose2D, Task


def test_examples_replay_against_real_schemas(registry):
    pol = load_policy("object_nav", CONFIGS / "agent")
    ts = build_toolset(registry, list(pol.tools), list(pol.local_tools))
    calls = example_calls(pol.files["examples"])
    assert len(calls) >= 8
    for eid, tool, args in calls:
        assert find_tool(ts.tools, tool), (eid, tool)
        clean, problems = coerce_arguments(ts.schemas[tool], args)
        assert not [p for p in problems if "unknown" in p or "missing" in p or "not one of" in p], (eid, tool, problems)


def test_toolset_from_registry(registry):
    pol = load_policy("object_nav", CONFIGS / "agent")
    ts = build_toolset(registry, list(pol.tools), list(pol.local_tools), "summary")
    assert ts.names == list(pol.tools) + list(pol.local_tools)
    assert "anyOf" not in str(ts.tools) and "additionalProperties" not in str(ts.tools)
    full = build_toolset(registry, list(pol.tools), list(pol.local_tools), "full")
    assert len(str(full.tools)) > 1.4 * len(str(ts.tools))
    with pytest.raises(KeyError):
        build_toolset(registry, ["no_such_skill"], [])


def test_policy_hash_changes_with_prompt_and_override():
    a = load_policy("object_nav", CONFIGS / "agent")
    b = load_policy("object_nav", CONFIGS / "agent")
    c = load_policy("object_nav", CONFIGS / "agent", {"limits": {"max_turns": 5}})
    assert a.hash == b.hash != c.hash and len(a.hash) == 12
    with pytest.raises(KeyError):
        load_policy("object_nav", CONFIGS / "agent", {"limits": {"max_turnz": 5}})
    assert "ONE tool call" in build_system_prompt(a)


def test_target_label_extraction():
    assert target_label_from("Find the red chair in the office and stop next to it.") == "chair"
    assert target_label_from("Go to the fire extinguisher.") == "fire extinguisher"
    assert target_label_from("Find a potted plant near the window") == "potted plant"


def test_world_to_map_path_rotation():
    p = world_to_map_path([(2.0, 1.0)], (1.0, 1.0, math.pi / 2), Pose2D(10.0, 5.0, 0.0))
    # one metre ahead in the world frame (+x) is to the right (-y) of a robot facing +y
    assert abs(p[0][0] - 10.0) < 1e-6 and abs(p[0][1] - 4.0) < 1e-6


def test_dead_reckon_reaches_waypoints_and_cancels():
    w = FakeWorld(seed=1, start=(1.0, 1.0, 0.0))
    res = DeadReckon(w).run(Task("go", mode="velocity_follow", episode_id="d1"), waypoints=[(3.0, 1.0), (3.0, 3.0), (1.5, 3.0)])
    assert res.success and math.hypot(w.pos.x - 1.5, w.pos.y - 3.0) < 0.5
    import threading
    ev = threading.Event(); ev.set()
    res = DeadReckon(FakeWorld(seed=1), cancel=ev).run(Task("go", episode_id="d2"), waypoints=[(3.0, 1.0)])
    assert res.stop_reason == "aborted"
    res = DeadReckon(FakeWorld(seed=1)).run(Task("go", episode_id="d3"))
    assert res.stop_reason == "error"


def test_dead_reckon_blocked_by_wall_ends_as_safety():
    w = FakeWorld(seed=1, start=(1.0, 1.0, 0.0))
    res = DeadReckon(w, max_blocked=2).run(Task("go", episode_id="d4"), waypoints=[(8.0, 1.0)])      # straight through the wall at x=5
    assert res.stop_reason == "safety" and not res.success


def test_scripted_search_in_fake_world(registry):
    pol = load_policy("object_nav", CONFIGS / "agent")
    ok = 0
    n = 20
    for i in range(n):
        w = FakeWorld(seed=300 + i)
        instr, tgt = w.make_task()
        res = ScriptedSearch(w, registry, pol).run(Task(instr, episode_id=f"s{i}", max_turns=40, time_budget_s=900))
        assert res.stop_reason in ("finished", "no_progress", "timeout")
        ok += bool(res.success and ground_truth_success(w, tgt))
    assert ok >= 0.7 * n, ok


def test_objnav_noisy_mock_episodes_no_loops(srv, make_agent, tmp_path):
    """20 episodes against a rule-based mock that follows hints and injects malformed calls, repeats and premature finishes."""
    ok = false_claims = 0
    n = 20
    for i in range(n):
        w = FakeWorld(seed=400 + i)
        instr, tgt = w.make_task()
        srv.handler = ObjNavMock(seed=i)
        srv.replies.clear()
        res = make_agent(w, {"limits": {"max_turns": 30}}).run(Task(instr, episode_id=f"m{i}", max_turns=30, time_budget_s=900))
        assert res.stop_reason in ("finished", "timeout", "no_progress"), (res.stop_reason, res.message)
        good = ground_truth_success(w, tgt)
        ok += bool(res.success and good)
        false_claims += bool(res.success and not good)
        assert res.turns <= 30
    assert ok >= 0.6 * n, (ok, false_claims)


def test_hint_keys_are_strings():
    pol = load_policy("object_nav", CONFIGS / "agent")
    assert set(pol.hints["verify"]) >= {"yes", "no", "unsure"}        # unquoted yes/no would load as booleans in YAML 1.1
