import json
import math
import threading

import pytest
from fake_world import FakeWorld, ground_truth_success
from mock_models import FollowMock
from x30_common import VLMClient
from x30_common.testing.fake_openai import text_reply, tool_call_reply

from x30_agent.core.follower import FollowerAgent, LoopDetector, heuristic_subgoals
from x30_agent.core.policy import load_policy
from x30_agent.core.primitives import Primitive, clamp, parse_action_text, to_skill
from x30_agent.core.types import Pose2D, Task
from conftest import CONFIGS

PLAN = text_reply(json.dumps({"subgoals": [{"do": "walk ahead", "done_when": "wall near"}, {"do": "stop", "done_when": "wall near"}]}))


@pytest.fixture()
def make_follower(srv, tmp_path):
    def _make(world, overrides=None, cancel=None, fmt="tool"):
        ov = dict(overrides or {})
        ov["output_format"] = fmt
        pol = load_policy("velocity_follow", CONFIGS / "agent", ov)
        client = VLMClient(srv.base_url, "fake", backoff_s=0.01, timeout_s=10, max_retries=1)
        return FollowerAgent(client, world, pol, trace_dir=str(tmp_path / "traces"), cancel=cancel)
    return _make


def act(action, amount=0, done=False):
    return tool_call_reply("act", {"action": action, "amount": amount, "subgoal_done": done})


def user_text(req):
    return " ".join(p.get("text", "") for p in req["messages"][-1]["content"] if p["type"] == "text")


def test_primitive_parser_and_clamp():
    assert parse_action_text("The next action is move forward 75 cm") == Primitive("move_forward", 75)
    assert parse_action_text("Turn to the left by 30 degrees") == Primitive("turn_left", 30)
    assert parse_action_text("rotate right 1.5 m") .action == "turn_right"
    assert parse_action_text("move forward 2 meters").amount == 200
    assert parse_action_text("stop").action == "stop" and parse_action_text("dance") is None
    pol = load_policy("velocity_follow", CONFIGS / "agent")
    assert clamp(Primitive("move_forward", 5000), pol.actions)[0].amount == 300
    assert clamp(Primitive("turn_left", 0), pol.actions)[0].amount == 30
    assert to_skill(Primitive("turn_right", 30)) == ("turn_to", {"degrees": -30})
    assert to_skill(Primitive("strafe_left", 40)) == ("move_relative", {"direction": "left", "distance_cm": 40})


def test_heuristic_subgoals():
    g = heuristic_subgoals("Go down the corridor, turn left at the door and then stop next to the table.")
    assert len(g) >= 3 and "table" in g[-1].do


def test_follow_steps_blocked_feedback_and_stop(srv, make_follower, tmp_path):
    w = FakeWorld(seed=2, start=(1.0, 1.0, math.pi))          # facing the west wall, 0.6 m away
    ag = make_follower(w, {"limits": {"stop_refusals_before_accept": 1}, "stop_check": {"enabled": False}})
    srv.queue(PLAN, act("move_forward", 200), act("move_forward", 50, True), act("stop", 0, True))
    res = ag.run(Task("Walk ahead and stop", mode="velocity_follow", episode_id="f1"))
    assert res.success and res.stop_reason == "finished"
    assert "BLOCKED" in user_text(srv.requests[2]) or "moved" in user_text(srv.requests[2])
    assert any("image_url" in json.dumps(r["messages"][-1]["content"]) for r in srv.requests[1:])
    rows = [json.loads(x) for x in (tmp_path / "traces" / "f1.jsonl").read_text().splitlines()]
    assert rows[0]["kind"] == "header" and [r["tool"] for r in rows if r["kind"] == "turn"] == ["act", "act", "act"]


def test_metric_target_parser():
    from x30_agent.core.follower import Subgoal, metric_target
    assert metric_target("turn left about 90 degrees") == ("turn", 90.0)
    assert metric_target("Turn right 45 deg") == ("turn", -45.0)
    assert metric_target("walk forward about 2.5 meters") == ("move", 250.0)
    assert metric_target("move ahead 80 cm") == ("move", 80.0)
    assert metric_target("walk down the corridor") is None and metric_target("move back 2 meters") is None
    g = Subgoal("turn left about 90 degrees")
    assert g.reached([0, 80.0], 0.85) and not g.reached([0, 60.0], 0.85) and not g.reached([0, -90.0], 0.85)
    assert Subgoal("walk forward 1 meters").reached([90.0, 0], 0.85)


def test_metric_subgoals_advance_without_the_model_saying_so(srv, make_follower):
    w = FakeWorld(seed=2)
    plan = text_reply(json.dumps({"subgoals": [{"do": "turn left about 90 degrees", "done_when": "x"}, {"do": "walk forward about 1 meters", "done_when": "y"},
                                               {"do": "stop next to the table", "done_when": "z"}]}))
    ag = make_follower(w, {"limits": {"stop_refusals_before_accept": 0}, "stop_check": {"enabled": False}})
    srv.queue(plan, act("turn_left", 90), act("move_forward", 100), act("stop", 0, True))
    res = ag.run(Task("Turn left 90 degrees, walk forward 1 meter, stop next to the table", mode="velocity_follow", episode_id="f_auto"))
    assert res.success and res.stats["auto_advanced"] == 2 and res.stats["stop_refusals"] == 0 and res.stats["subgoals_done"] == 3
    assert "is complete" in user_text(srv.requests[2])


def test_stop_refused_until_subgoals_done_then_override(srv, make_follower):
    w = FakeWorld(seed=2)
    ag = make_follower(w, {"limits": {"stop_refusals_before_accept": 2}, "stop_check": {"enabled": False}})
    srv.queue(PLAN, act("stop"), act("stop"), act("stop"))
    res = ag.run(Task("Walk ahead and stop", mode="velocity_follow", episode_id="f2"))
    assert res.success and res.guard_overridden and res.stats["stop_refusals"] == 2
    assert "Stop refused" in user_text(srv.requests[2])


def test_stop_check_no_refuses(srv, make_follower):
    w = FakeWorld(seed=2)
    ag = make_follower(w, {"limits": {"stop_refusals_before_accept": 1}})
    srv.queue(PLAN, act("stop", 0, True), act("stop", 0, True), text_reply("no, the goal is not in view"), act("stop", 0, True))
    # first stop: 1 of 2 subgoals marked done by a single subgoal_done -> not all done; refused
    res = ag.run(Task("Walk ahead and stop", mode="velocity_follow", episode_id="f3", max_turns=6))
    assert res.stats["stop_refusals"] >= 1


def test_loop_blocked_repeat_refused_and_safety_stop(srv, make_follower):
    w = FakeWorld(seed=2, start=(0.7, 4.0, math.pi))       # nose at the west wall: every forward move is blocked
    ag = make_follower(w, {"limits": {"max_consecutive_blocked": 4}, "loop": {"repeat_blocked": 2}})
    srv.queue(PLAN, *[act("move_forward", 100)] * 8)
    res = ag.run(Task("Walk ahead", mode="velocity_follow", episode_id="f4"))
    assert res.stop_reason == "safety" and res.stats["refused_moves"] >= 1 and res.stats["blocked"] >= 2
    assert sum(1 for c in w.skill_calls if c[0] == "move_relative") <= 3


def test_loop_detector_rules():
    from types import SimpleNamespace as NS
    cfg = NS(window=6, min_net_move_m=0.25, repeat_blocked=2, warn_limit=3, revisit_radius_m=0.4, revisit_count=3, oscillation_len=4)
    d = LoopDetector(cfg)
    p = Pose2D(1, 1)
    for _ in range(6):
        d.add(Primitive("move_forward", 50), False, p)
    assert "net" in d.warning() and d.warn_run == 1
    d2 = LoopDetector(cfg)
    for i in range(4):
        d2.add(Primitive("turn_left" if i % 2 == 0 else "turn_right", 30), False, Pose2D(i, 0))
    assert "alternately" in d2.warning()
    d3 = LoopDetector(cfg)
    d3.add(Primitive("move_forward", 50), True, p)
    d3.add(Primitive("move_forward", 50), True, p)
    assert d3.refuse(Primitive("move_forward", 50)) and not d3.refuse(Primitive("turn_left", 30))


def test_no_progress_loop_abort(srv, make_follower):
    w = FakeWorld(seed=2, start=(3.0, 3.0, 0.0))
    ag = make_follower(w, {"loop": {"min_net_move_m": 50.0, "window": 3, "warn_limit": 2}})
    srv.queue(PLAN, *[act("move_forward", 40 + i) for i in range(10)])
    res = ag.run(Task("Walk ahead", mode="velocity_follow", episode_id="f5"))
    assert res.stop_reason == "no_progress" and res.stats["loop_warnings"] >= 2


def test_text_output_format(srv, make_follower):
    w = FakeWorld(seed=2, start=(3.0, 3.0, 0.0))
    ag = make_follower(w, {"limits": {"stop_refusals_before_accept": 0}, "stop_check": {"enabled": False}}, fmt="text")
    srv.queue(PLAN, text_reply("ACTION: move forward 60 cm\nSUBGOAL_DONE: yes"), text_reply("The next action is turn left 30 degrees"),
              text_reply("ACTION: stop"))
    res = ag.run(Task("Walk ahead", mode="velocity_follow", episode_id="f6"))
    assert [c for c in w.skill_calls if c[0] in ("move_relative", "turn_to")][0] == ("move_relative", {"direction": "forward", "distance_cm": 60.0})
    assert res.success


def test_bad_reply_retries_then_noop_abort(srv, make_follower):
    w = FakeWorld(seed=2)
    ag = make_follower(w, {"limits": {"format_retries": 1, "max_noop_steps": 2}})
    srv.queue(PLAN, *[text_reply("hmm")] * 8)
    res = ag.run(Task("Walk ahead", mode="velocity_follow", episode_id="f7"))
    assert res.stop_reason == "error" and res.stats["noop_steps"] == 2 and res.stats["format_retries"] == 2


def test_cancel(srv, make_follower):
    w = FakeWorld(seed=2)
    ev = threading.Event()
    ev.set()
    res = make_follower(w, cancel=ev).run(Task("x", mode="velocity_follow", episode_id="f8"))
    assert res.stop_reason == "aborted"


def test_planner_failure_falls_back_to_heuristic(srv, make_follower):
    w = FakeWorld(seed=2)
    ag = make_follower(w, {"limits": {"stop_refusals_before_accept": 0}, "stop_check": {"enabled": False}})
    srv.queue(text_reply("not json at all"), act("stop"))
    res = ag.run(Task("Go forward, turn left and stop", mode="velocity_follow", episode_id="f9"))
    assert res.stats["subgoals"] >= 2 and res.success


def test_oracle_mock_episodes_in_fake_world(srv, make_follower):
    ok = 0
    n = 8
    for i in range(n):
        w = FakeWorld(seed=200 + i)
        _, tgt = w.make_task()
        srv.handler = FollowMock(w, tgt, seed=i)
        srv.replies.clear()
        res = make_follower(w, {"stop_check": {"enabled": True}}).run(Task(f"Walk to the {tgt.label}", mode="velocity_follow", episode_id=f"o{i}", max_turns=60, time_budget_s=900))
        ok += bool(res.success and ground_truth_success(w, tgt))
        assert res.stop_reason in ("finished", "no_progress", "safety", "timeout"), res
    assert ok >= n // 2
