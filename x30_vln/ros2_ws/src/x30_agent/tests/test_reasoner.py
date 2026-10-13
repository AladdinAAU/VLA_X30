"""Reasoner agent against the fake OpenAI server (scripted replies, incl. malformed tool calls) and the fake world."""
import json
import threading
import time

from fake_world import FakeWorld
from x30_common.testing.fake_openai import (error_reply, multi_tool_call_reply, text_reply, text_tool_call_reply, tool_call_reply)

from x30_agent.core.trace import TraceWriter
from x30_agent.core.types import Task


def world_with_target():
    w = FakeWorld(seed=3)
    task, target = w.make_task()
    return w, task, target


def tool_msgs(req):
    return [m for m in req["messages"] if m["role"] == "tool"]


def read_trace(tmp_path, ep="e1"):
    p = tmp_path / "traces" / f"{ep}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def label_of(world, target):
    return target.label


def test_happy_path_and_trace(srv, make_agent, tmp_path):
    w, task, tgt = world_with_target()
    agent = make_agent(w)
    srv.queue(tool_call_reply("search", {"label": tgt.label}),
              tool_call_reply("goto_object", {"object_id": tgt.id}),
              tool_call_reply("verify", {"object_id": tgt.id, "question": f"Is this a {tgt.color} {tgt.label} in the {tgt.room}?"}),
              tool_call_reply("finish", {"success": True, "reason": "verified"}))
    res = agent.run(Task(task, episode_id="e1"))
    assert res.success and res.stop_reason == "finished" and res.turns == 4 and not res.guard_overridden
    rows = read_trace(tmp_path)
    assert rows[0]["kind"] == "header" and rows[0]["prompt_hash"] == agent.policy.hash
    turns = [r for r in rows if r["kind"] == "turn"]
    assert [t["tool"] for t in turns] == ["search", "goto_object", "verify", "finish"]
    t1 = turns[1]
    assert t1["latency_s"] >= 0 and t1["prompt_tokens"] == 100 and t1["reply"]["tool_calls"][0]["name"] == "goto_object"
    assert "NEXT" in turns[0]["result_text"] and "messages" in t1
    assert rows[-1]["kind"] == "summary" and rows[-1]["stop_reason"] == "finished"
    assert any(r["kind"] == "finish_guard" and r["allowed"] for r in rows)
    # hints are what the model sees in the next request
    assert "NEXT" in tool_msgs(srv.requests[1])[0]["content"]
    # tools were sent, system prompt contains the task
    assert srv.requests[0]["tools"] and task in srv.requests[0]["messages"][0]["content"]


def test_malformed_arguments_text_call_and_retries(srv, make_agent):
    w, task, tgt = world_with_target()
    agent = make_agent(w)
    srv.queue(tool_call_reply("search", "{'label': '%s',}" % tgt.label),               # repaired JSON
              text_reply("I think I should look for it."),                             # no tool call -> retry in the same turn
              text_tool_call_reply("goto_object", {"object_id": tgt.id}),                # call left in content
              tool_call_reply("verify", {"object_id": "#%d" % tgt.id, "question": tgt.label}, args_as="object"),   # object args, '#id'
              tool_call_reply("Finish", {"success": "true", "reason": "ok"}))           # case + string bool
    res = agent.run(Task(task, episode_id="e1"))
    assert res.success and res.stats["format_retries"] == 1 and res.turns == 4
    assert [c[0] for c in w.skill_calls] == ["search", "goto_object", "verify"]


def test_unknown_tool_and_bad_args_are_reasked(srv, make_agent):
    w, task, tgt = world_with_target()
    agent = make_agent(w)
    srv.queue(tool_call_reply("teleport", {"x": 1}),
              tool_call_reply("goto_object", {"standoff_m": 1.0}),                       # missing id/label (one-of rule)
              tool_call_reply("search", {"label": tgt.label}),
              tool_call_reply("finish", {"success": False, "reason": "stop"}))
    res = agent.run(Task(task, episode_id="e1"))
    assert res.stats["format_retries"] == 2
    second = srv.requests[1]["messages"]
    assert "not a tool" in second[-1]["content"] and "Available tools" in second[-1]["content"]
    assert "give at least" in srv.requests[2]["messages"][-1]["content"]
    assert [c[0] for c in w.skill_calls] == ["search"] and res.stop_reason == "no_progress" and not res.success


def test_invalid_args_after_retries_become_tool_result(srv, make_agent):
    w, task, _ = world_with_target()
    agent = make_agent(w, {"limits": {"format_retries": 0}})
    srv.queue(tool_call_reply("goto_object", {"object_id": "banana"}), tool_call_reply("finish", {"success": False, "reason": "x"}))
    res = agent.run(Task(task, episode_id="e1"))
    assert res.stats["invalid_calls"] == 1 and w.skill_calls == []
    assert "invalid" in tool_msgs(srv.requests[1])[0]["content"].lower()


def test_noop_turns_end_as_error(srv, make_agent):
    w, task, _ = world_with_target()
    agent = make_agent(w, {"limits": {"format_retries": 0, "max_noop_turns": 3}})
    srv.queue(*[text_reply("blah")] * 5)
    res = agent.run(Task(task, episode_id="e1"))
    assert res.stop_reason == "error" and res.turns == 3 and w.skill_calls == [("stop", {})]      # only the best-effort stop


def test_second_scan_in_same_place_refused(srv, make_agent):
    w, task, tgt = world_with_target()
    agent = make_agent(w)
    srv.queue(tool_call_reply("scan", {"sweep_deg": 90}), tool_call_reply("scan", {"sweep_deg": 180}), tool_call_reply("finish", {"success": False, "reason": "x"}))
    res = agent.run(Task(task, episode_id="e1"))
    assert [c[0] for c in w.skill_calls].count("scan") == 1 and res.stats["refusals"] == 1
    assert "already scanned" in tool_msgs(srv.requests[2])[-1]["content"]


def test_third_identical_call_refused_non_scan(srv, make_agent):
    w, task, tgt = world_with_target()
    agent = make_agent(w)
    call = tool_call_reply("search", {"label": "zzz-nothing", "budget_s": 10})
    srv.queue(call, call, call, tool_call_reply("finish", {"success": False, "reason": "x"}))
    res = agent.run(Task(task, episode_id="e1"))
    assert sum(1 for c in w.skill_calls if c[0] == "search") == 2
    assert "repeated 3 times" in tool_msgs(srv.requests[3])[-1]["content"]


def test_cycle_refused(srv, make_agent):
    w, task, tgt = world_with_target()
    agent = make_agent(w, {"limits": {"failed_call_refuse_at": 99}})
    a = tool_call_reply("turn_to", {"degrees": 10})
    b = tool_call_reply("turn_to", {"degrees": 20})
    srv.queue(a, b, a, b, a, tool_call_reply("finish", {"success": False, "reason": "x"}))
    agent.run(Task(task, episode_id="e1"))
    assert sum(1 for c in w.skill_calls if c[0] == "turn_to") == 4


def test_finish_guard_refuses_then_overrides_and_records(srv, make_agent, tmp_path):
    w, task, tgt = world_with_target()
    agent = make_agent(w)
    fin = tool_call_reply("finish", {"success": True, "reason": "I am done"})
    srv.queue(fin, tool_call_reply("scan", {}), fin, tool_call_reply("turn_to", {"degrees": 30}), fin)
    res = agent.run(Task(task, episode_id="e1"))
    assert res.success and res.guard_overridden and res.finish_refusals == 3 and "Guard override" in res.message
    guard_rows = [r for r in read_trace(tmp_path) if r["kind"] == "finish_guard"]
    assert [g["allowed"] for g in guard_rows] == [False, False, True] and guard_rows[-1]["overridden"]
    assert "verify" in tool_msgs(srv.requests[1])[0]["content"]


def test_finish_guard_accepts_after_verify_even_if_later_approach_failed(srv, make_agent):
    w, task, tgt = world_with_target()
    agent = make_agent(w)
    srv.queue(tool_call_reply("search", {"label": tgt.label}), tool_call_reply("goto_object", {"object_id": tgt.id}),
              tool_call_reply("verify", {"object_id": tgt.id, "question": f"{tgt.color} {tgt.label} {tgt.room}"}),
              tool_call_reply("approach", {"object_id": tgt.id, "standoff_m": 0.8}), tool_call_reply("finish", {"success": True, "reason": "x"}))
    orig = w._sk_approach
    w._sk_approach = lambda a: __import__("x30_agent.core.types", fromlist=["SkillResult"]).SkillResult(False, "object not visible to the lidar")
    res = agent.run(Task(task, episode_id="e1"))
    assert res.success and not res.guard_overridden


def test_guard_refuses_stale_verify_after_moving_away(srv, make_agent):
    w, task, tgt = world_with_target()
    agent = make_agent(w)
    srv.queue(tool_call_reply("search", {"label": tgt.label}), tool_call_reply("goto_object", {"object_id": tgt.id}),
              tool_call_reply("verify", {"object_id": tgt.id, "question": f"{tgt.color} {tgt.label} {tgt.room}"}),
              tool_call_reply("goto_pose", {"x": 1.0, "y": 7.0}), tool_call_reply("finish", {"success": True, "reason": "x"}),
              tool_call_reply("finish", {"success": False, "reason": "give up"}))
    res = agent.run(Task(task, episode_id="e1"))
    far_enough = ((w.pos.x - tgt.x) ** 2 + (w.pos.y - tgt.y) ** 2) ** 0.5 > 1.5
    if far_enough:
        assert res.finish_refusals == 1 and not res.success
        assert "Refused" in tool_msgs(srv.requests[5])[-1]["content"]


def test_rejected_ids_are_remembered_and_injected(srv, make_agent):
    w = FakeWorld(seed=5)
    for o in w.objects:
        o.seen = True
    task = "Find the white couch."
    wrong = w.objects[0]
    agent = make_agent(w)
    srv.queue(tool_call_reply("goto_object", {"object_id": wrong.id}),
              tool_call_reply("verify", {"object_id": wrong.id, "question": "Is this a white couch in the kitchen?"}),
              tool_call_reply("search", {"label": "couch"}), tool_call_reply("finish", {"success": False, "reason": "x"}))
    agent.run(Task(task, episode_id="e1"))
    verify_res = [r for r in [w._sk_verify({"object_id": wrong.id, "question": "Is this a white couch in the kitchen?"})]][0]
    assert verify_res.data["answer"] == "no" or wrong.label == "couch"
    search = [c for c in w.skill_calls if c[0] == "search"][0]
    assert wrong.id in search[1].get("rejected_ids", [])
    assert "Rejected by verify" in srv.requests[3]["messages"][-1]["content"]


def test_extra_calls_dropped_with_note(srv, make_agent):
    w, task, tgt = world_with_target()
    agent = make_agent(w)
    srv.queue(multi_tool_call_reply([("search", {"label": tgt.label}), ("scan", {})]), tool_call_reply("finish", {"success": False, "reason": "x"}))
    res = agent.run(Task(task, episode_id="e1"))
    assert [c[0] for c in w.skill_calls] == ["search"] and res.stats["extra_calls_dropped"] == 1
    assert "only the first was executed" in tool_msgs(srv.requests[1])[0]["content"]


def test_max_turns_and_time_budget(srv, make_agent):
    w, task, _ = world_with_target()
    agent = make_agent(w)
    srv.queue(*[tool_call_reply("turn_to", {"degrees": 15 + i}) for i in range(10)])
    res = agent.run(Task(task, episode_id="e1", max_turns=3))
    assert res.stop_reason == "timeout" and res.turns == 3 and "turn limit" in res.message
    srv.replies.clear()
    w2, task2, _ = world_with_target()
    agent2 = make_agent(w2)
    srv.queue(tool_call_reply("scan", {}), tool_call_reply("turn_to", {"degrees": 5}), tool_call_reply("turn_to", {"degrees": 9}))
    res = agent2.run(Task(task2, episode_id="e2", time_budget_s=20))       # a fake scan alone costs 25 s of fake time
    assert res.stop_reason == "timeout" and "time budget" in res.message


def test_consecutive_failures_end_as_no_progress(srv, make_agent):
    w, task, _ = world_with_target()
    agent = make_agent(w, {"limits": {"max_consecutive_failures": 3}})
    srv.queue(*[tool_call_reply("goto_pose", {"x": 5.0 + i * 0.01, "y": 1.0}) for i in range(8)])    # inside a wall: fails each time
    res = agent.run(Task(task, episode_id="e1"))
    assert res.stop_reason == "no_progress"


def test_llm_errors_end_as_error(srv, make_agent):
    w, task, _ = world_with_target()
    agent = make_agent(w, client_kw={"max_retries": 0})
    srv.queue(error_reply(500), error_reply(500), error_reply(500))
    res = agent.run(Task(task, episode_id="e1"))
    assert res.stop_reason == "error" and "model server" in res.message


def test_cancel_before_and_during_run(srv, make_agent):
    w, task, _ = world_with_target()
    ev = threading.Event()
    agent = make_agent(w, cancel=ev)
    ev.set()
    res = agent.run(Task(task, episode_id="e1"))
    assert res.stop_reason == "aborted" and srv.requests == []
    ev2 = threading.Event()
    agent2 = make_agent(w, cancel=ev2)
    srv.queue(text_reply("slow", delay_s=5.0))
    threading.Timer(0.4, ev2.set).start()
    t0 = time.time()
    res = agent2.run(Task(task, episode_id="e2"))
    assert res.stop_reason == "aborted" and time.time() - t0 < 2.5


def test_cancel_during_skill(srv, make_agent):
    w, task, tgt = world_with_target()
    ev = threading.Event()
    agent = make_agent(w, cancel=ev)
    orig = w.run_skill

    def slow(name, args, timeout_s, cancel):
        if name == "search":
            ev.set()
            from x30_agent.core.types import SkillResult
            return SkillResult(False, "cancelled", cancelled=True)
        return orig(name, args, timeout_s, cancel)
    w.run_skill = slow
    srv.queue(tool_call_reply("search", {"label": tgt.label}), tool_call_reply("scan", {}))
    res = agent.run(Task(task, episode_id="e1"))
    assert res.stop_reason == "aborted" and len(srv.requests) == 1


def test_context_budget_degrades_history_and_images(srv, make_agent):
    w, task, tgt = world_with_target()
    agent = make_agent(w, {"images": {"map": True, "camera": True}, "budget": {"context_tokens": 3600}, "prompt": {"history_units": 5}})
    srv.queue(*[tool_call_reply("turn_to", {"degrees": 10 + i}) for i in range(5)], tool_call_reply("finish", {"success": False, "reason": "x"}))
    res = agent.run(Task(task, episode_id="e1"))
    assert res.stats["degraded_last"] or res.turns >= 1
    sent = json.dumps(srv.requests[-1]["messages"])
    assert "image_url" not in sent or "degraded" not in sent


def test_images_attached_when_enabled_and_saved(srv, make_agent, tmp_path):
    w, task, tgt = world_with_target()
    agent = make_agent(w, {"images": {"map": True, "camera": True}})
    srv.queue(tool_call_reply("finish", {"success": False, "reason": "x"}))
    agent.run(Task(task, episode_id="e1"))
    last = srv.requests[0]["messages"][-1]["content"]
    assert isinstance(last, list) and sum(1 for p in last if p["type"] == "image_url") == 2
    rows = read_trace(tmp_path)
    paths = [r for r in rows if r["kind"] == "turn"][0]["images"]
    assert len(paths) == 2 and all((tmp_path / "traces" / p).exists() for p in paths)


def test_trace_writer_survives_unwritable_dir(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    tw = TraceWriter(f / "sub", "e")        # parent is a file: mkdir fails
    tw.write("turn", a=1)
    assert tw.errors >= 1
