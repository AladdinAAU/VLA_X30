#!/usr/bin/env python3
"""Integration check of the agent nodes against tests/ros/fake_stack.py (run inside the container, see run_ros_check.sh).

Sends Navigate goals to /x30/agent/run and /x30/agent/follow and checks results, feedback, cancel latency, rejection of a second goal and
bad params. The model is the rule-based mock, so this checks plumbing, not quality. Writes JSON to the path in argv[1]."""
import json
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from x30_msgs.action import Navigate
from x30_msgs.srv import Json

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ros_check.json"
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)


class Client(Node):
    def __init__(self):
        super().__init__("agent_check")
        self.run = ActionClient(self, Navigate, "/x30/agent/run")
        self.follow = ActionClient(self, Navigate, "/x30/agent/follow")
        self.jclients = {n: self.create_client(Json, n) for n in ("/fake/reset", "/fake/state")}

    def call(self, name, payload, timeout=10.0):
        ev = threading.Event()
        fut = self.jclients[name].call_async(Json.Request(request=json.dumps(payload)))
        fut.add_done_callback(lambda _f: ev.set())
        assert ev.wait(timeout), f"timeout {name}"
        return json.loads(fut.result().response)

    def send(self, cli, instruction, label, mode, params=None, timeout_s=120.0, episode="e"):
        g = Navigate.Goal()
        g.mode, g.instruction, g.target_label, g.timeout_s, g.episode_id = mode, instruction, label, float(timeout_s), episode
        g.params_json = json.dumps(params) if params is not None else ""
        fb = []
        ev = threading.Event()
        fut = cli.send_goal_async(g, feedback_callback=lambda m: fb.append(m.feedback.status))
        fut.add_done_callback(lambda _f: ev.set())
        assert ev.wait(10), "goal not answered"
        h = fut.result()
        return h, fb

    def result(self, h, timeout=300.0):
        ev = threading.Event()
        rf = h.get_result_async()
        rf.add_done_callback(lambda _f: ev.set())
        assert ev.wait(timeout), "no result"
        return rf.result()


def main():
    rclpy.init()
    c = Client()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(c)
    threading.Thread(target=ex.spin, daemon=True).start()
    for cli, n in ((c.run, "/x30/agent/run"), (c.follow, "/x30/agent/follow")):
        check(f"action server {n} available", cli.wait_for_server(timeout_sec=30))
    if not all(x["ok"] for x in checks):
        return finish()
    for n in c.jclients.values():
        n.wait_for_service(timeout_sec=10)

    # 1. object_nav with the mock model, three worlds, ground truth from the fake world
    for seed in (11, 12, 13):
        info = c.call("/fake/reset", {"seed": seed, "mode": "object_nav", "skill_delay_s": 0.2})
        t0 = time.time()
        h, fb = c.send(c.run, info["instruction"], info["target_label"], "object_nav", {"max_turns": 25}, 120, f"on{seed}")
        r = c.result(h)
        st = c.call("/fake/state", {})
        check(f"object_nav mock seed {seed}", h.accepted and r.result.success and st["success"],
              f"stop={r.result.stop_reason} turns={r.result.agent_turns} dist={st['dist_to_target']:.2f} wall={time.time() - t0:.1f}s fb={len(fb)} msg={r.result.message[:60]}")
    check("feedback is published", len(fb) > 0, f"{len(fb)} messages, last: {fb[-1] if fb else ''}")

    # 2. velocity_follow with the mock follower
    for seed in (21, 22, 23):
        info = c.call("/fake/reset", {"seed": seed, "mode": "velocity_follow", "skill_delay_s": 0.05})
        t0 = time.time()
        h, fb = c.send(c.follow, info["instruction"], "", "velocity_follow", {"max_turns": 60}, 240, f"vf{seed}")
        r = c.result(h)
        st = c.call("/fake/state", {})
        check(f"velocity_follow mock seed {seed}", h.accepted and r.result.stop_reason in ("finished", "no_progress", "safety", "timeout") and r.result.agent_turns > 0,
              f"stop={r.result.stop_reason} success={r.result.success} turns={r.result.agent_turns} dist={st['dist_to_target']:.2f} gt={st['success']} wall={time.time() - t0:.1f}s")

    # 3. baselines through params_json
    info = c.call("/fake/reset", {"seed": 31, "mode": "object_nav", "skill_delay_s": 0.05})
    h, _ = c.send(c.run, info["instruction"], info["target_label"], "object_nav", {"agent_kind": "scripted_search"}, 120, "ss")
    r = c.result(h)
    st = c.call("/fake/state", {})
    check("baseline scripted_search via params_json", r.result.stop_reason in ("finished", "no_progress", "timeout") and r.result.agent_turns >= 0,
          f"success={r.result.success} gt={st['success']} stop={r.result.stop_reason} msg={r.result.message[:60]}")
    h, _ = c.send(c.follow, "go", "", "velocity_follow", {"agent_kind": "scripted_search"}, 30, "bad_kind")
    r = c.result(h)
    check("follow rejects agent_kind scripted_search with an error result", r.result.stop_reason == "error", r.result.message[:80])

    # 4. bad params_json, second goal while busy
    g = Navigate.Goal()
    g.mode, g.instruction, g.params_json, g.timeout_s = "object_nav", "x", "{not json", 30.0
    ev = threading.Event()
    f = c.run.send_goal_async(g)
    f.add_done_callback(lambda _f: ev.set())
    ev.wait(10)
    hh = f.result()
    r = c.result(hh) if hh.accepted else None
    check("bad params_json gives an error result, not a hang", r is not None and r.result.stop_reason == "error", r.result.message[:60] if r else "rejected")

    info = c.call("/fake/reset", {"seed": 41, "mode": "object_nav", "skill_delay_s": 1.0, "model_delay_s": 1.0})
    h1, _ = c.send(c.run, info["instruction"], info["target_label"], "object_nav", {"max_turns": 25}, 120, "busy1")
    time.sleep(0.5)
    h2, _ = c.send(c.run, info["instruction"], info["target_label"], "object_nav", {}, 120, "busy2")
    check("second goal while an episode runs is rejected", h1.accepted and not h2.accepted)

    # 5. cancel latency, several moments of the loop
    for label, kw, wait in (("during model call", {"skill_delay_s": 0.2, "model_delay_s": 8.0}, 2.0),
                            ("during a skill", {"skill_delay_s": 20.0, "model_delay_s": 0.0}, 3.0)):
        # make sure the previous goal is gone
        if label == "during model call":
            h1.cancel_goal_async()
            c.result(h1)
        info = c.call("/fake/reset", {"seed": 51, "mode": "object_nav", **kw})
        h, _ = c.send(c.run, info["instruction"], info["target_label"], "object_nav", {"max_turns": 25}, 300, "cancel")
        time.sleep(wait)
        t0 = time.time()
        cf = h.cancel_goal_async()
        r = c.result(h, 30)
        dt = time.time() - t0
        check(f"cancel {label} completes within 5 s", dt < 5.0 and not r.result.success, f"{dt:.2f} s, stop={r.result.stop_reason}, status={r.status}")
    info = c.call("/fake/reset", {"seed": 52, "mode": "velocity_follow", "skill_delay_s": 20.0})
    h, _ = c.send(c.follow, info["instruction"], "", "velocity_follow", {}, 300, "cancel_f")
    time.sleep(3.0)
    t0 = time.time()
    h.cancel_goal_async()
    r = c.result(h, 30)
    dt = time.time() - t0
    check("cancel follower during a primitive completes within 5 s", dt < 5.0, f"{dt:.2f} s, stop={r.result.stop_reason}")

    # 5b. eval-runner keys in params_json: trace_dir and seed
    info = c.call("/fake/reset", {"seed": 71, "mode": "object_nav", "skill_delay_s": 0.05})
    h, _ = c.send(c.run, info["instruction"], info["target_label"], "object_nav", {"max_turns": 25, "trace_dir": "/tmp/eval_traces", "seed": 5}, 120, "eval/ep 1")
    r = c.result(h)
    import os
    files = sorted(os.listdir("/tmp/eval_traces")) if os.path.isdir("/tmp/eval_traces") else []
    check("params_json trace_dir is honoured", any(f.endswith(".jsonl") for f in files) and r.result.success, f"files={files[:3]}")
    # a give-up (the model's finish(success=false)) must not be reported as 'finished'
    info = c.call("/fake/reset", {"seed": 72, "mode": "object_nav", "skill_delay_s": 0.05})
    h, _ = c.send(c.run, "Find the bicycle.", "bicycle", "object_nav", {"max_turns": 12}, 120, "absent")
    r = c.result(h)
    check("object that does not exist: not reported as finished", (not r.result.success) and r.result.stop_reason != "finished", f"stop={r.result.stop_reason} turns={r.result.agent_turns} msg={r.result.message[:60]}")

    # 6. the node is usable after a cancel
    info = c.call("/fake/reset", {"seed": 61, "mode": "object_nav", "skill_delay_s": 0.1})
    h, _ = c.send(c.run, info["instruction"], info["target_label"], "object_nav", {"max_turns": 25}, 120, "after")
    r = c.result(h)
    check("episode after a cancel works", h.accepted and r.result.success, f"stop={r.result.stop_reason}")
    finish()


def finish():
    ok = all(x["ok"] for x in checks)
    json.dump({"passed": sum(x["ok"] for x in checks), "total": len(checks), "checks": checks}, open(OUT, "w"), indent=1)
    print(f"{sum(x['ok'] for x in checks)}/{len(checks)} checks passed", flush=True)
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
