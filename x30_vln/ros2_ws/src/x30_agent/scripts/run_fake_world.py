#!/usr/bin/env python3
"""Run the agents end to end in the pure-python FAKE world (tests/fake_world.py) against a real or mock VLM server.

    python scripts/run_fake_world.py --mode object_nav --agent llm --n 20 --url http://localhost:8000/v1 --out results/agent_live/fake_objnav
    python scripts/run_fake_world.py --mode velocity_follow --agent llm --n 20 --max-turns 40 --images --out ...
    python scripts/run_fake_world.py --mode object_nav --agent scripted_search --n 20 --out ...

The success rate printed here is for the FAKE environment only. It says nothing about the robot.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT.parent / "x30_common")]

from fake_world import FakeWorld, ground_truth_success  # noqa: E402
from x30_common import VLMClient, load_models_config  # noqa: E402

from x30_agent.core.follower import FollowerAgent  # noqa: E402
from x30_agent.core.policy import load_policy  # noqa: E402
from x30_agent.core.reasoner import ReasonerAgent  # noqa: E402
from x30_agent.core.skills import load_registry  # noqa: E402
from x30_agent.core.types import Task  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="object_nav", choices=["object_nav", "velocity_follow"])
    ap.add_argument("--agent", default="llm", choices=["llm", "scripted_search"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(ROOT.parents[2] / "results" / "agent_live" / "fake_objnav"))
    ap.add_argument("--images", action="store_true", help="send the annotated map and camera frame")
    ap.add_argument("--route", action="store_true", help="velocity_follow: give a metric route description instead of just the target")
    ap.add_argument("--absent", action="store_true", help="object_nav: ask for an object that is not in the world; success = the agent gives up (claims success=false)")
    ap.add_argument("--noise", type=float, default=0.0, help="probability that the fake verify answers wrongly")
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--tool-choice", default=None)
    ap.add_argument("--overrides", default="{}", help="JSON policy overrides")
    a = ap.parse_args()

    configs = ROOT.parents[2] / "configs"
    registry = load_registry(configs / "skills.yaml")
    ov = json.loads(a.overrides)
    if a.images:
        ov.setdefault("images", {}).update({"map": True, "camera": True})
    if a.tool_choice:
        ov.setdefault("model", {})["tool_choice"] = a.tool_choice
    policy = load_policy(a.mode, configs / "agent", ov)
    out = Path(a.out)
    rows = []
    client = None
    if a.agent == "llm":
        cfg = load_models_config()
        client = VLMClient.from_config("reasoner", cfg, **({"base_url": a.url} if a.url else {}), **({"model": a.model} if a.model else {}))
        if not client.health():
            print("model server not reachable at", client.base_url)
            return 2
    for i in range(a.n):
        world = FakeWorld(seed=a.seed0 + i, noise=a.noise)
        instr, target = world.make_task()
        if a.absent:
            instr = "Find the bicycle and stop next to it."
        if a.mode == "velocity_follow":     # the follower has no map
            instr = route_instruction(world, target) if a.route else f"Walk to the {target.color} {target.label} in the {target.room} and stop next to it."
        task = Task(instr, mode=a.mode, episode_id=f"fake_{a.seed0 + i:04d}", max_turns=a.max_turns, time_budget_s=900)
        t0 = time.time()
        if a.agent == "llm" and a.mode == "velocity_follow":
            agent = FollowerAgent(client, world, policy, trace_dir=str(out / "traces"), models_hash=load_models_config().hash())
            res = agent.run(task)
        elif a.agent == "llm":
            agent = ReasonerAgent(client, world, registry, policy, trace_dir=str(out / "traces"), models_hash=load_models_config().hash())
            res = agent.run(task)
        else:
            from x30_agent.core.baselines import ScriptedSearch
            res = ScriptedSearch(world, registry, policy, trace_dir=str(out / "traces")).run(task, target_label=_label_from(instr, world))
        gt = (not res.success) if a.absent else ground_truth_success(world, target)
        row = {"episode": task.episode_id, "task": instr, "claimed": res.success, "gt_success": bool(gt) if a.absent else bool(res.success and gt), "gt_at_end": gt,
               "stop_reason": res.stop_reason, "turns": res.turns, "fake_time_s": round(res.duration_s, 1), "wall_s": round(time.time() - t0, 1),
               "path_m": round(world.path_len, 1), "guard_overridden": res.guard_overridden, "finish_refusals": res.finish_refusals,
               "format_retries": res.stats.get("format_retries"), "noop": res.stats.get("noop_turns"), "refusals": res.stats.get("refusals"),
               "invalid": res.stats.get("invalid_calls"), "llm": res.stats.get("llm", {}), "message": res.message[:120],
               "dist_to_target": round(math.hypot(world.pos.x - target.x, world.pos.y - target.y), 2)}
        rows.append(row)
        print(f"{row['episode']} {'OK ' if row['gt_success'] else 'FAIL'} claimed={row['claimed']} {row['stop_reason']} turns={row['turns']} "
              f"wall={row['wall_s']}s | {instr}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    n = len(rows)
    summ = {"environment": "FAKE pure-python world, not the robot", "mode": a.mode, "agent": a.agent, "episodes": n,
            "sr": sum(r["gt_success"] for r in rows) / n, "claimed_success": sum(r["claimed"] for r in rows) / n,
            "false_claims": sum(1 for r in rows if r["claimed"] and not r["gt_at_end"]),
            "stop_reasons": {k: sum(1 for r in rows if r["stop_reason"] == k) for k in {r["stop_reason"] for r in rows}},
            "mean_turns": sum(r["turns"] for r in rows) / n, "guard_overrides": sum(r["guard_overridden"] for r in rows),
            "mean_wall_s": sum(r["wall_s"] for r in rows) / n,
            "tokens": {"prompt": sum(r["llm"].get("prompt_tokens", 0) for r in rows), "completion": sum(r["llm"].get("completion_tokens", 0) for r in rows)},
            "llm_calls": sum(r["llm"].get("calls", 0) for r in rows), "format_retries": sum(r["format_retries"] or 0 for r in rows),
            "invalid_calls": sum(r["invalid"] or 0 for r in rows), "noop_turns": sum(r["noop"] or 0 for r in rows),
            "refusals": sum(r["refusals"] or 0 for r in rows), "options": vars(a), "prompt_hash": policy.hash}
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return 0


def route_instruction(world: FakeWorld, target) -> str:
    """R2R-style metric route description along the world's A* path (string-pulled), ending near the target.
    Ground truth of the fake world is used only to WRITE the instruction; the agent never sees the map."""
    pts = world._astar((world.pos.x, world.pos.y), (target.x, target.y), True) or [(target.x, target.y)]
    way, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not world._line_free(pts[i], pts[j]):
            j -= 1
        way.append(pts[j])
        i = j
    steps, heading = [], world.pos.yaw
    for k in range(1, len(way)):
        dx, dy = way[k][0] - way[k - 1][0], way[k][1] - way[k - 1][1]
        dist, hd = math.hypot(dx, dy), math.atan2(dy, dx)
        if k == len(way) - 1:
            dist -= 1.0                                   # stop about one metre short
        turn = math.degrees((hd - heading + math.pi) % (2 * math.pi) - math.pi)
        heading = hd
        if abs(turn) >= 15:
            steps.append(f"turn {'left' if turn > 0 else 'right'} about {int(round(abs(turn) / 15) * 15)} degrees")
        if dist >= 0.5:
            steps.append(f"walk forward about {round(dist * 2) / 2:g} meters")
    return ", then ".join(steps).capitalize() + f". The {target.color} {target.label} is then in front of you; stop next to it."


def _label_from(instr: str, world: FakeWorld) -> str:
    from fake_world import LABELS
    return next((l for l in sorted(LABELS, key=len, reverse=True) if l in instr.lower()), "")


if __name__ == "__main__":
    sys.exit(main())
