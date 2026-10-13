#!/usr/bin/env python3
"""LIVE measurements against the real VLM server: tool-call format error rate, decision quality, latency with/without images.

Single-turn prompts built from hand-written state templates (invented maps; not the robot). Each state has a known best next
step. The raw first reply of the model is scored (no retries), so the error rates are those of the model, not of the harness.

    X30_REASONER_URL=http://localhost:8000/v1 python scripts/live_eval.py --out results/agent_live/live_eval
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT.parent / "x30_common")]

from fake_world import FakeWorld  # noqa: E402
from x30_common import VLMClient, load_models_config  # noqa: E402
from x30_common.images import user_message  # noqa: E402
from x30_common.toolschema import closest_tool_name, coerce_arguments, find_tool, is_fatal  # noqa: E402

from x30_agent.core.hints import HintEngine  # noqa: E402
from x30_agent.core.memory import HarnessMemory  # noqa: E402
from x30_agent.core.policy import load_policy  # noqa: E402
from x30_agent.core.prompts import build_system_prompt  # noqa: E402
from x30_agent.core.skills import load_registry, validate_one_of  # noqa: E402
from x30_agent.core.tools import build_toolset  # noqa: E402

LABELS = ["chair", "fire extinguisher", "potted plant", "trash can", "couch", "refrigerator", "table", "tv"]
COLORS = ["red", "blue", "green", "white", "black"]
ROOMS = ["living room", "kitchen", "office"]


def make_states(rng: random.Random, per_type: int = 9) -> list[dict]:
    """Each state: task, map text, notes, optional last tool exchange, accepted next tools, expected id (or None)."""
    S = []
    for k in range(per_type):
        lab, col, room = rng.choice(LABELS), rng.choice(COLORS), rng.choice(ROOMS)
        oid = rng.randint(2, 15)
        dist = round(rng.uniform(1.5, 7.0), 1)
        line = f"#{oid} {lab} ({rng.uniform(1, 11):.1f}, {rng.uniform(1, 7):.1f}) {dist} m, bearing {rng.randint(-90, 90)}, {room}, {col}"
        task = rng.choice([f"Find the {lab} and stop next to it.", f"Go to the {col} {lab} in the {room}.", f"Find the {col} {lab}."])
        other = rng.choice([l for l in LABELS if l != lab])
        filler = f"#{oid + 20} {other} ({rng.uniform(1, 11):.1f}, {rng.uniform(1, 7):.1f}) {rng.uniform(2, 8):.1f} m, bearing {rng.randint(-90, 90)}, {rng.choice(ROOMS)}, {rng.choice(COLORS)}"
        S.append({"type": "A_empty_map", "task": task, "map": "0 objects mapped; 3% explored.", "accept": {"search", "scan", "explore"}, "id": None, "label": lab})
        S.append({"type": "B_target_in_map", "task": task.replace(f"{col} ", "") if "Go to" not in task else task, "map": f"2 objects mapped.\n{filler}\n{line}",
                  "accept": {"goto_object", "approach"}, "id": oid, "label": lab})
        S.append({"type": "C_arrived_needs_verify", "task": task, "map": f"2 objects mapped.\n{line}\n{filler}",
                  "exchange": ("goto_object", {"object_id": oid}, f"success. Arrived 1.1 m from #{oid} {lab}.\ndata: object_id={oid} distance_m=1.1\nYou are next to #{oid}. NEXT: verify(object_id={oid}, question=<ask about the task's colour/room/relation>)."),
                  "accept": {"verify"}, "id": oid, "label": lab})
        S.append({"type": "D_verified_yes_close", "task": task, "map": f"2 objects mapped.\n#{oid} {lab} (3.0, 2.0) 1.1 m, bearing 0, {room}, {col}\n{filler}",
                  "exchange": ("verify", {"object_id": oid, "question": task}, f"success. Looks like a match.\ndata: answer=yes plausibility=93 object_id={oid} distance_m=1.1\nVerified: #{oid} matches. NEXT: finish(success=true, reason) if you are within 1.5 m of it and the task is satisfied; otherwise approach it first."),
                  "accept": {"finish"}, "id": None, "label": lab, "finish_success": True})
        bad, good = oid, oid + 1
        S.append({"type": "E_rejected_other_known", "task": task, "map": f"3 objects mapped.\n#{bad} {lab} (3.0, 2.0) 1.1 m, bearing 0, {room}, {col}\n#{good} {lab} (6.0, 4.0) 4.5 m, bearing 40, {room}, {col}\n{filler}",
                  "notes": f"Rejected by verify (never use these ids): #{bad}",
                  "exchange": ("verify", {"object_id": bad, "question": task}, f"success. Looks different.\ndata: answer=no plausibility=8 object_id={bad} distance_m=1.1\nRejected: #{bad} is not the target and is excluded from now on. Other unchecked candidates of the same kind are already mapped: #{good}. NEXT: goto_object(object_id={good}), then verify it."),
                  "accept": {"goto_object", "approach"}, "id": good, "label": lab})
        S.append({"type": "F_exhausted_give_up", "task": task, "map": f"2 objects mapped; 100% explored.\n{filler}",
                  "notes": f"Failed: search(label={lab}): Nothing found; the map is exhausted.; search(label=synonym): Nothing found; the map is exhausted.",
                  "exchange": ("search", {"label": lab + "s"}, "failure. Nothing found for it; the map is exhausted.\ndata: candidates=[] explored=True exhausted=True\nNothing (new) found and nothing left to explore. NEXT: try a synonym label once, or finish(success=false, reason)."),
                  "accept": {"finish"}, "lenient": {"finish", "search"}, "id": None, "label": lab, "finish_success": False})
        S.append({"type": "G_search_failed_not_exhausted", "task": task, "map": f"1 objects mapped; 20% explored.\n{filler}",
                  "exchange": ("search", {"label": lab, "budget_s": 10}, "failure. Search budget ended without a candidate.\ndata: candidates=[] explored=True exhausted=False\nNot found yet. NEXT: explore (or scan in a new place), then search again."),
                  "accept": {"explore", "scan", "search"}, "id": None, "label": lab})
        S.append({"type": "H_unsure_move_closer", "task": task, "map": f"2 objects mapped.\n#{oid} {lab} (3.0, 2.0) 3.6 m, bearing 10, {room}, {col}\n{filler}",
                  "exchange": ("verify", {"object_id": oid, "question": task}, f"success. Too far to tell.\ndata: answer=unsure plausibility=40 object_id={oid} distance_m=3.6\nUnsure (poor view). NEXT: approach(object_id={oid}) or turn_to it and verify again; if still unsure twice, look at other candidates."),
                  "accept": {"approach", "goto_object", "turn_to"}, "id": oid, "label": lab})
    return S


def build_messages(state: dict, system: str, hints_pol) -> list[dict]:
    mem = HarnessMemory()
    text = [f"[Turn {rng_turn(state)}/30, 40s of 300s used]", f"Task: {state['task']}", "Robot pose (map): x=1.20 y=3.40 heading=90 deg",
            "Map summary:\n" + state["map"]]
    if state.get("notes"):
        text.append("Harness notes:\n" + state["notes"])
    text.append("Choose the next tool call.")
    msgs = [{"role": "system", "content": system + f"\n\nTask: {state['task']}"}]
    if state.get("exchange"):
        name, args, result = state["exchange"]
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
                 {"role": "tool", "tool_call_id": "c1", "content": result}]
    msgs.append({"role": "user", "content": "\n".join(text)})
    _ = mem, hints_pol
    return msgs


def rng_turn(state: dict) -> int:
    return 1 if not state.get("exchange") else 3


def score(client: VLMClient, ts, state: dict, msgs, tool_choice: str) -> dict:
    t0 = time.time()
    try:
        r = client.chat(msgs, ts.tools, tool_choice, max_tokens=300, temperature=0.2)
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}", "type": state["type"]}
    out = {"type": state["type"], "latency_s": round(r.latency_s, 3), "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
           "n_calls": len(r.tool_calls), "notes": r.notes, "content": r.content[:80], "first_call_name": None, "parse_ok": False,
           "name_valid": False, "args_valid": False, "correct": False}
    if not r.tool_calls:
        return out
    c = r.tool_calls[0]
    out["first_call_name"] = c.name
    out["parse_ok"] = not c.error
    nm = c.name if find_tool(ts.tools, c.name) else closest_tool_name(ts.tools, c.name)
    out["name_valid"] = nm == c.name
    if nm:
        args, problems = coerce_arguments(ts.schemas[nm], c.arguments)
        spec = ts.skills.get(nm)
        if spec is not None:
            o = validate_one_of(spec, args)
            if o:
                problems.append(o)
        fatal = [p for p in problems if is_fatal(p)]
        out["args_valid"] = not fatal and not c.error
        out["problems"] = fatal
        ok_name = nm in state["accept"]
        ok_id = state["id"] is None or args.get("object_id") == state["id"]
        ok_fin = state.get("finish_success") is None or bool(args.get("success")) == state["finish_success"]
        out["correct"] = bool(ok_name and ok_id and ok_fin and out["args_valid"])
        # what the harness would make of it: verify without question gets one from the task; a synonym search is tolerated in state F
        fixed = [p for p in fatal if not (nm == "verify" and "give at least" in p)]
        lenient_name = nm in state.get("lenient", state["accept"])
        if nm == "search" and "lenient" in state:
            ok_fin = True
        out["correct_with_harness"] = bool(lenient_name and ok_id and ok_fin and not fixed and not c.error)
        out["args_valid_with_harness"] = not fixed and not c.error
        out["args"] = args
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parents[2] / "results" / "agent_live" / "live_eval"))
    ap.add_argument("--per-type", type=int, default=9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args()
    configs = ROOT.parents[2] / "configs"
    registry = load_registry(configs / "skills.yaml")
    pol = load_policy("object_nav", configs / "agent")
    ts = build_toolset(registry, list(pol.tools), list(pol.local_tools), pol.description_mode, pol.tool_descriptions)
    system = build_system_prompt(pol)
    client = VLMClient.from_config("reasoner", load_models_config())
    if not client.health():
        print("server not reachable:", client.base_url)
        return 2
    states = make_states(random.Random(a.seed), a.per_type)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"environment": "hand-made single-turn states (invented maps), not the robot", "n_states": len(states), "prompt_hash": pol.hash,
                    "tools_chars": len(json.dumps(ts.tools)), "model": client.config_dict()}
    for tc in ("auto", "required"):
        rows = [score(client, ts, s, build_messages(s, system, pol.hints), tc) for s in states]
        ok = [r for r in rows if "error" not in r]
        by_type: dict = {}
        for r in ok:
            by_type.setdefault(r["type"], []).append(r["correct"])
        lat = sorted(r["latency_s"] for r in ok)
        report[tc] = {"n": len(rows), "server_errors": len(rows) - len(ok),
                      "no_tool_call": sum(1 for r in ok if r["n_calls"] == 0), "multi_calls": sum(1 for r in ok if r["n_calls"] > 1),
                      "unparseable_args": sum(1 for r in ok if r["n_calls"] and not r["parse_ok"]),
                      "unknown_tool": sum(1 for r in ok if r["n_calls"] and not r["name_valid"]),
                      "invalid_args": sum(1 for r in ok if r["n_calls"] and r["name_valid"] and not r["args_valid"]),
                      "format_error_rate": round(sum(1 for r in ok if not (r["n_calls"] and r["parse_ok"] and r["name_valid"] and r["args_valid"])) / max(1, len(ok)), 3),
                      "quirk_notes": sorted({n for r in ok for n in r["notes"]}),
                      "invalid_args_after_harness_autofill": sum(1 for r in ok if r["n_calls"] and r["name_valid"] and not r.get("args_valid_with_harness", False)),
                      "decision_accuracy_with_harness": round(sum(r.get("correct_with_harness", False) for r in ok) / max(1, len(ok)), 3),
                      "decision_correct": sum(r["correct"] for r in ok), "decision_accuracy": round(sum(r["correct"] for r in ok) / max(1, len(ok)), 3),
                      "accuracy_by_type": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_type.items())},
                      "latency_mean_s": round(sum(lat) / max(1, len(lat)), 3), "latency_p50_s": lat[len(lat) // 2] if lat else None,
                      "latency_p95_s": lat[int(0.95 * (len(lat) - 1))] if lat else None, "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in ok) / max(1, len(ok)))}
        (out / f"states_{tc}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        print(tc, json.dumps(report[tc], indent=1), flush=True)
    if not a.skip_images:
        w = FakeWorld(seed=1)
        w.explored[:] = True
        for o in w.objects:
            o.seen = True
        cam, mp = w.camera_image(), w.map_image()
        lat = {"no_images": [], "camera": [], "map": [], "map+camera": []}
        sample = states[: 12]
        for s in sample:
            m0 = build_messages(s, system, pol.hints)
            variants = {"no_images": [], "camera": [cam], "map": [mp], "map+camera": [mp, cam]}
            for name, imgs in variants.items():
                m = list(m0)
                if imgs:
                    m[-1] = user_message(m[-1]["content"], imgs, max_side=448, quality=80)
                try:
                    r = client.chat(m, ts.tools, "auto", max_tokens=300, temperature=0.2)
                    lat[name].append((r.latency_s, r.prompt_tokens))
                except Exception as ex:
                    print("image variant error", name, ex)
        report["image_latency"] = {k: {"n": len(v), "latency_mean_s": round(sum(x for x, _ in v) / max(1, len(v)), 3),
                                       "prompt_tokens_mean": round(sum(t for _, t in v) / max(1, len(v)))} for k, v in lat.items()}
        print(json.dumps(report["image_latency"], indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
