"""Rule-based stand-ins for the VLM, served through the fake OpenAI server. They exist to exercise the HARNESS (parsing,
guards, loops, cancel) with controlled noise; they say nothing about the real model's quality."""
from __future__ import annotations

import json
import math
import random
import re

from x30_common.testing.fake_openai import Reply, text_reply, text_tool_call_reply, tool_call_reply

from x30_agent.core.baselines import target_label_from


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


class ObjNavMock:
    """Follows the NEXT hints in the last tool result, with injected mistakes (probabilities per request)."""

    def __init__(self, seed=0, p_malformed=0.05, p_text=0.03, p_nocall=0.03, p_repeat=0.05, p_prefinish=0.03):
        self.rng = random.Random(seed)
        self.p = dict(malformed=p_malformed, text=p_text, nocall=p_nocall, repeat=p_repeat, prefinish=p_prefinish)
        self.last: tuple | None = None
        self.task = ""

    def __call__(self, req: dict, idx: int) -> Reply:
        msgs = req["messages"]
        state = _text(msgs[-1])
        m = re.search(r"Task: (.+)", state)
        self.task = m.group(1).strip() if m else self.task
        tool_texts = [_text(x) for x in msgs if x["role"] == "tool"]
        label = target_label_from(self.task)
        call = self._next(tool_texts[-1] if tool_texts else "", label, state)
        r = self.rng.random()
        pm = self.p
        if r < pm["nocall"]:
            return text_reply("I will look around.")
        if r < pm["nocall"] + pm["prefinish"]:
            call = ("finish", {"success": True, "reason": "looks done"})
        elif r < pm["nocall"] + pm["prefinish"] + pm["repeat"] and self.last:
            call = self.last
        self.last = call
        r2 = self.rng.random()
        if r2 < pm["malformed"]:
            return tool_call_reply(call[0], "{'" + "', '".join(f"{k}': {json.dumps(v)}" for k, v in call[1].items()) + ",}")
        if r2 < pm["malformed"] + pm["text"]:
            return text_tool_call_reply(call[0], call[1])
        return tool_call_reply(call[0], call[1])

    def _next(self, last_tool: str, label: str, state: str) -> tuple[str, dict]:
        m = re.search(r"NEXT: (\w+)\(([^)]*)\)", last_tool)
        ids = re.findall(r"#(\d+)", last_tool)
        question = f"Is this what the task asks for? Task: {self.task}"
        if not last_tool:
            seen = re.findall(r"#(\d+) " + re.escape(label), state)
            return ("goto_object", {"object_id": int(seen[0])}) if seen else ("search", {"label": label})
        if "Rejected by verify" in state and "refused" not in last_tool and not m:
            return ("search", {"label": label})
        if "refused" in last_tool.lower() or "Refused" in last_tool:
            return ("explore", {})
        if not m:
            return ("explore", {})
        name = m.group(1)
        mid = re.search(r"object_id=(\d+)", m.group(2))
        oid = int(mid.group(1)) if mid else (int(ids[0]) if ids else None)
        if name == "verify" and oid is not None:
            return ("verify", {"object_id": oid, "question": question})
        if name in ("goto_object", "approach") and oid is not None:
            return (name, {"object_id": oid})
        if name == "finish":
            return ("finish", {"success": "success=false" not in m.group(2), "reason": "done"})
        if name == "search":
            return ("search", {"label": label})
        if name in ("explore", "scan"):
            return (name, {})
        return ("explore", {})


class FollowMock:
    """Oracle for the instruction follower: aims at the target along the world's A* path, with noise and early stops."""

    def __init__(self, world, target, seed=0, p_random=0.10, p_wrong_turn=0.05, p_early_stop=0.04, p_malformed=0.04, n_sub=2):
        self.w, self.t = world, target
        self.rng = random.Random(seed)
        self.p = (p_random, p_wrong_turn, p_early_stop, p_malformed)
        self.n_sub = n_sub
        self.start_len = None

    def __call__(self, req: dict, idx: int) -> Reply:
        msgs = req["messages"]
        sys_txt = _text(msgs[0]) if msgs[0]["role"] == "system" else ""
        if "Split the navigation instruction" in sys_txt:
            sub = [{"do": "walk to the " + self.t.label, "done_when": "the robot is close to it"}]
            if self.n_sub > 1:
                sub.append({"do": "stop next to it", "done_when": "it is within one metre"})
            return text_reply(json.dumps({"subgoals": sub}))
        if not req.get("tools"):
            d = math.hypot(self.w.pos.x - self.t.x, self.w.pos.y - self.t.y)
            return text_reply("yes, it is right here" if d < 2.0 else "no, still far")
        w = self.w
        pose = w.pos
        dist = math.hypot(self.t.x - pose.x, self.t.y - pose.y)
        pts = w._astar((pose.x, pose.y), (self.t.x, self.t.y), True) or [(self.t.x, self.t.y)]
        if self.start_len is None:
            self.start_len = max(1.0, len(pts))
        frac = 1.0 - len(pts) / self.start_len
        done = dist < 1.3
        sg = done or (self.n_sub > 1 and frac > 0.5)
        if done:
            act = ("stop", 0)
        else:
            tgt = next((p for p in pts if math.hypot(p[0] - pose.x, p[1] - pose.y) > 0.5), pts[-1])
            err = math.degrees(math.atan2(tgt[1] - pose.y, tgt[0] - pose.x) - pose.yaw)
            err = (err + 180) % 360 - 180
            if abs(err) > 8:
                act = ("turn_left" if err > 0 else "turn_right", min(90, round(abs(err))))
            else:
                act = ("move_forward", min(40, max(20, round(math.hypot(tgt[0] - pose.x, tgt[1] - pose.y) * 100))))
        r = self.rng.random()
        pr, pw, pe, pm = self.p
        if r < pr:
            act = self.rng.choice([("move_forward", 150), ("turn_left", 45), ("turn_right", 45), ("move_backward", 30)])
        elif r < pr + pw and act[0].startswith("turn"):
            act = ("turn_left" if act[0] == "turn_right" else "turn_right", act[1])
        elif r < pr + pw + pe:
            act = ("stop", 0)
        args = {"action": act[0], "amount": act[1], "subgoal_done": bool(sg)}
        if self.rng.random() < pm:
            return tool_call_reply("act", "{'action': '" + act[0] + "', 'amount': " + str(act[1]) + ",}")
        return tool_call_reply("act", args)
