"""Instruction follower for mode velocity_follow: NaVILA-style step-by-step language actions with a VLM.

Per step the model sees the instruction (split into subgoals), a few recent egocentric frames (first frame kept, then evenly
sampled, then the current one), the measured outcome of its last moves and optional map context, and answers with one
primitive (`move_forward 50`, `turn_left 30`, `stop`). The harness executes it through the skills move_relative / turn_to /
stop, which close the loop on odometry and stop on the lidar safety layer, and reports what really happened
("moved 38 cm", "BLOCKED after 40 cm"). Stop is a decision with checks: all subgoals done, optional VLM stop check.
"""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field

from x30_common.images import encode_image, image_part, text_part
from x30_common.jsonrepair import loads_lenient
from x30_common.toolschema import coerce_arguments, is_fatal, make_tool
from x30_common.vlm_client import VLMCancelled, VLMClient, VLMError

from .policy import Policy
from .primitives import ACTIONS, TURNS, Primitive, clamp, describe, parse_action_text, to_skill
from .trace import TraceWriter
from .types import (ABORTED, ERROR, FINISHED, NO_PROGRESS, SAFETY, TIMEOUT, AgentEnv, AgentResult, Pose2D, SkillResult, Task)


def act_tool() -> dict:
    return make_tool(
        "act",
        "Choose the robot's next move. amount is centimetres for move_*/strafe_*, degrees for turn_*, ignored for stop.",
        {"action": {"type": "string", "enum": list(ACTIONS)},
         "amount": {"type": "number", "description": "cm or degrees"},
         "subgoal_done": {"type": "boolean", "description": "true if the CURRENT subgoal is already satisfied in the current view"},
         "note": {"type": "string", "description": "a few words about what you see"}},
        ["action"])


_TURN_AMOUNT = re.compile(r"\b(?:turn|rotate|pivot)\b[^.,;]*?\b(left|right)\b[^.,;]*?(\d+(?:\.\d+)?)\s*(?:°|deg)", re.I)
_MOVE_AMOUNT = re.compile(r"\b(?:walk|move|go|advance|proceed)\b(?![^.,;]*\b(?:back|backward|left|right|strafe)\b)[^.,;]*?(\d+(?:\.\d+)?)\s*(cm|centimet\w*|m\b|meters?|metres?)", re.I)


def metric_target(text: str) -> tuple[str, float] | None:
    """('turn', signed degrees, left +) or ('move', cm) when a subgoal names an explicit angle or forward distance, else None."""
    m = _TURN_AMOUNT.search(text or "")
    if m:
        return "turn", float(m.group(2)) * (1 if m.group(1).lower() == "left" else -1)
    m = _MOVE_AMOUNT.search(text or "")
    if m:
        v = float(m.group(1))
        return "move", v if m.group(2).lower().startswith("c") else v * 100.0
    return None


@dataclass
class Subgoal:
    do: str
    done_when: str = ""
    done: bool = False

    @property
    def metric(self) -> tuple[str, float] | None:
        return metric_target(self.do)

    def reached(self, prog: list, fraction: float) -> bool:
        """True when the measured progress covers `fraction` of the named amount (direction respected)."""
        m = self.metric
        if m is None:
            return False
        kind, amount = m
        if kind == "turn":
            return prog[1] * (1 if amount > 0 else -1) >= fraction * abs(amount)
        return prog[0] >= fraction * amount


def heuristic_subgoals(instruction: str) -> list[Subgoal]:
    parts = re.split(r",\s*(?:and\s+)?(?:then\s+)?|\s+and then\s+|\.\s+|;\s*|\bthen\b", instruction.strip().rstrip("."), flags=re.I)
    parts = [p.strip() for p in parts if p and len(p.strip()) > 2]
    return [Subgoal(p) for p in parts[:6]] or [Subgoal(instruction)]


class LoopDetector:
    """Detects blocked repetition, no net progress, oscillation and revisits from the executed steps."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.steps: list[dict] = []        # {action, amount, blocked, pose}
        self.warn_run = 0

    def add(self, prim: Primitive, blocked: bool, pose: Pose2D | None) -> None:
        self.steps.append({"a": prim.action, "n": prim.amount, "blocked": blocked, "pose": pose})

    def refuse(self, prim: Primitive) -> str:
        """Reason to refuse `prim` before executing it, or ''."""
        n = int(self.cfg.repeat_blocked)
        tail = self.steps[-n:]
        if len(tail) == n and all(s["blocked"] and s["a"] == prim.action for s in tail) and prim.action not in TURNS + ("stop",):
            return f"{prim.action} was blocked {n} times in a row"
        return ""

    def warning(self) -> str:
        c = self.cfg
        w = self.steps[-int(c.window):]
        msgs = []
        moves = [s for s in w if s["a"] not in TURNS + ("stop",)]
        if len(moves) >= 3 and len(w) >= int(c.window) and w[0]["pose"] and w[-1]["pose"]:
            net = w[0]["pose"].dist(w[-1]["pose"])
            if net < float(c.min_net_move_m):
                msgs.append(f"Over the last {len(w)} steps the robot moved only {net:.2f} m net: you are not making progress.")
        turns = [s for s in self.steps[-int(c.oscillation_len):] if s["a"] in TURNS]
        if len(turns) >= int(c.oscillation_len) and all(turns[i]["a"] != turns[i + 1]["a"] for i in range(len(turns) - 1)):
            msgs.append("You are turning left and right alternately: pick one direction and commit.")
        poses = [s["pose"] for s in self.steps[:-1] if s["pose"] is not None and s["a"] not in TURNS]
        if self.steps and self.steps[-1]["pose"] is not None and self.steps[-1]["a"] not in TURNS:
            near = sum(1 for p in poses if p.dist(self.steps[-1]["pose"]) < float(c.revisit_radius_m))
            if near >= int(c.revisit_count):
                msgs.append("The robot keeps coming back to the same place: try a clearly different direction.")
        self.warn_run = self.warn_run + 1 if msgs else 0
        return " ".join(msgs)


class FollowerAgent:
    def __init__(self, client: VLMClient, env: AgentEnv, policy: Policy, trace_dir: str | None = None,
                 cancel: threading.Event | None = None, models_hash: str = ""):
        self.client, self.env, self.policy = client, env, policy
        self.trace_dir = trace_dir
        self.cancel = cancel or threading.Event()
        self.models_hash = models_hash
        self.tools = [act_tool()]
        self.system = policy.text("system").strip()

    # ------------------------------------------------------------------ run
    def run(self, task: Task) -> AgentResult:
        pol = self.policy
        t0, wall0 = self.env.now(), time.time()
        max_steps = int(task.max_turns or pol.limits.max_steps)
        budget_s = float(task.time_budget_s or pol.limits.time_budget_s)
        tr = TraceWriter(self.trace_dir, task.episode_id, enabled=bool(self.trace_dir))
        tr.write("header", episode=task.episode_id, mode="velocity_follow", task=task.instruction, prompt_hash=pol.hash,
                 models_hash=self.models_hash, model=self.client.config_dict(), output_format=pol.output_format,
                 limits={"max_steps": max_steps, "time_budget_s": budget_s})
        stats = {"format_retries": 0, "noop_steps": 0, "blocked": 0, "loop_warnings": 0, "stop_refusals": 0, "stop_checks": 0,
                 "stop_check_no": 0, "refused_moves": 0, "clamped": 0, "auto_advanced": 0}
        subgoals = self._plan(task, tr)
        loop = LoopDetector(pol.loop)
        frames: list = []
        first_frame = None
        log: list[str] = []
        feedback = ""
        prog = [0.0, 0.0]                      # measured progress on the current subgoal: net forward cm, net turn deg (left positive)
        result: AgentResult | None = None
        step = noop_run = llm_err = blocked_run = stop_refused = 0
        guard_overridden = False
        try:
            while result is None:
                if self.cancel.is_set():
                    result = AgentResult(False, ABORTED, "cancelled")
                    break
                elapsed = self.env.now() - t0
                if elapsed > budget_s:
                    result = AgentResult(False, TIMEOUT, f"time budget of {budget_s:.0f} s used")
                    break
                if step >= max_steps:
                    result = AgentResult(False, TIMEOUT, f"step limit {max_steps} reached")
                    break
                step += 1
                pose = self.env.pose()
                cur = self.env.camera_image()
                if cur is not None:
                    if first_frame is None:
                        first_frame = cur
                    frames.append(cur)
                    del frames[:-(int(pol.images.history_frames) * 6 + 1)]
                shown = self._sample_frames(first_frame, frames, int(pol.images.history_frames))
                warn = loop.warning()
                if warn:
                    stats["loop_warnings"] += 1
                    if loop.warn_run >= int(pol.loop.warn_limit):
                        result = AgentResult(False, NO_PROGRESS, "loop detected: " + warn)
                        tr.write("event", name="loop_abort", step=step, warning=warn)
                        break
                msgs, paths = self._messages(task, subgoals, shown, log, feedback, warn, step, max_steps, pose, tr, prog)
                dec = self._ask(msgs, pol.output_format)
                stats["format_retries"] += dec["retries"]
                row = {"turn": step, "prompt_hash": pol.hash, "pose": None if pose is None else [round(pose.x, 3), round(pose.y, 3), round(pose.yaw, 3)],
                       "images": paths, "latency_s": round(dec["latency_s"], 3), "prompt_tokens": dec["prompt_tokens"],
                       "completion_tokens": dec["completion_tokens"], "format_retries": dec["retries"], "reply": dec["reply"],
                       "feedback_in": feedback, "loop_warning": warn, "subgoals": [("x" if g.done else "-") + g.do for g in subgoals]}
                if pol.trace.messages:
                    from x30_common.images import strip_images
                    row["messages"] = strip_images(msgs)
                if dec["kind"] == "cancelled":
                    tr.write("turn", **row, tool=None, parse_ok=False)
                    result = AgentResult(False, ABORTED, "cancelled")
                    break
                if dec["kind"] == "error":
                    llm_err += 1
                    tr.write("turn", **row, tool=None, parse_ok=False, note=dec["note"])
                    if llm_err >= int(pol.limits.max_llm_errors):
                        result = AgentResult(False, ERROR, "LLM error: model server: " + dec["note"])
                    continue
                llm_err = 0
                if dec["kind"] == "noop":
                    stats["noop_steps"] += 1
                    noop_run += 1
                    feedback = "Your last reply had no valid action. Answer with the act tool."
                    tr.write("turn", **row, tool=None, parse_ok=False, note=dec["note"])
                    if noop_run >= int(pol.limits.max_noop_steps):
                        result = AgentResult(False, ERROR, f"{noop_run} steps without a usable action")
                    continue
                noop_run = 0
                prim, note = clamp(dec["prim"], pol.actions)
                if note:
                    stats["clamped"] += 1
                if dec["subgoal_done"]:
                    self._advance(subgoals)
                    prog[:] = [0.0, 0.0]
                all_done = all(g.done for g in subgoals)
                # ---- stop decision
                if prim.action == "stop":
                    verdict, why = self._stop_decision(task, subgoals, all_done, step, frames, stop_refused, stats, tr)
                    row.update(tool="act", args={"action": "stop", "subgoal_done": dec["subgoal_done"]}, parse_ok=True, stop_verdict=verdict, stop_why=why)
                    if verdict in ("accept", "override"):
                        guard_overridden = verdict == "override"
                        self.env.run_skill("stop", {}, 10.0, self.cancel)
                        tr.write("turn", **row)
                        msg = "stopped: " + (why or "all subgoals done")
                        result = AgentResult(True, FINISHED, msg, guard_overridden=guard_overridden)
                        break
                    stop_refused += 1
                    stats["stop_refusals"] += 1
                    feedback = f"Stop refused: {why}. Continue with the instruction."
                    log.append(f"{step}. stop -> refused ({why})")
                    tr.write("turn", **row)
                    continue
                # ---- execute
                refused = loop.refuse(prim)
                if refused:
                    stats["refused_moves"] += 1
                    feedback = f"Refused: {refused}. Turn, strafe, or choose another way."
                    log.append(f"{step}. {describe(prim)} -> refused ({refused})")
                    tr.write("turn", **row, tool="act", args={"action": prim.action, "amount": prim.amount}, parse_ok=True, refused=refused)
                    loop.add(Primitive("refused", 0), True, pose)
                    blocked_run += 1
                    if blocked_run >= int(pol.limits.max_consecutive_blocked):
                        result = AgentResult(False, SAFETY, "blocked repeatedly")
                    continue
                skill, sargs = to_skill(prim)
                t1 = time.time()
                res = self.env.run_skill(skill, sargs, float(pol.limits.skill_timeout_s), self.cancel)
                if res.cancelled:
                    result = AgentResult(False, ABORTED, "cancelled during " + describe(prim))
                    tr.write("turn", **row, tool="act", args={"action": prim.action, "amount": prim.amount}, parse_ok=True, note="cancelled")
                    break
                self._add_progress(prog, prim, res)
                auto_note = ""
                if bool(pol.prompt.auto_advance_metric):
                    cur_sg = next((g for g in subgoals if not g.done), None)
                    if cur_sg is not None and cur_sg.reached(prog, float(pol.prompt.auto_advance_fraction)):
                        cur_sg.done = True
                        tr.write("event", name="auto_advance", step=step, subgoal=cur_sg.do, progress=[round(prog[0]), round(prog[1])])
                        stats["auto_advanced"] += 1
                        auto_note = f" Subgoal '{cur_sg.do}' is complete (measured progress reached)."
                        prog[:] = [0.0, 0.0]
                blocked = bool(res.data.get("blocked")) or (not res.success and "block" in res.message.lower())
                feedback, outcome = self._feedback(prim, res, blocked)
                feedback += auto_note
                loop.add(prim, blocked, self.env.pose())
                if blocked:
                    stats["blocked"] += 1
                    blocked_run += 1
                else:
                    blocked_run = 0
                log.append(f"{step}. {describe(prim)} -> {outcome}")
                tr.write("turn", **row, tool="act", args={"action": prim.action, "amount": prim.amount, "subgoal_done": dec["subgoal_done"]},
                         parse_ok=True, clamp_note=note, success=res.success, result=res.to_dict(), skill_s=round(time.time() - t1, 3),
                         blocked=blocked, feedback_out=feedback)
                self.env.feedback(f"step {step}: {describe(prim)}", skill, float("nan"), step)
                if blocked_run >= int(pol.limits.max_consecutive_blocked):
                    result = AgentResult(False, SAFETY, f"{blocked_run} blocked moves in a row")
        except Exception as ex:
            tr.write("event", name="exception", error=f"{type(ex).__name__}: {ex}")
            result = AgentResult(False, ERROR, f"agent exception: {type(ex).__name__}: {ex}")
        result.turns = step
        result.duration_s = self.env.now() - t0
        result.finish_refusals = stats["stop_refusals"]
        result.stats = {**stats, "llm": self.client.stats.summary(since=wall0), "subgoals_done": sum(g.done for g in subgoals),
                        "subgoals": len(subgoals), "prompt_hash": pol.hash}
        if result.stop_reason in (ABORTED, TIMEOUT, ERROR, NO_PROGRESS, SAFETY):
            try:
                self.env.run_skill("stop", {}, 10.0, threading.Event())
            except Exception:
                pass
        tr.write("summary", **result.to_dict(), trace_write_errors=tr.errors)
        tr.close()
        return result

    # ------------------------------------------------------------------ planning
    def _plan(self, task: Task, tr: TraceWriter) -> list[Subgoal]:
        if not self.policy.prompt.subgoals:
            return [Subgoal(task.instruction)]
        try:
            res = self.client.chat([{"role": "system", "content": self.policy.text("planner").strip()},
                                    {"role": "user", "content": f'Instruction: "{task.instruction}"'}],
                                   max_tokens=300, temperature=0.0, cancel=self.cancel)
            val, _ = loads_lenient(res.content)
            items = val.get("subgoals") if isinstance(val, dict) else val
            sg = [Subgoal(str(i.get("do", "")), str(i.get("done_when", ""))) for i in items if isinstance(i, dict) and i.get("do")]
            if sg:
                tr.write("event", name="plan", subgoals=[(g.do, g.done_when) for g in sg], source="vlm", latency_s=round(res.latency_s, 2))
                return sg[:6]
        except (VLMError, ValueError, AttributeError, TypeError) as ex:
            tr.write("event", name="plan_failed", error=f"{type(ex).__name__}: {ex}")
        sg = heuristic_subgoals(task.instruction)
        tr.write("event", name="plan", subgoals=[(g.do, g.done_when) for g in sg], source="heuristic")
        return sg

    @staticmethod
    def _advance(subgoals: list[Subgoal]) -> None:
        for g in subgoals:
            if not g.done:
                g.done = True
                return

    # ------------------------------------------------------------------ prompt
    @staticmethod
    def _sample_frames(first, frames: list, k: int) -> list:
        """NaVILA style: first frame, then uniformly sampled earlier frames, then the current one (k history + 1)."""
        if not frames:
            return []
        cur = frames[-1]
        past = frames[:-1]
        if k <= 0 or not past:
            return [cur]
        pick = [first] if first is not None and first is not cur else []
        rest = k - len(pick)
        if rest > 0 and past:
            idx = sorted({int(round(i * (len(past) - 1) / max(1, rest))) for i in range(1, rest + 1)}) if len(past) > 1 else [0]
            pick += [past[i] for i in idx if past[i] is not first][:rest]
        return pick + [cur]

    def _messages(self, task, subgoals, shown, log, feedback, warn, step, max_steps, pose, tr, progress=None):
        pol = self.policy
        parts: list[dict] = []
        paths: list[str] = []
        ms, q = int(pol.images.max_side_px), int(pol.images.jpeg_quality)
        if len(shown) > 1:
            parts.append(text_part("Earlier views (oldest first):"))
            for i, im in enumerate(shown[:-1]):
                parts.append(image_part(im, ms, q))
        if shown:
            parts.append(text_part("CURRENT view:"))
            parts.append(image_part(shown[-1], ms, q))
            for i, im in enumerate(shown):
                paths.append(tr.save_image(f"s{step:03d}_f{i}", encode_image(im, ms, q)))
        if pol.images.map:
            mp = self.env.map_image()
            if mp is not None:
                parts.append(text_part("Top-down map (robot = arrow):"))
                parts.append(image_part(mp, ms, q))
        lines = [f'Instruction: "{task.instruction}"']
        if pol.prompt.subgoals:
            cur = next((i for i, g in enumerate(subgoals) if not g.done), None)
            lines.append("Subgoals:")
            for i, g in enumerate(subgoals):
                tag = "done" if g.done else ("CURRENT" if i == cur else "todo")
                lines.append(f"  {i + 1}. [{tag}] {g.do}" + (f" (done when: {g.done_when})" if g.done_when and not g.done else ""))
            if cur is None:
                lines.append("All subgoals are done: choose stop unless the view shows the goal is not yet reached.")
        if progress and (abs(progress[0]) >= 1 or abs(progress[1]) >= 1):
            lines.append(f"Measured progress on the CURRENT subgoal so far: net forward {progress[0]:.0f} cm, net turn {progress[1]:+.0f} deg (left is +).")
        lines.append(f"Step {step}/{max_steps}.")
        if pol.images.map_summary:
            s = self.env.map_summary(pose, 12)
            if s:
                lines.append("Known objects:\n" + s)
        n = int(pol.prompt.recent_actions)
        if log:
            lines.append("Your recent moves and what happened:")
            lines += [f"  {x}" for x in log[-n:]]
        if feedback:
            lines.append(f"Last feedback: {feedback}")
        if warn:
            lines.append(f"WARNING: {warn}")
        lines.append("Choose the next action." if pol.output_format == "tool" else
                     "Reply with one line: ACTION: <action> <amount>  (and SUBGOAL_DONE: yes/no).")
        parts.append(text_part("\n".join(lines)))
        sysmsg = self.system
        if pol.output_format == "text":
            sysmsg += "\n\nAnswer format: `ACTION: move_forward 50` (or turn_left 30, stop, ...) then `SUBGOAL_DONE: yes|no`. No other text."
        return [{"role": "system", "content": sysmsg}, {"role": "user", "content": parts}], paths

    def _ask(self, msgs: list[dict], fmt: str) -> dict:
        pol = self.policy
        work = list(msgs)
        retries = 0
        out = {"kind": "noop", "retries": 0, "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "reply": {}, "note": ""}
        while True:
            try:
                res = self.client.chat(work, self.tools if fmt == "tool" else None, pol.model.tool_choice if fmt == "tool" else None,
                                       max_tokens=int(pol.model.max_tokens), temperature=float(pol.model.temperature), cancel=self.cancel)
            except VLMCancelled:
                return {**out, "kind": "cancelled", "retries": retries}
            except VLMError as ex:
                return {**out, "kind": "error", "note": f"{type(ex).__name__}: {ex}", "retries": retries}
            out.update(retries=retries, latency_s=res.latency_s, prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                       reply={"content": res.content, "notes": res.notes, "finish_reason": res.finish_reason,
                              "tool_calls": [{"name": t.name, "raw_arguments": t.raw_arguments, "repaired": t.repaired, "error": t.error} for t in res.tool_calls]})
            prim, sub_done, note = self._parse(res, fmt)
            if prim is not None:
                return {**out, "kind": "call", "prim": prim, "subgoal_done": sub_done, "note": note}
            out["note"] = note
            if retries >= int(pol.limits.format_retries):
                return {**out, "kind": "noop"}
            retries += 1
            if res.tool_calls:
                am = res.assistant_message()
                am["tool_calls"] = am["tool_calls"][:1]
                work = work + [am, {"role": "tool", "tool_call_id": res.tool_calls[0].id, "content": f"Invalid: {note}. Call act with a valid action."}]
            else:
                work = work + [{"role": "assistant", "content": res.content or "(empty)"},
                               {"role": "user", "content": f"Invalid reply ({note}). " + ("Call the act tool once." if fmt == "tool" else "Reply with `ACTION: <action> <amount>`.")}]

    @staticmethod
    def _parse(res, fmt: str) -> tuple[Primitive | None, bool, str]:
        if fmt == "tool" or res.tool_calls:
            for tc in res.tool_calls:
                if tc.name != "act":
                    continue
                args, problems = coerce_arguments(act_tool()["function"]["parameters"], tc.arguments)
                if any(is_fatal(p) for p in problems):
                    return None, False, "; ".join(p for p in problems if is_fatal(p))
                return Primitive(args["action"], float(args.get("amount") or 0.0)), bool(args.get("subgoal_done", False)), ""
            if res.tool_calls:
                return None, False, f"unknown tool {res.tool_calls[0].name!r}"
            if fmt == "tool":
                p = parse_action_text(res.content)           # tolerate a plain-text answer in tool mode
                return (p, False, "action parsed from text") if p else (None, False, "no tool call")
        text = res.content
        line = next((l for l in text.splitlines() if l.strip().lower().startswith("action")), text)
        line = line.split(":", 1)[1] if ":" in line and line.strip().lower().startswith("action") else line
        p = parse_action_text(line)
        done = bool(re.search(r"subgoal[_ ]?done\s*[:=]\s*(yes|true)", text, re.I))
        return (p, done, "") if p else (None, False, f"no action found in {text[:60]!r}")

    # ------------------------------------------------------------------ stop / feedback
    def _stop_decision(self, task, subgoals, all_done, step, frames, refused_before, stats, tr) -> tuple[str, str]:
        pol = self.policy
        if step < int(pol.limits.min_steps_before_stop) + 0 and not all_done:
            return "refuse", "too early"
        if not all_done:
            left = [g.do for g in subgoals if not g.done]
            why = f"subgoal not done yet: '{left[0]}'"
            if refused_before >= int(pol.limits.stop_refusals_before_accept):
                return "override", why + " (accepted after repeated stop requests)"
            return "refuse", why
        if pol.stop_check.enabled and frames:
            stats["stop_checks"] += 1
            ans, why = self._stop_check(task, subgoals, frames, tr)
            if ans == "no":
                stats["stop_check_no"] += 1
                if refused_before >= int(pol.limits.stop_refusals_before_accept):
                    return "override", f"stop check said no: {why} (accepted after repeated stop requests)"
                return "refuse", f"the view does not show the goal reached ({why})"
        return "accept", ""

    def _stop_check(self, task, subgoals, frames, tr) -> tuple[str, str]:
        pol = self.policy
        last = subgoals[-1]
        prompt = pol.text("stop_check").format(instruction=task.instruction, last_subgoal=last.do, done_when=last.done_when or "the instruction is fulfilled")
        imgs = frames[-int(pol.stop_check.frames):]
        parts = [image_part(i, int(pol.images.max_side_px), int(pol.images.jpeg_quality)) for i in imgs] + [text_part(prompt)]
        try:
            res = self.client.chat([{"role": "user", "content": parts}], max_tokens=int(pol.stop_check.max_tokens), temperature=0.0, cancel=self.cancel)
        except VLMError as ex:
            tr.write("event", name="stop_check_error", error=str(ex))
            return "unsure", "stop check unavailable"
        t = res.content.strip().lower()
        ans = "yes" if t.startswith("yes") else "no" if t.startswith("no") else "unsure"
        tr.write("event", name="stop_check", answer=ans, reply=res.content[:200], latency_s=round(res.latency_s, 2))
        if ans == "unsure" and not pol.stop_check.accept_unsure:
            ans = "no"
        return ans, res.content.strip()[:100]

    @staticmethod
    def _add_progress(prog: list, prim: Primitive, res: SkillResult) -> None:
        d = res.data or {}
        if prim.action in TURNS:
            ach = d.get("achieved_deg")
            if isinstance(ach, (int, float)):
                prog[1] += abs(ach) * (1 if prim.action == "turn_left" else -1)
        elif prim.action in ("move_forward", "move_backward"):
            ach = d.get("achieved_cm")
            if isinstance(ach, (int, float)):
                prog[0] += abs(ach) * (1 if prim.action == "move_forward" else -1)

    @staticmethod
    def _feedback(prim: Primitive, res: SkillResult, blocked: bool) -> tuple[str, str]:
        d = res.data or {}
        if prim.action in TURNS:
            ach = d.get("achieved_deg")
            sign = 1 if prim.action == "turn_left" else -1
            outcome = f"turned {abs(ach):.0f} deg" if isinstance(ach, (int, float)) else ("done" if res.success else "failed")
            _ = sign
        else:
            ach = d.get("achieved_cm")
            if blocked:
                outcome = f"BLOCKED after {ach:.0f} cm of {prim.amount:g}" if isinstance(ach, (int, float)) else "BLOCKED"
                obs = d.get("obstacle_distance_cm")
                if isinstance(obs, (int, float)):
                    outcome += f" (obstacle {obs:.0f} cm ahead)"
            else:
                outcome = f"moved {ach:.0f} cm" if isinstance(ach, (int, float)) else ("done" if res.success else "failed")
        if not res.success and not blocked:
            outcome += f" - {res.message[:80]}"
        fb = f"{describe(prim)}: {outcome}."
        if blocked:
            fb += " Something is in the way: do not repeat the same move; turn, strafe or choose another direction."
        return fb, outcome
