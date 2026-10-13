"""Reasoner agent for mode object_nav: a tool-calling loop around a VLM, with a harness that does the bookkeeping.

The model chooses ONE skill per turn. Everything the drone project learned the hard way is done by the harness:
rejected objects are remembered and injected, identical repeated calls are refused, every result carries a NEXT
hint, `finish(success=true)` is guarded by fresh verification, malformed replies are re-asked, and every turn is traced.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from x30_common.images import encode_image, user_message
from x30_common.toolschema import closest_tool_name, coerce_arguments, find_tool, is_fatal, tool_name
from x30_common.vlm_client import VLMCancelled, VLMClient, VLMContextOverflow, VLMError

from .guard import FinishGuard
from .hints import HintEngine, render_result
from .memory import HarnessMemory, VerifyRecord, call_key, fmt_args
from .policy import Policy
from .prompts import TokenEstimator, build_system_prompt, image_tokens_for
from .skills import SkillSpec, validate_one_of
from .tools import ToolSet, build_toolset
from .trace import TraceWriter
from .types import (ABORTED, ERROR, FINISHED, NO_PROGRESS, TIMEOUT, AgentEnv, AgentResult, Pose2D, SkillResult, Task)


@dataclass
class _Decision:
    kind: str                                   # call | noop | error | cancelled
    name: str = ""
    args: dict = field(default_factory=dict)
    call_id: str = ""
    problems: list = field(default_factory=list)
    reply: dict = field(default_factory=dict)   # raw model reply summary for the trace
    extra_calls: list = field(default_factory=list)
    retries: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    note: str = ""
    invalid: bool = False                       # the call exists but its arguments are unusable


class ReasonerAgent:
    def __init__(self, client: VLMClient, env: AgentEnv, registry: dict[str, SkillSpec], policy: Policy,
                 trace_dir: str | None = None, cancel: threading.Event | None = None, models_hash: str = "", mode: str = "object_nav"):
        self.client, self.env, self.registry, self.policy = client, env, registry, policy
        self.trace_dir = trace_dir
        self.cancel = cancel or threading.Event()
        self.models_hash = models_hash
        self.mode = mode
        self.toolset: ToolSet = build_toolset(registry, list(policy.tools), list(policy.local_tools), policy.description_mode,
                                              policy.tool_descriptions)
        self.hints = HintEngine(policy.hints)
        self.guard = FinishGuard(policy.guard, self.hints)
        self.system_prompt = build_system_prompt(policy)
        self.est = TokenEstimator(policy.budget.chars_per_token)
        self._task: Task | None = None
        ctx = int(policy.budget.context_tokens or getattr(client, "context_tokens", 0) or 8192)
        self.context_tokens = ctx

    # ------------------------------------------------------------------ public
    def run(self, task: Task) -> AgentResult:
        pol = self.policy
        self._task = task
        t0 = self.env.now()
        wall0 = time.time()
        max_turns = int(task.max_turns or pol.limits.max_turns)
        budget_s = float(task.time_budget_s or pol.limits.time_budget_s)
        mem = HarnessMemory()
        tr = TraceWriter(self.trace_dir, task.episode_id, enabled=bool(self.trace_dir))
        tr.write("header", episode=task.episode_id, mode=self.mode, task=task.instruction, target_label=task.target_label,
                 prompt_hash=pol.hash, policy=pol.name, models_hash=self.models_hash, model=self.client.config_dict(),
                 tools=self.toolset.names, limits={"max_turns": max_turns, "time_budget_s": budget_s},
                 context_tokens=self.context_tokens, images={"map": pol.images.map, "camera": pol.images.camera})
        units: list[list[dict]] = []
        stats = {"format_retries": 0, "noop_turns": 0, "invalid_calls": 0, "refusals": 0, "tool_counts": {}, "llm_errors": 0,
                 "extra_calls_dropped": 0, "context_overflows": 0}
        result: AgentResult | None = None
        turn = 0
        noop_run = llm_err_run = 0
        degraded_last: list[str] = []
        try:
            while result is None:
                if self.cancel.is_set():
                    result = AgentResult(False, ABORTED, "cancelled")
                    break
                elapsed = self.env.now() - t0
                if elapsed > budget_s:
                    result = AgentResult(False, TIMEOUT, f"time budget of {budget_s:.0f} s used")
                    break
                if turn >= max_turns:
                    result = AgentResult(False, TIMEOUT, f"turn limit {max_turns} reached")
                    break
                turn += 1
                pose = self.env.pose()
                summary = self._safe_summary(pose)
                msgs, img_paths, degraded, est = self._build_messages(task, turn, max_turns, elapsed, budget_s, pose, summary, mem, units, tr)
                degraded_last = degraded
                self.env.feedback("thinking", "", float("nan"), turn)
                dec = self._decide(msgs)
                stats["format_retries"] += dec.retries
                self.est.calibrate(est, dec.prompt_tokens)
                row = {"turn": turn, "prompt_hash": pol.hash, "pose": _pose_list(pose), "map_summary": summary, "notes": mem.notes(pol.prompt.action_log_lines),
                       "images": img_paths, "degraded": degraded, "est_prompt_tokens": est, "prompt_tokens": dec.prompt_tokens,
                       "completion_tokens": dec.completion_tokens, "latency_s": round(dec.latency_s, 3), "format_retries": dec.retries,
                       "reply": dec.reply, "n_history_units": len(units)}
                if self.trace_dir and pol.trace.messages:
                    row["messages"] = _plain(msgs)
                if dec.kind == "cancelled":
                    tr.write("turn", **row, tool=None, parse_ok=False, note="cancelled")
                    result = AgentResult(False, ABORTED, "cancelled")
                    break
                if dec.kind == "error":
                    stats["llm_errors"] += 1
                    llm_err_run += 1
                    tr.write("turn", **row, tool=None, parse_ok=False, note=dec.note)
                    if llm_err_run >= pol.limits.max_llm_errors:
                        result = AgentResult(False, ERROR, f"LLM error: model server: {dec.note}")
                    continue
                llm_err_run = 0
                if dec.kind == "noop":
                    stats["noop_turns"] += 1
                    noop_run += 1
                    tr.write("turn", **row, tool=None, parse_ok=False, note=dec.note)
                    if noop_run >= pol.limits.max_noop_turns:
                        result = AgentResult(False, ERROR, f"{noop_run} turns without a usable tool call")
                    continue
                noop_run = 0
                parse_ok = not dec.invalid
                if dec.invalid:
                    stats["invalid_calls"] += 1
                # ---- execute (or refuse) the call
                name, args = dec.name, dec.args
                hint_prefix = ""
                if dec.extra_calls:
                    stats["extra_calls_dropped"] += len(dec.extra_calls)
                    hint_prefix = self.hints.text("extra_calls_dropped", n=1 + len(dec.extra_calls)) + " "
                done_reason, res, hint, refused, guard_row, skill_s = self._act(task, name, args, dec, mem, turn, pose, t0, budget_s)
                stats["tool_counts"][name] = stats["tool_counts"].get(name, 0) + 1
                if refused:
                    stats["refusals"] += 1
                text = render_result(res, (hint_prefix + hint).strip(), pol.prompt.result_max_chars)
                call_msg = {"role": "assistant", "content": "", "tool_calls": [{"id": dec.call_id, "type": "function",
                            "function": {"name": name, "arguments": _json(args)}}]}
                units.append([call_msg, {"role": "tool", "tool_call_id": dec.call_id, "content": text}])
                del units[:-max(1, pol.prompt.history_units)]
                mem.action_log.append(f"{turn}. {name}({fmt_args(args)}) -> {'ok' if res.success else 'FAILED'}: {res.message[:70]}")
                tr.write("turn", **row, tool=name, args=args, parse_ok=parse_ok, problems=dec.problems, refused=refused,
                         success=res.success, result=res.to_dict(), hint=hint, skill_s=round(skill_s, 3), result_text=text)
                if guard_row:
                    tr.write("finish_guard", turn=turn, **guard_row)
                if mem.refusals >= pol.limits.max_refusals:
                    result = AgentResult(False, NO_PROGRESS, f"{mem.refusals} refused repeated calls")
                elif mem.consecutive_failures >= pol.limits.max_consecutive_failures:
                    result = AgentResult(False, NO_PROGRESS, f"{mem.consecutive_failures} skill failures in a row")
                elif done_reason is not None:
                    result = done_reason
                self.env.feedback(f"turn {turn}: {name}", name, float("nan"), turn)
        except Exception as ex:                        # never leave the robot task hanging on a bug
            tr.write("event", name="exception", error=f"{type(ex).__name__}: {ex}")
            result = AgentResult(False, ERROR, f"agent exception: {type(ex).__name__}: {ex}")
        result.turns = turn
        result.duration_s = self.env.now() - t0
        result.finish_refusals = mem.finish_refusals
        gave_up = bool(result.stats.get("gave_up"))        # the model itself called finish(success=false): the robot is already idle
        result.stats = {**stats, "llm": self.client.stats.summary(since=wall0), "rejected": sorted(mem.rejected),
                        "prompt_hash": pol.hash, "degraded_last": degraded_last}
        if result.stop_reason in (ABORTED, TIMEOUT, ERROR, NO_PROGRESS) and "stop" in self.registry and not gave_up:
            self._best_effort_stop()
        tr.write("summary", **result.to_dict(), trace_write_errors=tr.errors)
        tr.close()
        return result

    # ------------------------------------------------------------------ prompt
    def _safe_summary(self, pose: Pose2D | None) -> str:
        try:
            return (self.env.map_summary(pose, self.policy.prompt.summary_max_objects) or "").strip()
        except Exception as ex:                       # map service hiccup must not end the episode
            return f"(map summary unavailable: {type(ex).__name__})"

    def _build_messages(self, task, turn, max_turns, elapsed, budget_s, pose, summary, mem, units, tr):
        pol = self.policy
        want_cam, want_map = bool(pol.images.camera), bool(pol.images.map)
        cam = self.env.camera_image() if want_cam else None
        mp = self.env.map_image() if want_map else None
        images = [("map", mp), ("camera", cam)]
        images = [(n, i) for n, i in images if i is not None]
        hist = len(units)
        log_lines = pol.prompt.action_log_lines
        summary_lines = summary.splitlines()
        degraded: list[str] = []
        limit = self.context_tokens - int(pol.model.max_tokens) - int(pol.budget.safety_margin_tokens)
        itok = image_tokens_for(int(pol.images.max_side_px))
        while True:
            state = self._state_text(task, turn, max_turns, elapsed, budget_s, pose, "\n".join(summary_lines), mem, log_lines)
            msgs = [{"role": "system", "content": self.system_prompt + f"\n\nTask: {task.instruction}"}]
            for u in units[len(units) - hist:] if hist else []:
                msgs += u
            if images:
                msgs.append(user_message(state, [i for _, i in images], max_side=int(pol.images.max_side_px), quality=int(pol.images.jpeg_quality)))
            else:
                msgs.append({"role": "user", "content": state})
            est = self.est.messages(msgs, self.toolset.tools, itok)
            if est <= limit:
                break
            if any(n == "camera" for n, _ in images):
                images = [(n, i) for n, i in images if n != "camera"]; degraded.append("drop_camera")
            elif images:
                images = []; degraded.append("drop_map_image")
            elif hist > 1:
                hist -= 1; degraded.append("history-1")
            elif len(summary_lines) > 6:
                summary_lines = summary_lines[: max(6, len(summary_lines) // 2)]; degraded.append("summary_half")
            elif log_lines > 3:
                log_lines = 3; degraded.append("short_log")
            elif hist > 0:
                hist = 0; degraded.append("no_history")
            else:
                degraded.append("over_budget")
                break
        paths = []
        for n, im in images:
            try:
                paths.append(tr.save_image(f"t{turn:03d}_{n}", encode_image(im, max_side=int(pol.images.max_side_px), quality=int(pol.images.jpeg_quality))))
            except Exception:
                paths.append("")
        return msgs, paths, degraded, est

    def _state_text(self, task, turn, max_turns, elapsed, budget_s, pose, summary, mem, log_lines) -> str:
        pol = self.policy
        head = f"[Turn {turn}/{max_turns}" + (f", {elapsed:.0f}s of {budget_s:.0f}s used]" if pol.prompt.show_time else "]")
        parts = [head, f"Task: {task.instruction}"]
        if pose is not None:
            import math
            parts.append(f"Robot pose (map): x={pose.x:.2f} y={pose.y:.2f} heading={math.degrees(pose.yaw):.0f} deg")
        parts.append("Map summary:\n" + (summary or "(empty: no objects mapped yet)"))
        notes = mem.notes(log_lines)
        if notes:
            parts.append("Harness notes:\n" + notes)
        parts.append("Choose the next tool call.")
        return "\n".join(parts)

    # ------------------------------------------------------------------ model call with format retries
    def _decide(self, msgs: list[dict]) -> _Decision:
        pol = self.policy
        dec = _Decision("noop")
        work = list(msgs)
        retries = 0
        overflow_shrunk = False
        while True:
            try:
                res = self.client.chat(work, self.toolset.tools, pol.model.tool_choice, max_tokens=int(pol.model.max_tokens),
                                       temperature=float(pol.model.temperature), cancel=self.cancel)
            except VLMCancelled:
                return _Decision("cancelled", retries=retries)
            except VLMContextOverflow as ex:
                self.est.scale *= 1.25
                if overflow_shrunk or len(work) <= 2:
                    return _Decision("error", note=f"context overflow: {ex.n_prompt}/{ex.n_ctx}", retries=retries)
                overflow_shrunk = True
                work = [work[0]] + work[-1:]                        # keep system + current state only
                if ex.n_ctx:
                    self.context_tokens = min(self.context_tokens, int(ex.n_ctx))
                continue
            except VLMError as ex:
                return _Decision("error", note=f"{type(ex).__name__}: {ex}", retries=retries)
            dec = _Decision("noop", retries=retries, latency_s=res.latency_s, prompt_tokens=res.prompt_tokens,
                            completion_tokens=res.completion_tokens)
            dec.reply = {"content": res.content, "reasoning": res.reasoning[:500], "finish_reason": res.finish_reason, "notes": res.notes,
                         "tool_calls": [{"id": t.id, "name": t.name, "raw_arguments": t.raw_arguments, "repaired": t.repaired,
                                         "error": t.error, "source": t.source} for t in res.tool_calls], "timings": res.timings}
            fix: str = ""
            if not res.tool_calls:
                dec.note = "no tool call"
                fix = self.hints.text("no_tool_call")
            else:
                first = res.tool_calls[0]
                nm = first.name if find_tool(self.toolset.tools, first.name) else (closest_tool_name(self.toolset.tools, first.name) or "")
                if not nm:
                    dec.note = f"unknown tool {first.name!r}"
                    fix = self.hints.text("unknown_tool", name=first.name, tools=", ".join(self.toolset.names))
                else:
                    args, problems = coerce_arguments(self.toolset.schemas[nm], first.arguments)
                    spec = self.toolset.skills.get(nm)
                    if nm == "verify" and not args.get("question") and not args.get("label") and self._task is not None:
                        # the 8B model often sends verify(object_id) alone: ask about the task itself instead of failing the call
                        args["question"] = f"Is this what the task asks for? Task: {self._task.instruction}"
                        dec.note = "verify question filled from the task"
                    if spec is not None:
                        o = validate_one_of(spec, args)
                        if o:
                            problems.append(o)
                    if first.error:
                        problems.append(first.error)
                    fatal = [p for p in problems if is_fatal(p)]
                    dec.name, dec.args, dec.call_id, dec.problems = nm, args, first.id, problems
                    dec.extra_calls = [t.name for t in res.tool_calls[1:]]
                    if fatal and nm != "finish":
                        dec.kind, dec.invalid, dec.note = "call", True, "; ".join(fatal)
                        fix = self.hints.text("bad_arguments", name=nm, problems=dec.note)
                    else:
                        dec.kind = "call"
                        if nm == "finish":
                            args.setdefault("success", False)
                            args.setdefault("reason", "")
                        return dec
            if retries >= int(pol.limits.format_retries):
                if dec.kind == "call":            # invalid arguments after the last retry: hand the error to the model as a tool result
                    return dec
                dec.kind = "noop"
                return dec
            retries += 1
            if res.tool_calls:
                am = res.assistant_message()
                am["tool_calls"] = am["tool_calls"][:1]
                work = work + [am, {"role": "tool", "tool_call_id": res.tool_calls[0].id, "content": fix}]
            else:
                work = work + [{"role": "assistant", "content": res.content or "(no tool call)"}, {"role": "user", "content": fix}]

    # ------------------------------------------------------------------ acting
    def _act(self, task, name, args, dec, mem: HarnessMemory, turn, pose, t0, budget_s):
        """Returns (AgentResult or None, SkillResult, hint, refused, guard_row, skill_seconds)."""
        pol = self.policy
        if dec.invalid:
            res = SkillResult(False, "Invalid arguments: " + "; ".join(dec.problems))
            return None, res, self.hints.text("bad_arguments", name=name, problems="; ".join(dec.problems)), False, None, 0.0
        if name == "finish":
            return self._finish(args, mem, pose, turn)
        if name == "report_target":
            return self._report_target(args, turn, t0, budget_s)
        args = self._inject(name, args, mem)
        key = call_key(name, args)
        refusal = self._refusal(name, args, key, mem, pose)
        mem.recent_calls.append(key)
        del mem.recent_calls[:-12]
        if refusal:
            mem.refusals += 1
            return None, SkillResult(False, "refused by the harness: " + refusal[0]), refusal[1], True, None, 0.0
        t1 = time.time()
        if name == "query_map":
            res = self._query_map(args)
        else:
            spec = self.registry[name]
            timeout = min(spec.default_timeout_s + float(pol.limits.skill_timeout_margin_s), max(5.0, budget_s - (self.env.now() - t0) + 5.0))
            self.env.feedback(f"running {name}", name, float("nan"), turn)
            res = self.env.run_skill(name, args, timeout, self.cancel)
            if res.cancelled:
                return AgentResult(False, ABORTED, "cancelled during " + name), res, "", False, None, time.time() - t1
        skill_s = time.time() - t1
        blocked_before = mem.fail_counts.get(key, 0) >= 1 and "block" in res.message.lower()
        self._remember(name, args, res, key, mem, turn, pose)
        hint = self.hints.for_result(name, args, res, mem.rejected, blocked_before, self._unchecked_like(name, args, res, mem, pose))
        return None, res, hint, False, None, skill_s

    def _unchecked_like(self, name, args, res, mem: HarnessMemory, pose) -> list:
        """After a verify "no": ids of other mapped, not rejected objects with the same label, nearest first."""
        if name != "verify" or str((res.data or {}).get("answer", "")).lower() != "no":
            return []
        vid = (res.data or {}).get("object_id", args.get("object_id"))
        info = self.env.object_info(int(vid)) if vid is not None else None
        label = (info or {}).get("label") or mem.candidates.get(vid)
        if not label:
            return []
        out = []
        for oid, lab in mem.candidates.items():
            if lab == label and oid != vid and oid not in mem.rejected:
                o = self.env.object_info(int(oid))
                d = ((o["x"] - pose.x) ** 2 + (o["y"] - pose.y) ** 2) ** 0.5 if o and pose is not None else 1e9
                out.append((d, oid))
        return [oid for _, oid in sorted(out)]

    def _finish(self, args, mem, pose, turn):
        success = bool(args.get("success"))
        reason = str(args.get("reason", ""))
        d = self.guard.check(success, mem, self.env, pose)
        row = {"allowed": d.allowed, "overridden": d.overridden, "reason": d.reason, "claim_success": success, "evidence": d.evidence}
        if not d.allowed:
            res = SkillResult(False, "finish refused. " + d.message)
            mem.action_log.append(f"{turn}. finish(success=true) REFUSED: {d.reason}")
            return None, res, "", False, row, 0.0
        msg = reason + (f" [{d.message}]" if d.overridden else "")
        out = AgentResult(success, FINISHED if success else NO_PROGRESS, msg or ("success" if success else "gave up"), guard_overridden=d.overridden,
                          stats={} if success else {"gave_up": True})   # a give-up is not "finished" (eval contract)
        return out, SkillResult(True, "Task finished." + (f" {d.message}" if d.overridden else "")), "", False, row, 0.0

    def _report_target(self, args, turn, t0, budget_s):
        """locate_nav policy: the model only says WHERE the target is. The harness hands that place to the planner (goto_object /
        goto_pose = known_map_nav or Nav2 behind navigate_to_pose) in ONE call and the episode ends with the planner's outcome."""
        if args.get("found") is False:
            return (AgentResult(False, NO_PROGRESS, "gave up: " + str(args.get("reason", "")), stats={"gave_up": True}),
                    SkillResult(True, "Reported: target not found."), "", False, None, 0.0)
        if args.get("object_id") is not None:
            skill, sargs = "goto_object", {"object_id": int(args["object_id"]), "standoff_m": 1.0}
        elif args.get("x") is not None and args.get("y") is not None:
            skill, sargs = "goto_pose", {"x": float(args["x"]), "y": float(args["y"]), "tolerance_m": 1.0}
        else:
            return None, SkillResult(False, "report_target needs object_id, or x and y, or found=false."), "", False, None, 0.0
        t1 = time.time()
        timeout = max(60.0, budget_s - (self.env.now() - t0) + 5.0)
        self.env.feedback(f"running {skill}", skill, float("nan"), turn)
        res = self.env.run_skill(skill, sargs, timeout, self.cancel)
        if res.cancelled:
            return AgentResult(False, ABORTED, "cancelled during " + skill), res, "", False, None, time.time() - t1
        # the planner may have moved an unreachable goal (inside furniture / a pocket) and then reports arrival at ITS goal: the skill's distance
        # check against the original point (reason "tolerance") must not turn that into a failure. Success is scored by the ground truth anyway.
        arrived = (res.data or {}).get("reason") == "tolerance"
        ok = bool(res.success) or arrived
        out = AgentResult(ok, FINISHED if ok else NO_PROGRESS, f"reported {sargs}; planner: {res.message[:200]}" + (" [accepted: planner reported arrival at its own goal]" if arrived and not res.success else ""))
        return out, res, "", False, None, time.time() - t1

    def _inject(self, name: str, args: dict, mem: HarnessMemory) -> dict:
        a = dict(args)
        spec = self.toolset.skills.get(name)
        props = (spec.schema.get("properties", {}) if spec else {})
        if "rejected_ids" in props and mem.rejected:
            a["rejected_ids"] = sorted(set(a.get("rejected_ids", [])) | mem.rejected)
        if "object_id" in props and a.get("object_id") is None and mem.last_object is not None:
            if name == "verify" or ("label" not in a and name in ("goto_object", "approach")):
                a["object_id"] = mem.last_object
        return a

    def _refusal(self, name, args, key, mem: HarnessMemory, pose):
        L = self.policy.limits
        run = mem.repeat_run(key) + 1
        if run >= int(L.repeat_refuse_at):
            return (f"identical call repeated {run} times", self.hints.text("refused_repeat", n=run))
        if mem.fail_counts.get(key, 0) >= int(L.failed_call_refuse_at):
            return (f"identical call failed {mem.fail_counts[key]} times", self.hints.text("refused_failed", n=mem.fail_counts[key]))
        if mem.alternating(key):
            return ("alternating between two calls", self.hints.text("refused_cycle"))
        if name == "scan":
            d = mem.near_previous_scan(pose, float(L.scan_min_separation_m))
            if d is not None:
                return (f"already scanned {d:.1f} m from here", self.hints.text("refused_scan", d=d))
        return None

    def _query_map(self, args: dict) -> SkillResult:
        op = args.get("op", "list")
        try:
            r = self.env.query_map(op, **{k: v for k, v in args.items() if k != "op"})
        except Exception as ex:
            return SkillResult(False, f"map query failed: {type(ex).__name__}: {ex}")
        ok = bool(r.get("ok", True)) and not r.get("error")
        text = r.get("text") or r.get("error") or ("ok" if ok else "failed")
        data = {k: v for k, v in r.items() if k not in ("ok", "text", "error")}
        return SkillResult(ok, str(text)[:400], data)

    # ------------------------------------------------------------------ memory update
    def _remember(self, name, args, res: SkillResult, key, mem: HarnessMemory, turn, pose):
        d = res.data or {}
        if res.success:
            mem.consecutive_failures = 0
        elif name not in ("query_map", "verify"):
            mem.consecutive_failures += 1
            mem.fail_counts[key] = mem.fail_counts.get(key, 0) + 1
            mem.failed_notes.append(f"{name}({fmt_args(args, 40)}): {res.message[:50]}")
        oid = args.get("object_id")
        if name == "search":
            for c in d.get("candidates") or []:
                if isinstance(c, dict) and c.get("id") is not None:
                    mem.candidates[c["id"]] = c.get("label", "")
            best = [c for c in (d.get("candidates") or []) if isinstance(c, dict) and c.get("id") not in mem.rejected]
            if res.success and best:
                mem.last_object = int(best[0]["id"])
        elif name in ("goto_object", "approach"):
            oid = d.get("object_id", oid)
            if oid is not None:
                mem.last_object = int(oid)
            if res.success:
                mem.visited.append((pose, name))
        elif name == "explore":
            f = d.get("found")
            if isinstance(f, dict) and f.get("id") is not None:
                mem.last_object = int(f["id"])
        elif name == "scan" and res.success and pose is not None:
            mem.scans.append(pose)
        elif name == "verify":
            ans = str(d.get("answer", "")).lower()
            vid = d.get("object_id", oid)
            if ans in ("yes", "no", "unsure"):
                try:
                    pl = float(d.get("plausibility", 0) or 0)
                except (TypeError, ValueError):
                    pl = 0.0
                dist = d.get("distance_m")
                mem.verifies.append(VerifyRecord(int(vid) if vid is not None else None, ans, pl, turn, pose,
                                                 float(dist) if isinstance(dist, (int, float)) else None, str(args.get("label", ""))))
                if ans == "no" and vid is not None:
                    mem.rejected.add(int(vid))
                if ans != "no":
                    mem.consecutive_failures = 0

    def _best_effort_stop(self) -> None:
        try:
            self.env.run_skill("stop", {}, 10.0, threading.Event())
        except Exception:
            pass


def _pose_list(p: Pose2D | None):
    return None if p is None else [round(p.x, 3), round(p.y, 3), round(p.yaw, 3)]


def _json(o) -> str:
    import json
    return json.dumps(o, separators=(",", ":"), default=str)


def _plain(msgs: list[dict]) -> list[dict]:
    from x30_common.images import strip_images
    return strip_images(msgs)
