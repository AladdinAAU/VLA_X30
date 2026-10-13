"""Non-LLM baselines behind the same agent interfaces, for ablations.

* ScriptedSearch (object_nav): explore until the semantic map has the target label, approach it, optionally verify, finish.
  No language model decides anything; `verify` (a skill) still uses the VLM unless use_verify is false.
* DeadReckon (velocity_follow / goal): execute a reference path (waypoints) with turn_to / move_relative, re-aiming from odometry
  after every leg. No perception at all.
Both write the same trace format and return AgentResult, and honour cancel.
"""
from __future__ import annotations

import math
import re
import threading
import time

from .guard import FinishGuard
from .hints import HintEngine
from .memory import HarnessMemory, VerifyRecord, call_key
from .policy import Policy
from .trace import TraceWriter
from .types import (ABORTED, ERROR, FINISHED, NO_PROGRESS, SAFETY, TIMEOUT, AgentEnv, AgentResult, Pose2D, SkillResult, Task)

_LABEL_RE = re.compile(r"(?:find|go to|locate|reach|navigate to|look for|stop (?:at|next to|near))\s+(?:the|a|an)\s+(.+?)(?:\s+(?:in|near|next|and|on|at|by)\b|[.,;]|$)", re.I)
_COLORS = {"red", "blue", "green", "white", "black", "yellow", "orange", "brown", "grey", "gray", "silver", "purple", "pink"}


def target_label_from(instruction: str) -> str:
    m = _LABEL_RE.search(instruction)
    words = (m.group(1) if m else instruction).split()
    words = [w for w in words if w.lower() not in _COLORS]
    return " ".join(words).strip().lower()


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class ScriptedSearch:
    def __init__(self, env: AgentEnv, registry: dict, policy: Policy, trace_dir: str | None = None,
                 cancel: threading.Event | None = None, use_verify: bool = True, max_rejects: int = 12):
        self.env, self.registry, self.policy = env, registry, policy
        self.trace_dir, self.cancel = trace_dir, cancel or threading.Event()
        self.use_verify, self.max_rejects = use_verify, max_rejects
        self.guard = FinishGuard(policy.guard, HintEngine(policy.hints))

    def _call(self, name: str, args: dict, mem: HarnessMemory, tr: TraceWriter, turn: int) -> SkillResult:
        t = time.time()
        res = self.env.run_skill(name, args, float(self.registry[name].default_timeout_s + 10), self.cancel)
        mem.recent_calls.append(call_key(name, args))
        tr.write("turn", turn=turn, prompt_hash="scripted", tool=name, args=args, parse_ok=True, success=res.success, result=res.to_dict(),
                 latency_s=0.0, skill_s=round(time.time() - t, 3), pose=_pl(self.env.pose()))
        return res

    def run(self, task: Task, target_label: str = "") -> AgentResult:
        label = target_label or task.target_label or target_label_from(task.instruction)
        t0 = self.env.now()
        budget_s = float(task.time_budget_s or 300)
        max_turns = int(task.max_turns or 40)
        mem = HarnessMemory()
        tr = TraceWriter(self.trace_dir, task.episode_id, enabled=bool(self.trace_dir))
        tr.write("header", episode=task.episode_id, mode="object_nav", agent="scripted_search", task=task.instruction, label=label, prompt_hash="scripted")
        turn, fails, explores = 0, 0, 0
        res_out: AgentResult | None = None
        while res_out is None:
            if self.cancel.is_set():
                res_out = AgentResult(False, ABORTED, "cancelled"); break
            if self.env.now() - t0 > budget_s:
                res_out = AgentResult(False, TIMEOUT, "time budget used"); break
            if turn >= max_turns:
                res_out = AgentResult(False, TIMEOUT, "turn limit reached"); break
            turn += 1
            r = self._call("search", {"label": label, "rejected_ids": sorted(mem.rejected), "budget_s": 60}, mem, tr, turn)
            if r.cancelled:
                res_out = AgentResult(False, ABORTED, "cancelled"); break
            cands = [c for c in (r.data.get("candidates") or []) if isinstance(c, dict) and c.get("id") not in mem.rejected]
            if not (r.success and cands):
                if r.data.get("exhausted") or fails >= 3:
                    res_out = AgentResult(False, NO_PROGRESS, f"'{label}' not found: map exhausted"); break
                fails += 1
                explores += 1
                turn += 1
                self._call("explore", {"strategy": "informative", "until_label": label}, mem, tr, turn)
                if explores % 2 == 0:
                    turn += 1
                    self._call("scan", {}, mem, tr, turn)
                continue
            oid = int(cands[0]["id"])
            turn += 1
            g = self._call("goto_object", {"object_id": oid, "standoff_m": 1.0}, mem, tr, turn)
            if not g.success:
                fails += 1
                if fails >= 2:
                    mem.rejected.add(oid)
                continue
            if self.use_verify:
                turn += 1
                v = self._call("verify", {"object_id": oid, "question": f"Is this what the task asks for? Task: {task.instruction}"}, mem, tr, turn)
                ans = str(v.data.get("answer", "")).lower()
                try:
                    pl = float(v.data.get("plausibility", 0) or 0)
                except (TypeError, ValueError):
                    pl = 0.0
                mem.verifies.append(VerifyRecord(oid, ans, pl, turn, self.env.pose(), v.data.get("distance_m")))
                if ans != "yes":
                    mem.rejected.add(oid)
                    if len(mem.rejected) > self.max_rejects:
                        res_out = AgentResult(False, NO_PROGRESS, "too many rejected candidates"); break
                    continue
            d = self.guard.check(True, mem, self.env, self.env.pose()) if self.use_verify else None
            tr.write("finish_guard", turn=turn, allowed=True if d is None else d.allowed, claim_success=True)
            res_out = AgentResult(True, FINISHED, f"scripted: reached #{oid} {label}")
        res_out.turns = turn
        res_out.duration_s = self.env.now() - t0
        res_out.stats = {"agent": "scripted_search", "rejected": sorted(mem.rejected), "llm": {}}
        tr.write("summary", **res_out.to_dict())
        tr.close()
        return res_out


def world_to_map_path(path: list, start_world: tuple, pose_map: Pose2D) -> list[tuple[float, float]]:
    """Reference path given in the simulator world frame -> the robot's map frame, anchored at the start pose.
    start_world = (x, y, yaw) of the episode start in the world frame; pose_map = the robot's pose in the map frame at start."""
    sx, sy, syaw = start_world
    c, s = math.cos(-syaw), math.sin(-syaw)
    out = []
    for p in path:
        dx, dy = p[0] - sx, p[1] - sy
        rx, ry = c * dx - s * dy, s * dx + c * dy                      # in the start-pose frame
        cm, sm = math.cos(pose_map.yaw), math.sin(pose_map.yaw)
        out.append((pose_map.x + cm * rx - sm * ry, pose_map.y + sm * rx + cm * ry))
    return out


class DeadReckon:
    def __init__(self, env: AgentEnv, trace_dir: str | None = None, cancel: threading.Event | None = None, turn_tol_deg: float = 6.0,
                 reach_tol_m: float = 0.35, max_leg_cm: float = 300.0, max_iters_per_wp: int = 8, max_blocked: int = 3):
        self.env, self.trace_dir, self.cancel = env, trace_dir, cancel or threading.Event()
        self.turn_tol, self.reach_tol, self.max_leg = math.radians(turn_tol_deg), reach_tol_m, max_leg_cm
        self.max_iters, self.max_blocked = max_iters_per_wp, max_blocked

    def run(self, task: Task, waypoints: list | None = None) -> AgentResult:
        t0 = self.env.now()
        budget_s = float(task.time_budget_s or 300)
        wps = [tuple(w[:2]) for w in (waypoints if waypoints is not None else task.reference_path)]
        if not wps and task.goal_pose is not None:
            wps = [(task.goal_pose.x, task.goal_pose.y)]
        tr = TraceWriter(self.trace_dir, task.episode_id, enabled=bool(self.trace_dir))
        tr.write("header", episode=task.episode_id, mode=task.mode, agent="dead_reckon", task=task.instruction, waypoints=wps, prompt_hash="dead_reckon")
        step, blocked_run = 0, 0
        out: AgentResult | None = None
        if not wps:
            out = AgentResult(False, ERROR, "dead_reckon needs a reference path or goal pose")
        for wi, (wx, wy) in enumerate(wps):
            for _ in range(self.max_iters):
                if out is not None:
                    break
                if self.cancel.is_set():
                    out = AgentResult(False, ABORTED, "cancelled"); break
                if self.env.now() - t0 > budget_s:
                    out = AgentResult(False, TIMEOUT, "time budget used"); break
                pose = self.env.pose()
                if pose is None:
                    out = AgentResult(False, ERROR, "no pose"); break
                dist = math.hypot(wx - pose.x, wy - pose.y)
                if dist < self.reach_tol:
                    break
                err = wrap(math.atan2(wy - pose.y, wx - pose.x) - pose.yaw)
                if abs(err) > self.turn_tol:
                    step += 1
                    r = self.env.run_skill("turn_to", {"degrees": math.degrees(err)}, 30.0, self.cancel)
                    tr.write("turn", turn=step, tool="turn_to", args={"degrees": round(math.degrees(err), 1)}, success=r.success, result=r.to_dict(), waypoint=wi, parse_ok=True)
                    if r.cancelled:
                        out = AgentResult(False, ABORTED, "cancelled"); break
                step += 1
                leg = min(self.max_leg, max(10.0, dist * 100.0))
                r = self.env.run_skill("move_relative", {"direction": "forward", "distance_cm": leg}, 40.0, self.cancel)
                tr.write("turn", turn=step, tool="move_relative", args={"distance_cm": round(leg, 1)}, success=r.success, result=r.to_dict(), waypoint=wi, parse_ok=True,
                         pose=_pl(self.env.pose()))
                if r.cancelled:
                    out = AgentResult(False, ABORTED, "cancelled"); break
                blocked = bool(r.data.get("blocked"))
                blocked_run = blocked_run + 1 if blocked else 0
                if blocked_run >= self.max_blocked:
                    out = AgentResult(False, SAFETY, f"blocked {blocked_run} times on the way to waypoint {wi}"); break
            if out is not None:
                break
        if out is None:
            out = AgentResult(True, FINISHED, f"dead_reckon: {len(wps)} waypoint(s) executed")
        out.turns = step
        out.duration_s = self.env.now() - t0
        out.stats = {"agent": "dead_reckon", "llm": {}}
        if out.stop_reason != FINISHED:
            try:
                self.env.run_skill("stop", {}, 10.0, threading.Event())
            except Exception:
                pass
        tr.write("summary", **out.to_dict())
        tr.close()
        return out


def _pl(p: Pose2D | None):
    return None if p is None else [round(p.x, 3), round(p.y, 3), round(p.yaw, 3)]
