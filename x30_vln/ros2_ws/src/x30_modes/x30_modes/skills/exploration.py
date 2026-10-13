"""Frontier exploration and object search."""
from __future__ import annotations

import math

from ..primitives import parse_bearing
from ..semantic import select_candidates
from .base import SkillConfig, SkillContext, SkillResult, describe_bearing
from .motion import format_objects


class ExplorationSkills:
    def __init__(self, api, cfg: SkillConfig, motion, navigation):
        self.api, self.cfg, self.motion, self.nav = api, cfg, motion, navigation
        self._failed: list[tuple[float, float, float]] = []      # (x, y, time) of frontier goals that could not be reached

    # ------------------------------------------------------------ helpers
    def _exclude(self) -> list[list[float]]:
        now = self.api.now()
        self._failed = [f for f in self._failed if now - f[2] < self.cfg.failed_frontier_ttl_s]
        return [[f[0], f[1]] for f in self._failed]

    def bias_point(self, bias, pose) -> list[float] | None:
        """Direction word / number (relative to the robot) or an [x, y] map point -> [x, y] map point."""
        if bias is None or pose is None and not isinstance(bias, (list, tuple)):
            return None
        if isinstance(bias, (list, tuple)) and len(bias) >= 2:
            return [float(bias[0]), float(bias[1])]
        if isinstance(bias, str) and "," in bias:
            parts = bias.replace("[", "").replace("]", "").split(",")
            try:
                return [float(parts[0]), float(parts[1])]
            except ValueError:
                pass
        ang = pose.yaw + math.radians(parse_bearing(bias))
        return [pose.x + 6.0 * math.cos(ang), pose.y + 6.0 * math.sin(ang)]

    def _explore_once(self, ctx: SkillContext, strategy: str, bias, scan: bool) -> dict:
        """Visit one frontier. status: ok | exhausted | unavailable | unreachable | cancelled | timeout | error."""
        cfg, api = self.cfg, self.api
        pose = api.map_pose()
        payload: dict = {"strategy": strategy, "exclude": self._exclude()}
        if strategy == "bias" or bias is not None:
            try:
                payload["bias"] = self.bias_point(bias if bias is not None else "ahead", pose)
            except ValueError as ex:
                return {"status": "error", "message": f"{ex}"}
        resp = api.next_frontier(payload, ctx.cancel_event)
        if ctx.stop_reason():
            return {"status": ctx.stop_reason(), "message": ""}
        if resp is None:
            return {"status": "unavailable", "message": ""}
        if not resp.get("ok", True):
            return {"status": "error", "message": str(resp.get("message") or resp.get("error") or "frontier service reported an error")}
        if not resp.get("found", False):
            return {"status": "exhausted", "message": str(resp.get("reason", "")), "remaining": 0}
        goal = resp.get("goal_pose") or resp.get("goal") or {k: resp[k] for k in ("x", "y", "yaw") if k in resp}
        if isinstance(goal, (list, tuple)):  # frontier_explorer style [x, y(, yaw)]
            goal = dict(zip(("x", "y", "yaw"), goal))
        gx, gy = float(goal["x"]), float(goal["y"])
        yaw = float(goal["yaw"]) if goal.get("yaw") is not None else None
        sub = ctx.child(cfg.explore_goal_timeout_s)
        r = self.nav.nav_to(sub, gx, gy, yaw, cfg.goal_tolerance_m + 0.3)
        if not r["ok"]:
            if ctx.stop_reason():
                return {"status": ctx.stop_reason(), "goal": (gx, gy), "message": ""}
            self._failed.append((gx, gy, api.now()))
            return {"status": "unreachable", "goal": (gx, gy), "reason": r["reason"], "remaining": resp.get("remaining"),
                    "message": self.nav._nav_failure_message(r, f"frontier ({gx:.1f}, {gy:.1f})", cfg.goal_tolerance_m)}
        if scan:
            self.motion.scan(ctx, {"sweep_deg": 360, "step_deg": cfg.explore_scan_step_deg, "dwell_s": cfg.explore_scan_dwell_s})
        return {"status": "ok", "goal": (gx, gy), "remaining": resp.get("remaining"), "size": resp.get("size")}

    UNAVAILABLE = ("The frontier service {s} is not available (navigation / frontier explorer not running). "
                   "Next: use goto_pose to a known free point, or move_relative and scan to look around manually.")

    # ------------------------------------------------------------ skills
    def explore(self, ctx: SkillContext, args: dict) -> SkillResult:
        cfg, api = self.cfg, self.api
        strategy, steps = args.get("strategy", "informative"), int(args.get("steps", 1))
        until, do_scan = args.get("until_label"), bool(args.get("scan", True))
        bias = args.get("bias")
        before = {o.id for o in api.objects()}
        visited, fails, last_remaining = [], 0, None
        data = {"visited": visited, "new_objects": [], "remaining_frontiers": None, "exhausted": False, "found": None}
        stop_msg = ""
        for i in range(steps):
            if until:
                found = select_candidates(api.objects(), until, (), cfg.confident_instance)
                if found:
                    data["found"] = found[0].to_dict()
                    stop_msg = f"Found a '{until}' (#{found[0].id}) in the map, stopped exploring."
                    break
            r = self._explore_once(ctx, strategy, bias, do_scan)
            st = r["status"]
            if st == "ok":
                visited.append(list(r["goal"]))
                last_remaining = r.get("remaining")
                fails = 0
                ctx.feedback(f"explored frontier {i + 1}/{steps}", (i + 1) / steps)
            elif st == "exhausted":
                data["exhausted"] = True
                data["remaining_frontiers"] = 0
                break
            elif st == "unavailable":
                return SkillResult(False, self.UNAVAILABLE.format(s=cfg.frontier_service), {**data, "reason": "no_frontier_service"})
            elif st in ("cancelled", "timeout"):
                stop_msg = f"Exploration {'cancelled' if st == 'cancelled' else 'ran out of time'} after {len(visited)} frontier(s)."
                break
            else:                                              # unreachable / error
                fails += 1
                if fails >= cfg.max_consecutive_failures:
                    return SkillResult(False, f"{fails} frontiers in a row could not be reached; last: {r['message']} "
                                       "Next: move_relative / turn_to to get out of a tight spot, or goto_pose to a known free point.",
                                       {**data, "reason": r.get("reason", st)})
        data["remaining_frontiers"] = last_remaining if not data["exhausted"] else 0
        pose = api.map_pose()
        new = [o for o in api.objects() if o.id not in before]
        data["new_objects"] = [o.to_dict((pose.x, pose.y), pose.yaw) if pose else o.to_dict() for o in new]
        seen = f" New objects: {format_objects(new, pose)}." if new else " No new objects seen."
        if data["exhausted"] and not visited:
            return SkillResult(False, "No unexplored frontier is left: the reachable map is fully explored. The target is not in the explored area "
                               "(or was missed). Next: scan, search with another label, verify a doubtful object, or finish with failure.", data)
        if data["exhausted"]:
            return SkillResult(False, f"Explored {len(visited)} frontier(s), now nothing is left: the reachable map is fully explored.{seen} "
                               "Next: search with another label, scan, or finish with failure.", data)
        if stop_msg:
            return SkillResult(data["found"] is not None, stop_msg + seen, data)
        rem = f" {last_remaining} frontiers remain." if last_remaining is not None else ""
        where = ", ".join(f"({v[0]:.1f}, {v[1]:.1f})" for v in visited) or "none"
        return SkillResult(bool(visited), f"Explored {len(visited)} frontier(s): {where}.{rem}{seen}", data)

    def search(self, ctx: SkillContext, args: dict) -> SkillResult:
        cfg, api = self.cfg, self.api
        label = str(args["label"])
        rejected = list(args.get("rejected_ids", []))
        budget = float(args.get("budget_s", 120.0))
        min_conf = float(args.get("min_confidence", cfg.confident_instance))
        attrs = args.get("attributes", [])
        t_end = ctx.now() + budget
        tried_leads: set[int] = set()
        hops, explored, exhausted, note = 0, 0, False, ""
        while True:
            pose = api.map_pose()
            xy = (pose.x, pose.y) if pose else None
            cands = select_candidates(api.query_objects(label=label), label, rejected, min_conf, False, xy, attrs)
            if cands:
                return self._found(label, cands, pose, explored)
            reason = ctx.stop_reason()
            if reason:
                return SkillResult(False, f"Search for '{label}' {'cancelled' if reason == 'cancelled' else 'hit the skill time limit'} after exploring "
                                   f"{explored} frontier(s); nothing found yet.", {"candidates": [], "explored": explored > 0, "exhausted": False})
            if ctx.now() >= t_end:
                break
            leads = [o for o in select_candidates(api.objects(), label, set(rejected) | tried_leads, 0.0, True, xy) if o.bearing_only]
            if leads and hops < cfg.search_bearing_hops:
                lead = leads[0]
                tried_leads.add(lead.id)
                hops += 1
                goal, _ = self.nav.standoff_goal(lead, 2.5)
                if goal is not None:
                    ctx.feedback(f"search: moving toward a distant {label}", 0.0)
                    r = self.nav.nav_to(ctx.child(cfg.explore_goal_timeout_s), goal.x, goal.y, goal.yaw, cfg.goal_tolerance_m + 0.5)
                    if r["ok"]:
                        self.motion.scan(ctx, {"sweep_deg": 120, "step_deg": 60, "dwell_s": cfg.explore_scan_dwell_s})
                    continue
            ctx.feedback(f"search: exploring for {label}", min((ctx.now() - (t_end - budget)) / budget, 0.99))
            r = self._explore_once(ctx.child(max(t_end - ctx.now(), 5.0) + cfg.explore_goal_timeout_s), "informative", None, True)
            st = r["status"]
            if st == "ok":
                explored += 1
            elif st == "exhausted":
                exhausted = True
                break
            elif st == "unavailable":
                note = f" (the frontier service {cfg.frontier_service} is not available, so nothing could be explored)"
                break
            elif st == "error":
                note = f" ({r['message']})"
                break
            # unreachable frontiers are remembered in self._failed and skipped by the next request; cancel/timeout end the loop above
        nobj = len(api.objects())
        src = "" if api.have_object_source() else " The semantic map is not publishing (no /semantic_map/objects), so detections cannot be seen."
        data = {"candidates": [], "explored": explored > 0, "exhausted": exhausted}
        if exhausted:
            return SkillResult(False, f"'{label}' not found: explored {explored} frontier(s) and nothing is left to explore{note}; the map holds {nobj} objects, "
                               f"none a '{label}' (rejected: {sorted(rejected) or 'none'}).{src} Next: try a synonym label, scan again, "
                               "verify a doubtful object from the map, or finish with failure.", data)
        return SkillResult(False, f"'{label}' not found within the {budget:.0f} s budget after exploring {explored} frontier(s){note}; the map holds {nobj} objects.{src} "
                           "Next: call search again with a larger budget_s, or another label/synonym.", data)

    def _found(self, label: str, cands, pose, explored: int) -> SkillResult:
        xy = (pose.x, pose.y) if pose else None
        rows = [o.to_dict(xy, pose.yaw if pose else None) for o in cands[:5]]
        for r, o in zip(rows, cands):
            r["attributes"] = o.attributes
        best = cands[0]
        desc = format_objects(cands, pose, 5)
        extra = f" {len(cands) - 1} other candidate(s) listed; if the first is wrong, repeat with it in rejected_ids." if len(cands) > 1 else ""
        return SkillResult(True, f"Found {len(cands)} '{label}' candidate(s), best first: {desc}.{extra} "
                           f"Next: goto_object(object_id={best.id}) then verify.", {"candidates": rows, "explored": explored > 0, "exhausted": False})
