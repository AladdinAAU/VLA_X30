"""Velocity-primitive skills: move_relative, turn_to, scan. Closed loop on odometry, abort when the safety guard blocks."""
from __future__ import annotations

import math

from ..geometry import Pose2D, bearing_to, wrap_pi
from ..primitives import TranslateController, TurnController, parse_bearing
from .base import SkillConfig, SkillContext, SkillResult, describe_bearing


def format_objects(objs, pose: Pose2D | None, limit: int = 6) -> str:
    """'chair #3 (2.1 m, 30 deg to the left), ...' for the reasoner."""
    items = []
    for o in objs[:limit]:
        if pose is not None:
            d = o.to_dict((pose.x, pose.y), pose.yaw)
            items.append(f"{o.label} #{o.id} ({d['distance_m']:.1f} m, {describe_bearing(d['bearing_deg'])})")
        else:
            items.append(f"{o.label} #{o.id}")
    more = f" and {len(objs) - limit} more" if len(objs) > limit else ""
    return ", ".join(items) + more


class MotionSkills:
    def __init__(self, api, cfg: SkillConfig):
        self.api, self.cfg = api, cfg

    # ------------------------------------------------------------ closed-loop executors
    def _guard_abort(self) -> str:
        """'' or a reason string when the guard has blocked the robot long enough / an e-stop is active."""
        sf = self.api.safety()
        if sf is not None and sf.stopped and sf.reason == "estop":
            return "estop"
        if self.api.blocked_for() > self.cfg.primitive_blocked_abort_s:
            return sf.reason if sf is not None and sf.reason else "obstacle"
        return ""

    def translate(self, ctx: SkillContext, direction: str, distance_m: float) -> dict:
        """Returns {'achieved_m', 'reason' ('' = done), 'clearance_m'}."""
        cfg, api = self.cfg, self.api
        pose0 = api.odom_pose()
        if pose0 is None:
            return {"achieved_m": 0.0, "reason": "no_odom", "clearance_m": math.inf}
        v_max = {"forward": cfg.lin_speed, "backward": cfg.back_speed}.get(direction, cfg.strafe_speed)
        ctrl = TranslateController(pose0, direction, distance_m, v_max, cfg.lin_accel, cfg.lin_decel,
                                   min_speed=min(0.06, v_max), tol=cfg.move_tol_m, coast_s=cfg.coast_s)
        t0, period = ctx.now(), 1.0 / cfg.primitive_rate_hz
        last_gain, best = t0, 0.0
        reason, clearance = "", math.inf
        while True:
            reason = ctx.stop_reason()
            if reason:
                break
            pose = api.odom_pose()
            if pose is None:
                reason = "no_odom"
                break
            cmd, done = ctrl.step(pose, ctx.now() - t0)
            if done:
                break
            reason = self._guard_abort()
            if reason:
                sf = api.safety()
                clearance = sf.min_obstacle_distance if sf is not None else math.inf
                break
            if ctrl.travelled > best + 0.01:
                best, last_gain = ctrl.travelled, ctx.now()
            elif ctx.now() - last_gain > 3.0:
                reason = "stalled"
                break
            api.send_cmd(*cmd)
            ctx.sleep(period)
        api.halt()
        ctx.sleep(0.5)                     # let the locomotion come to rest before measuring
        final = api.odom_pose() or pose0
        along, _ = ctrl.progress(final)
        return {"achieved_m": along, "reason": reason, "clearance_m": clearance}

    def rotate(self, ctx: SkillContext, delta: float, w_max: float | None = None) -> dict:
        """Rotate in place by `delta` rad (left positive). Returns {'achieved_rad', 'reason'}."""
        cfg, api = self.cfg, self.api
        pose0 = api.odom_pose()
        if pose0 is None:
            return {"achieved_rad": 0.0, "reason": "no_odom"}
        ctrl = TurnController(pose0.yaw, delta, w_max or cfg.turn_speed, cfg.turn_accel, cfg.turn_decel,
                              tol=math.radians(cfg.turn_tol_deg), coast_s=cfg.coast_s)
        t0, period = ctx.now(), 1.0 / cfg.primitive_rate_hz
        last_gain, best, reason = t0, 0.0, ""
        while True:
            reason = ctx.stop_reason()
            if reason:
                break
            pose = api.odom_pose()
            if pose is None:
                reason = "no_odom"
                break
            cmd, done = ctrl.step(pose.yaw, ctx.now() - t0)
            if done:
                break
            reason = self._guard_abort()
            if reason:
                break
            if abs(ctrl.turned) > best + math.radians(1.0):
                best, last_gain = abs(ctrl.turned), ctx.now()
            elif ctx.now() - last_gain > 4.0:
                reason = "stalled"
                break
            api.send_cmd(*cmd)
            ctx.sleep(period)
        api.halt()
        ctx.sleep(0.5)
        final = api.odom_pose()
        if final is not None:
            ctrl.tracker.update(final.yaw)
        return {"achieved_rad": ctrl.turned, "reason": reason}

    # ------------------------------------------------------------ skills
    def move_relative(self, ctx: SkillContext, args: dict) -> SkillResult:
        direction, dist_cm = args.get("direction", "forward"), float(args["distance_cm"])
        r = self.translate(ctx, direction, dist_cm / 100.0)
        got_cm = r["achieved_m"] * 100.0
        data = {"requested_cm": dist_cm, "achieved_cm": round(got_cm, 1), "error_cm": round(got_cm - dist_cm, 1),
                "blocked": r["reason"] in ("obstacle", "no_lidar", "tilt"),
                "obstacle_distance_cm": None if math.isinf(r["clearance_m"]) else round(r["clearance_m"] * 100.0, 0),
                "reason": r["reason"] or "done"}
        side = {"forward": "ahead", "backward": "behind", "left": "to the left", "right": "to the right"}[direction]
        if not r["reason"]:
            return SkillResult(True, f"Moved {direction} {dist_cm:.0f} cm (measured {got_cm:.0f} cm, error {got_cm - dist_cm:+.0f} cm).", data)
        partial = f"Moved {direction} {got_cm:.0f} of {dist_cm:.0f} cm, then "
        if r["reason"] == "obstacle":
            clr = f" {r['clearance_m'] * 100:.0f} cm" if not math.isinf(r["clearance_m"]) else ""
            return SkillResult(False, partial + f"stopped by the safety guard: obstacle{clr} {side}. "
                               "Next: turn_to another direction, move_relative the other way, or goto_pose around it.", data)
        if r["reason"] == "estop":
            return SkillResult(False, partial + "the soft e-stop is active. Nothing can move until it is released; call stop and tell the operator.", data)
        if r["reason"] == "no_odom":
            return SkillResult(False, "No odometry (TF odom->base_link) is available, so no closed-loop motion is possible. Check that the robot/simulator is running.", data)
        if r["reason"] == "stalled":
            return SkillResult(False, partial + "the robot stopped making progress (commands are not being executed?). Check /x30/ready and the locomotion, then retry a short move.", data)
        if r["reason"] == "cancelled":
            return SkillResult(False, partial + "the skill was cancelled.", data)
        return SkillResult(False, partial + f"the time limit ran out ({r['reason']}). Try a shorter distance.", data)

    def _turn_delta(self, args: dict) -> tuple[float | None, str]:
        """Requested rotation in rad from the turn_to arguments, or (None, error message)."""
        if "degrees" in args:
            return math.radians(float(args["degrees"])), ""
        if "bearing" in args:
            try:
                return math.radians(parse_bearing(args["bearing"])), ""
            except ValueError as ex:
                return None, str(ex)
        pose = self.api.map_pose()
        if "object_id" in args:
            o = self.api.get_object(int(args["object_id"]))
            if o is None:
                known = ", ".join(str(x.id) for x in self.api.objects()[:15]) or "none"
                return None, f"Object #{args['object_id']} is not in the semantic map (known ids: {known}). Use search or scan to find it."
            if pose is None:
                return None, "No map pose (TF map->base_link) yet, so the direction to an object is unknown. Wait for localisation or use a relative turn."
            return bearing_to(pose, o.x, o.y), ""
        if "heading_deg" in args:
            if pose is None:
                return None, "No map pose (TF map->base_link) yet, so an absolute heading cannot be used. Use degrees or bearing instead."
            return wrap_pi(math.radians(float(args["heading_deg"])) - pose.yaw), ""
        return None, "turn_to needs one of degrees, bearing, object_id, heading_deg."

    def turn_to(self, ctx: SkillContext, args: dict) -> SkillResult:
        delta, err = self._turn_delta(args)
        if delta is None:
            return SkillResult(False, err, {"reason": "bad_target"})
        r = self.rotate(ctx, delta)
        req, got = math.degrees(delta), math.degrees(r["achieved_rad"])
        pose = self.api.map_pose()
        data = {"requested_deg": round(req, 1), "achieved_deg": round(got, 1), "error_deg": round(got - req, 1),
                "heading_map_deg": None if pose is None else round(math.degrees(pose.yaw), 1), "reason": r["reason"] or "done"}
        if not r["reason"]:
            where = "" if data["heading_map_deg"] is None else f" Now facing {data['heading_map_deg']:.0f} deg in the map."
            return SkillResult(True, f"Turned {'left' if req >= 0 else 'right'} {abs(got):.0f} deg (requested {abs(req):.0f}, error {got - req:+.0f}).{where}", data)
        why = {"obstacle": "the safety guard blocked the rotation (obstacle very close to the body); move_relative away from it first",
               "estop": "the soft e-stop is active", "no_odom": "no odometry is available",
               "stalled": "the robot stopped rotating (locomotion not executing commands?)", "cancelled": "the skill was cancelled"}.get(
            r["reason"], "the time limit ran out")
        return SkillResult(False, f"Turned {abs(got):.0f} of {abs(req):.0f} deg, then stopped: {why}.", data)

    def scan(self, ctx: SkillContext, args: dict) -> SkillResult:
        sweep, step, dwell = float(args.get("sweep_deg", 360)), float(args.get("step_deg", 60)), float(args.get("dwell_s", 1.0))
        before = {o.id for o in self.api.objects()}
        n = max(int(math.ceil(sweep / step - 1e-6)), 1)
        done_deg, stops, reason = 0.0, 0, ""
        for i in range(n):
            ang = min(step, sweep - i * step)
            ctx.feedback(f"scan stop {i + 1}/{n}", (i + 1) / n)
            r = self.rotate(ctx, math.radians(ang))
            done_deg += math.degrees(r["achieved_rad"])
            if r["reason"]:
                reason = r["reason"]
                break
            stops += 1
            ctx.sleep(dwell)
            if ctx.stop_reason():
                reason = ctx.stop_reason()
                break
        ctx.sleep(1.2)                                       # let perception and the map integrate the last view
        pose = self.api.map_pose()
        objs = self.api.objects()
        new = [o for o in objs if o.id not in before]
        if pose is not None:
            objs.sort(key=lambda o: math.hypot(o.x - pose.x, o.y - pose.y))
            near = [o for o in objs if math.hypot(o.x - pose.x, o.y - pose.y) <= 6.0]
        else:
            near = objs
        data = {"new_objects": [o.to_dict((pose.x, pose.y), pose.yaw) if pose else o.to_dict() for o in new],
                "nearby": [o.to_dict((pose.x, pose.y), pose.yaw) if pose else o.to_dict() for o in near[:10]], "stops": stops}
        head = f"Scanned {done_deg:.0f} deg in {stops} stops."
        if reason:
            head = f"Scan interrupted after {done_deg:.0f} deg ({reason})."
        if not objs:
            body = " The semantic map is empty: nothing detected yet. If this persists, perception or the semantic map may not be running."
        else:
            body = (f" New objects: {format_objects(new, pose)}." if new else " No new objects.") + \
                   (f" Nearby: {format_objects(near, pose)}." if near else " Nothing mapped within 6 m.")
        return SkillResult(not reason, head + body, data)
