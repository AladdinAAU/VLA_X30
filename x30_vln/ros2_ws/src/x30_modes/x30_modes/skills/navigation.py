"""Planner-based and close-range navigation skills: goto_pose, goto_object, approach."""
from __future__ import annotations

import math
import time

import numpy as np

from ..geometry import Pose2D, bearing_to, quat_from_yaw, standoff_pose, wrap_pi
from ..semantic import ObjectRecord, select_candidates
from .base import SkillConfig, SkillContext, SkillResult, describe_bearing

# action_msgs/GoalStatus codes
STATUS_SUCCEEDED, STATUS_CANCELED, STATUS_ABORTED = 4, 5, 6


class NavigationSkills:
    def __init__(self, api, cfg: SkillConfig, motion, group):
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        self.api, self.cfg, self.motion = api, cfg, motion
        self._goal_cls = NavigateToPose.Goal
        self.client = ActionClient(api.node, NavigateToPose, cfg.nav2_action, callback_group=group)

    # ------------------------------------------------------------ Nav2 wrapper
    def nav_to(self, ctx: SkillContext, x: float, y: float, yaw: float | None, tol: float) -> dict:
        """Drive to (x, y) in the map frame with Nav2. Returns {'ok', 'reason', 'remaining_m', 'pose', 'clearance_m'}."""
        cfg, api = self.cfg, self.api
        out = {"ok": False, "reason": "", "remaining_m": math.nan, "pose": None, "clearance_m": math.inf}
        if not self.client.wait_for_server(timeout_sec=3.0):
            out["reason"] = "no_nav2"
            return out
        start = api.map_pose()
        if yaw is None:
            yaw = math.atan2(y - start.y, x - start.x) if start is not None else 0.0
        goal = self._goal_cls()
        goal.pose.header.frame_id = cfg.map_frame
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        q = quat_from_yaw(yaw)
        goal.pose.pose.orientation.x, goal.pose.pose.orientation.y, goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = q
        remaining = [math.nan]

        def on_fb(fb):
            remaining[0] = float(fb.feedback.distance_remaining)

        gfut = self.client.send_goal_async(goal, feedback_callback=on_fb)
        t_acc = time.monotonic() + 5.0
        while not gfut.done():
            if time.monotonic() > t_acc or ctx.stop_reason():
                out["reason"] = ctx.stop_reason() or "nav2_no_response"
                return out
            time.sleep(0.02)
        gh = gfut.result()
        if gh is None or not gh.accepted:
            out["reason"] = "rejected"
            return out
        rfut = gh.get_result_async()
        ref_pose, ref_t = start, ctx.now()
        reason = ""
        while not rfut.done():
            reason = ctx.stop_reason()
            if not reason and api.blocked_for() > cfg.nav_blocked_abort_s:
                reason = "blocked"
                sf = api.safety()
                out["clearance_m"] = sf.min_obstacle_distance if sf is not None else math.inf
            if not reason:
                p = api.map_pose()
                if p is not None and ref_pose is not None:
                    if math.hypot(p.x - ref_pose.x, p.y - ref_pose.y) >= cfg.nav_stuck_distance_m:
                        ref_pose, ref_t = p, ctx.now()
                    elif ctx.now() - ref_t > cfg.nav_stuck_s:
                        reason = "stuck"
                elif p is not None:
                    ref_pose, ref_t = p, ctx.now()
            if reason:
                cf = gh.cancel_goal_async()
                t_end = time.monotonic() + 5.0
                while not (cf.done() and rfut.done()) and time.monotonic() < t_end:
                    time.sleep(0.02)
                api.halt()
                break
            ctx.feedback(f"navigating, {remaining[0]:.1f} m left" if not math.isnan(remaining[0]) else "navigating", 0.0)
            time.sleep(0.1)
        pose = api.map_pose()
        out["pose"] = pose
        if pose is not None:
            out["remaining_m"] = math.hypot(x - pose.x, y - pose.y)
        if reason:
            out["reason"] = reason
            return out
        status = rfut.result().status
        if status == STATUS_SUCCEEDED:
            if out["remaining_m"] <= tol or math.isnan(out["remaining_m"]):
                out["ok"] = True
            else:
                out["reason"] = "tolerance"
        else:
            out["reason"] = "nav2_canceled" if status == STATUS_CANCELED else "nav2_aborted"
        return out

    def _nav_failure_message(self, r: dict, what: str, tol: float) -> str:
        rem = r["remaining_m"]
        left = f"{rem:.1f} m from the target" if not math.isnan(rem) else "short of the target"
        reason = r["reason"]
        if reason == "no_nav2":
            return ("Nav2 is not available (no navigate_to_pose action server). Navigation is probably not running. "
                    "Next: use move_relative / turn_to for short moves, or tell the operator.")
        if reason == "nav2_aborted":
            return (f"Could not reach {what}: Nav2 found no valid path or the controller failed ({left}). The target may be inside an "
                    "obstacle, in unknown space, or behind a closed passage. Next: pick a free point near it (see the map summary), "
                    "or explore to reveal more of the map, then retry.")
        if reason == "blocked":
            c = r["clearance_m"]
            near = f" (obstacle {c * 100:.0f} cm from the body)" if not math.isinf(c) else ""
            return (f"Stopped on the way to {what}: the safety guard has held the robot for more than "
                    f"{self.cfg.nav_blocked_abort_s:.0f} s{near}, {left}. Next: move_relative backward 50 cm or turn_to another heading, "
                    "then retry or choose a different target.")
        if reason == "stuck":
            return (f"Gave up on {what}: the robot did not move {self.cfg.nav_stuck_distance_m:.2f} m in {self.cfg.nav_stuck_s:.0f} s, {left}. "
                    "Next: move_relative backward, then retry with a nearer intermediate point.")
        if reason == "timeout":
            return f"Timed out on the way to {what}, {left}. Next: call again (it continues from here) or pick an intermediate point."
        if reason == "cancelled":
            return f"Navigation to {what} was cancelled, {left}."
        if reason == "tolerance":
            return (f"Nav2 reports arrival but the robot is {left} (tolerance {tol:.1f} m). Next: call goto_pose again or move_relative "
                    "for the last bit.")
        return f"Navigation to {what} failed ({reason}), {left}. Next: retry once or choose another target."

    # ------------------------------------------------------------ skills
    def goto_pose(self, ctx: SkillContext, args: dict) -> SkillResult:
        x, y, tol = float(args["x"]), float(args["y"]), float(args.get("tolerance_m", self.cfg.goal_tolerance_m))
        yaw = math.radians(float(args["yaw_deg"])) if "yaw_deg" in args else None
        grid = self.api.grid()
        moved_note = ""
        if grid is not None and grid.value(x, y) >= 50:
            # the requested point (e.g. a room centre) is inside furniture: use the nearest free spot within snap_m (default 2 m)
            snap = float(args.get("snap_m", 2.0))
            spot = None
            for k in range(1, int(snap / 0.2) + 1):
                ring = [(x + 0.2 * k * math.cos(a), y + 0.2 * k * math.sin(a)) for a in (i * math.pi / 8 for i in range(16))]
                free = [p for p in ring if grid.disc_free(p[0], p[1], 0.3)]
                if free:
                    spot = free[0]
                    break
            if spot is None:
                return SkillResult(False, f"The point ({x:.1f}, {y:.1f}) is inside an obstacle in the map and no free spot is within {snap:.1f} m. "
                                   "Next: choose a free point nearby (for example 0.5-1 m away from the wall or furniture).", {"reason": "occupied"})
            moved_note = f" (the requested point ({x:.1f}, {y:.1f}) was inside an obstacle; moved {math.hypot(spot[0] - x, spot[1] - y):.1f} m to the nearest free spot)"
            x, y = spot
        r = self.nav_to(ctx, x, y, yaw, tol)
        pose = r["pose"]
        data = {"remaining_m": None if math.isnan(r["remaining_m"]) else round(r["remaining_m"], 2), "nav2_status": r["reason"] or "succeeded",
                "reason": r["reason"] or "done"}
        if pose is not None:
            data.update(final_x=round(pose.x, 2), final_y=round(pose.y, 2), final_yaw_deg=round(math.degrees(pose.yaw), 1))
        if r["ok"]:
            return SkillResult(True, f"Arrived at ({x:.1f}, {y:.1f}), {r['remaining_m']:.2f} m from the target point." + moved_note, data)
        return SkillResult(False, self._nav_failure_message(r, f"({x:.1f}, {y:.1f})", tol) + moved_note, data)

    def resolve_object(self, args: dict) -> tuple[ObjectRecord | None, str]:
        """Object from object_id or label (nearest confident, not rejected). Returns (record, error message)."""
        pose = self.api.map_pose()
        if "object_id" in args:
            o = self.api.get_object(int(args["object_id"]))
            if o is None:
                known = ", ".join(f"{x.label} #{x.id}" for x in self.api.objects()[:12]) or "nothing mapped yet"
                return None, (f"Object #{args['object_id']} is not in the semantic map (known: {known}). Next: call search or scan to find it.")
            if o.bearing_only:
                return None, (f"Object #{o.id} ({o.label}) is only known by its bearing, its position is unreliable. "
                              "Next: explore or scan closer, or search again so it gets a position.")
            return o, ""
        label = str(args.get("label", ""))
        rejected = args.get("rejected_ids", [])
        xy = (pose.x, pose.y) if pose else None
        cands = select_candidates(self.api.query_objects(label=label), label, rejected, self.cfg.confident_instance, False, xy)
        if cands:
            return cands[0], ""
        any_label = select_candidates(self.api.objects(), label, (), 0.0, True, xy)
        if any_label and all(o.id in set(rejected) for o in any_label):
            return None, f"Every mapped '{label}' is already rejected (ids {sorted(set(rejected))}). Next: search with the rejected_ids list to find a new one, or explore."
        if any_label:
            return None, (f"'{label}' was seen but is not reliable yet (low confidence or bearing only). Next: scan or explore closer to it, then try again.")
        return None, f"No '{label}' in the semantic map. Next: call search(label='{label}') to look for it."

    def standoff_goal(self, obj: ObjectRecord, standoff: float) -> tuple[Pose2D | None, str]:
        pose = self.api.map_pose()
        if pose is None:
            return None, "No map pose (TF map->base_link) yet: localisation is not ready. Wait a few seconds and retry."
        grid = self.api.grid()
        res = standoff_pose((obj.x, obj.y), (obj.sx, obj.sy), (pose.x, pose.y), standoff, self.cfg.front, grid,
                            self.cfg.clearance_radius_m, self.cfg.allow_unknown_for_standoff)
        if res is None:
            return None, (f"No free spot {standoff:.1f} m from {obj.label} #{obj.id} in the known map (it may be in a corner or unexplored). "
                          "Next: try a smaller standoff_m, goto_pose to a nearby free point, or explore around it.")
        return res[0], ""

    def goto_object(self, ctx: SkillContext, args: dict) -> SkillResult:
        obj, err = self.resolve_object(args)
        if obj is None:
            return SkillResult(False, err, {"reason": "unknown_object"})
        standoff = float(args.get("standoff_m", 1.0))
        return self._goto_resolved(ctx, obj, standoff)

    def _goto_resolved(self, ctx: SkillContext, obj: ObjectRecord, standoff: float) -> SkillResult:
        goal, err = self.standoff_goal(obj, standoff)
        if goal is None:
            return SkillResult(False, err, {"reason": "no_standoff", "object_id": obj.id})
        r = self.nav_to(ctx, goal.x, goal.y, goal.yaw, self.cfg.goal_tolerance_m)
        what = f"{obj.label} #{obj.id}"
        data = {"object_id": obj.id, "label": obj.label, "reason": r["reason"] or "done"}
        pose = r["pose"] or self.api.map_pose()
        if pose is not None:
            d = obj.to_dict((pose.x, pose.y), pose.yaw)
            data.update(distance_m=d["distance_m"], bearing_deg=d["bearing_deg"], robot_x=round(pose.x, 2), robot_y=round(pose.y, 2),
                        robot_yaw_deg=round(math.degrees(pose.yaw), 1))
        if not r["ok"]:
            return SkillResult(False, self._nav_failure_message(r, what, self.cfg.goal_tolerance_m), data)
        # nav2 accepts a loose yaw; finish facing the object
        if pose is not None:
            delta = bearing_to(pose, obj.x, obj.y)
            if abs(delta) > math.radians(10):
                self.motion.rotate(ctx, delta)
                pose = self.api.map_pose() or pose
                d = obj.to_dict((pose.x, pose.y), pose.yaw)
                data.update(distance_m=d["distance_m"], bearing_deg=d["bearing_deg"])
        dist = data.get("distance_m")
        msg = f"Reached {what}: {dist:.1f} m from its centre" if dist is not None else f"Reached {what}"
        return SkillResult(True, msg + f", facing it (standoff {standoff:.1f} m). Next: verify to confirm it is the right object, or approach for a closer look.", data)

    def approach(self, ctx: SkillContext, args: dict) -> SkillResult:
        cfg, api = self.cfg, self.api
        obj, err = self.resolve_object(args)
        if obj is None:
            return SkillResult(False, err, {"reason": "unknown_object"})
        req = float(args.get("standoff_m", 0.8))
        standoff = max(req, cfg.min_standoff_m)
        note = f" (standoff raised from {req:.2f} to the safety minimum {standoff:.2f} m)" if standoff > req + 1e-6 else ""
        pose = api.map_pose()
        if pose is None:
            return SkillResult(False, "No map pose (TF map->base_link) yet: localisation is not ready. Wait a few seconds and retry.", {"reason": "no_pose"})
        what = f"{obj.label} #{obj.id}"
        if math.hypot(obj.x - pose.x, obj.y - pose.y) > cfg.approach_far_range_m:
            far = self._goto_resolved(ctx, obj, standoff + 0.8)
            if not far.success:
                return SkillResult(False, "Too far for a direct approach and the planner could not bring the robot close first. " + far.message, far.data)
        used_lidar, calm, err_d, brg = False, 0, math.nan, 0.0
        period, reason = 0.1, ""
        tol_b = math.radians(cfg.approach_bearing_tol_deg)
        while True:
            reason = ctx.stop_reason()
            if reason:
                break
            pose = api.map_pose()
            if pose is None:
                reason = "no_pose"
                break
            cur = api.get_object(obj.id) or obj
            brg = bearing_to(pose, cur.x, cur.y)
            dist_c = math.hypot(cur.x - pose.x, cur.y - pose.y)
            d_nose, used_lidar = self._nose_distance(cur, brg, dist_c)
            err_d = d_nose - standoff
            if abs(err_d) <= cfg.approach_tol_m and abs(brg) <= tol_b:
                calm += 1
                if calm >= 3:
                    break
            else:
                calm = 0
            vx = wz = 0.0
            if abs(brg) > math.radians(25):
                wz = float(np.clip(1.5 * brg, -cfg.turn_speed, cfg.turn_speed))
            else:
                wz = float(np.clip(1.5 * brg, -0.4, 0.4))
                if err_d > cfg.approach_tol_m:
                    vx = max(0.05, min(cfg.approach_speed, math.sqrt(2.0 * 0.4 * err_d)))
                elif err_d < -cfg.approach_tol_m:
                    vx = -min(0.1, max(0.04, -err_d))
            if api.blocked_for() > 3.0:
                reason = "blocked"
                break
            api.send_cmd(vx, 0.0, wz)
            ctx.feedback(f"approach: {d_nose:.2f} m to surface", 0.0)
            ctx.sleep(period)
        api.halt()
        data = {"object_id": obj.id, "range_m": None if math.isnan(err_d) else round(err_d + standoff, 2), "standoff_m": standoff,
                "bearing_error_deg": round(math.degrees(brg), 1), "used_lidar": used_lidar, "reason": reason or "done"}
        if not reason:
            return SkillResult(True, f"Approached {what}: nose is {err_d + standoff:.2f} m from its surface (target {standoff:.2f} m{note}), "
                               f"facing it ({abs(math.degrees(brg)):.0f} deg off). "
                               + ("" if used_lidar else "The range comes from the map position (no lidar return on the object), so it may be off by ~0.3 m. ")
                               + "Next: verify.", data)
        why = {"blocked": "the safety guard stopped the robot before the standoff: something is in the way. Next: goto_object to plan around it",
               "cancelled": "the skill was cancelled", "timeout": "the time limit ran out",
               "no_pose": "the map pose was lost"}.get(reason, reason)
        rng = f"{err_d + standoff:.2f} m from its surface" if not math.isnan(err_d) else "range unknown"
        return SkillResult(False, f"Approach of {what} stopped at {rng}: {why}.", data)

    def _nose_distance(self, obj: ObjectRecord, bearing: float, dist_center: float) -> tuple[float, bool]:
        """Distance (m) from the robot's nose to the object's surface: lidar returns inside the object's angular window, else geometry."""
        cfg = self.cfg
        half = 0.5 * min(max(obj.sx, obj.sy, 0.3), 1.5)
        pts = self.api.obstacle_points()
        if pts is not None and pts.size:
            m = cfg.self_margin
            body = (pts[:, 0] > -cfg.rear - m) & (pts[:, 0] < cfg.front + m) & (np.abs(pts[:, 1]) < cfg.half_width + m)
            p = pts[~body]
            if p.size:
                rel = wrap_pi_array(np.arctan2(p[:, 1], p[:, 0]) - bearing)
                window = max(math.atan2(half + 0.15, max(dist_center, 0.3)), math.radians(8.0))
                along = np.hypot(p[:, 0], p[:, 1]) * np.cos(rel)
                sel = (np.abs(rel) <= window) & (along > 0.0) & (along < dist_center + 1.0)
                if sel.any():
                    return float(along[sel].min()) - cfg.front, True
        return dist_center - half - cfg.front, False


def wrap_pi_array(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi
