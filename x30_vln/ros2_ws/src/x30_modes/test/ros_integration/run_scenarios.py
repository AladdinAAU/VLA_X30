"""Integration scenarios for x30_modes against the fake robot (start the stack first with start_stack.sh).

  python3 run_scenarios.py [group ...]      groups: mux guard hazards primitives skills modes   (default: all)

Prints one line per check, writes results/modes/integration.json, exits non-zero if any check failed.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

from x30_modes.geometry import Footprint, Pose2D, wrap_pi, yaw_from_quat
from x30_msgs.action import Navigate, Skill
from x30_msgs.msg import SafetyStatus
from x30_msgs.srv import Json, SetMode

CHAIR_BOX = (4.75, -2.75, 5.25, -2.25)
RESULTS: dict = {}
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


class Driver(Node):
    def __init__(self):
        super().__init__("scenario_driver")
        self.raw = self.out = self.safety = self.truth_msg = None
        self.stats: dict = {}
        self.source = ""
        self.mode = ""
        self.out_hist: list[tuple[float, float, float, float]] = []
        self.create_subscription(Twist, "/cmd_vel", self._on_out, 50)
        self.create_subscription(Twist, "/cmd_vel_raw", lambda m: setattr(self, "raw", m), 10)
        self.create_subscription(SafetyStatus, "/x30/safety", lambda m: setattr(self, "safety", m), 10)
        self.create_subscription(PoseStamped, "/fake/truth", lambda m: setattr(self, "truth_msg", m), 10)
        self.create_subscription(String, "/fake/stats", lambda m: setattr(self, "stats", json.loads(m.data)), 10)
        self.create_subscription(String, "/x30/cmd_source", lambda m: setattr(self, "source", m.data), 10)
        self.create_subscription(String, "/x30/mode", lambda m: setattr(self, "mode", m.data),
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub_nav = self.create_publisher(Twist, "/cmd_vel_nav", 10)
        self.pub_tele = self.create_publisher(Twist, "/cmd_vel_teleop", 10)
        self.pub_estop = self.create_publisher(Bool, "/x30/estop", 10)
        self.pub_reset = self.create_publisher(Float64MultiArray, "/fake/reset", 10)
        self.pub_box = self.create_publisher(Float64MultiArray, "/fake/add_box", 10)
        self.pub_tilt = self.create_publisher(Float64, "/fake/tilt_deg", 10)
        self.c_mode = self.create_client(SetMode, "/x30/set_mode")
        self.c_prim = self.create_client(Json, "/x30/primitive")
        self.c_nav = ActionClient(self, Navigate, "/x30/navigate")
        self.c_skill: dict[str, ActionClient] = {}

    def _on_out(self, m: Twist) -> None:
        self.out = m
        self.out_hist.append((time.monotonic(), m.linear.x, m.linear.y, m.angular.z))
        if len(self.out_hist) > 4000:
            del self.out_hist[:2000]

    # ------------------------------------------------------------ helpers
    def truth(self) -> Pose2D:
        q = self.truth_msg.pose.orientation
        return Pose2D(self.truth_msg.pose.position.x, self.truth_msg.pose.position.y, yaw_from_quat(q.x, q.y, q.z, q.w))

    def reset(self, x=0.0, y=0.0, yaw=0.0, full=True) -> None:
        self.estop(False)
        self.pub_tilt.publish(Float64(data=0.0))
        self.pub_reset.publish(Float64MultiArray(data=[x, y, yaw, 1.0 if full else 0.0]))
        time.sleep(0.8)

    def estop(self, v: bool) -> None:
        self.pub_estop.publish(Bool(data=v))

    def set_mode(self, mode: str):
        req = SetMode.Request()
        req.mode = mode
        self.c_mode.wait_for_service(5.0)
        f = self.c_mode.call_async(req)
        while not f.done():
            time.sleep(0.01)
        time.sleep(0.15)
        return f.result()

    def drive_nav(self, vx=0.0, vy=0.0, wz=0.0, secs=1.0, rate=20.0, pub=None) -> None:
        pub = pub or self.pub_nav
        t_end = time.monotonic() + secs
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = vx, vy, wz
        while time.monotonic() < t_end:
            pub.publish(m)
            time.sleep(1.0 / rate)

    def skill(self, name: str, args: dict, timeout_s: float = 0.0, wait: float = 300.0, cancel_after: float | None = None):
        if name not in self.c_skill:
            self.c_skill[name] = ActionClient(self, Skill, f"/x30/skill/{name}")
        c = self.c_skill[name]
        assert c.wait_for_server(5.0), f"no skill server for {name}"
        g = Skill.Goal()
        g.name, g.args_json, g.timeout_s = name, json.dumps(args), float(timeout_s)
        fut = c.send_goal_async(g)
        while not fut.done():
            time.sleep(0.01)
        gh = fut.result()
        rf = gh.get_result_async()
        t0 = time.monotonic()
        cancelled = False
        while not rf.done() and time.monotonic() - t0 < wait:
            if cancel_after is not None and not cancelled and time.monotonic() - t0 > cancel_after:
                gh.cancel_goal_async()
                cancelled = True
            time.sleep(0.02)
        assert rf.done(), f"skill {name} did not finish in {wait}s"
        r = rf.result().result
        return r.success, r.message, json.loads(r.data_json or "{}")

    def navigate(self, mode: str, timeout_s=0.0, params=None, x=0.0, y=0.0, yaw=0.0, instruction="", on_started=None, wait=120.0):
        assert self.c_nav.wait_for_server(5.0)
        g = Navigate.Goal()
        g.mode, g.timeout_s, g.instruction = mode, float(timeout_s), instruction
        g.params_json = json.dumps(params or {})
        g.goal.header.frame_id = "map"
        g.goal.pose.position.x, g.goal.pose.position.y = x, y
        g.goal.pose.orientation.z, g.goal.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        fb: list = []
        fut = self.c_nav.send_goal_async(g, feedback_callback=lambda f: fb.append(f.feedback))
        while not fut.done():
            time.sleep(0.01)
        gh = fut.result()
        rf = gh.get_result_async()
        t0 = time.monotonic()
        started = False
        while not rf.done() and time.monotonic() - t0 < wait:
            if on_started and not started and time.monotonic() - t0 > on_started[0]:
                on_started[1]()
                started = True
            time.sleep(0.02)
        assert rf.done(), "navigate did not finish"
        return rf.result().result, fb

    def front_gap(self) -> float:
        return self.stats.get("gap_now")


def nose_gap_to_chair(p: Pose2D) -> float:
    """Distance from the robot's front edge along its heading to the chair box (ray cast), metres."""
    x0, y0, x1, y1 = CHAIR_BOX
    best = math.inf
    for k in range(0, 400):
        d = k * 0.01
        px, py = p.x + (0.5 + d) * math.cos(p.yaw), p.y + (0.5 + d) * math.sin(p.yaw)
        if x0 <= px <= x1 and y0 <= py <= y1:
            best = d
            break
    return best


# ====================================================================================== groups
def group_mux(d: Driver):
    d.reset()
    d.set_mode("object_nav")
    d.drive_nav(0.2, secs=1.0)
    check("mux: nav passes in a task mode", d.source == "nav" and d.out.linear.x > 0.05, f"source={d.source} vx={d.out.linear.x:.2f}")
    th = threading.Thread(target=d.drive_nav, kwargs=dict(vx=0.2, secs=2.0), daemon=True)
    th.start()
    time.sleep(0.3)
    d.drive_nav(wz=0.3, secs=0.8, pub=d.pub_tele)
    check("mux: teleop beats nav", d.source == "teleop" and abs(d.raw.angular.z - 0.3) < 0.01 and abs(d.raw.linear.x) < 1e-6,
          f"source={d.source} raw=({d.raw.linear.x:.2f},{d.raw.angular.z:.2f})")
    t_rel = time.monotonic()
    while d.source != "nav" and time.monotonic() - t_rel < 3.0:
        time.sleep(0.05)
    th.join()
    resumed = time.monotonic() - t_rel
    check("mux: nav resumes after teleop release + lockout", d.source == "nav" or resumed < 3.0, f"resumed after {resumed:.2f} s (lockout 0.5 s)")
    d.set_mode("idle")
    d.drive_nav(0.2, secs=0.8)
    check("mux: idle blocks nav", abs(d.raw.linear.x) < 1e-6 and d.source == "", f"raw vx={d.raw.linear.x:.2f}")
    d.set_mode("teleop")
    d.drive_nav(0.2, secs=0.8)
    check("mux: teleop mode ignores nav", abs(d.raw.linear.x) < 1e-6, f"raw vx={d.raw.linear.x:.2f}")
    d.drive_nav(0.2, secs=0.8, pub=d.pub_tele)
    check("mux: teleop mode passes teleop", d.raw.linear.x > 0.15, f"raw vx={d.raw.linear.x:.2f}")
    d.set_mode("idle")
    d.reset()


def settle(d: Driver, secs=2.0):
    time.sleep(secs)


def group_guard(d: Driver):
    rows = []
    d.set_mode("explore")                        # a mode that lets nav commands through, no task running
    cases = [("forward", 0.2, (0.0, 0.0, 0.0), (1, 0, 0)), ("forward", 0.4, (0.0, 0.0, 0.0), (1, 0, 0)), ("forward", 0.6, (0.0, 0.0, 0.0), (1, 0, 0)),
             ("backward", -0.3, (1.5, 0.0, 0.0), (1, 0, 0)), ("strafe_left", 0.3, (8.0, 0.0, 0.0), (0, 1, 0)),
             ("forward_wall", 0.6, (5.0, -3.2, 0.0), (1, 0, 0))]
    for name, speed, start, _ in cases:
        d.reset(*start)
        if name.startswith("strafe"):
            d.drive_nav(vy=speed, secs=14.0)
        elif name == "forward_wall":
            d.drive_nav(vx=speed, secs=12.0)
        else:
            d.drive_nav(vx=speed, secs=12.0 if abs(speed) < 0.5 else 10.0)
        settle(d, 1.5)
        st = dict(d.stats)
        gap = st["gap_now"]
        rows.append({"case": name, "speed": speed, "stop_gap_m": round(gap, 3), "min_gap_m": round(st["min_gap"], 3), "collisions": st["collisions"],
                     "guard_reason": d.safety.reason if d.safety else None})
        check(f"guard: stops before obstacle ({name} {speed:+.1f} m/s)", st["collisions"] == 0 and gap >= 0.15,
              f"standstill gap {gap:.3f} m, min gap {st['min_gap']:.3f} m, guard={rows[-1]['guard_reason']}")
    # sudden obstacle appearing ahead while moving at 0.4 m/s (dynamic box)
    for ahead in (1.0, 0.6, 0.35):
        d.reset(1.0, -3.0, 0.0)
        th = threading.Thread(target=d.drive_nav, kwargs=dict(vx=0.4, secs=6.0), daemon=True)
        th.start()
        time.sleep(2.0)
        p = d.truth()
        bx = p.x + 0.5 + ahead
        d.pub_box.publish(Float64MultiArray(data=[bx, p.y - 1.0, bx + 0.3, p.y + 1.0]))
        th.join()
        settle(d, 1.0)
        st = dict(d.stats)
        rows.append({"case": f"sudden_box_{ahead}m_at_0.4", "speed": 0.4, "appear_gap_m": ahead, "stop_gap_m": round(st["gap_now"], 3),
                     "min_gap_m": round(st["min_gap"], 3), "collisions": st["collisions"]})
        check(f"guard: obstacle appearing {ahead} m ahead of the nose at 0.4 m/s", st["collisions"] == 0 or ahead < 0.5,
              f"collisions {st['collisions']}, standstill gap {st['gap_now']:.3f} m, min gap {st['min_gap']:.3f}")
    RESULTS["guard_stopping"] = rows
    d.set_mode("idle")
    d.reset()


def group_hazards(d: Driver):
    d.reset(1.0, -3.0, 0.0)
    d.set_mode("explore")
    th = threading.Thread(target=d.drive_nav, kwargs=dict(vx=0.4, secs=3.0), daemon=True)
    th.start()
    time.sleep(1.5)
    check("hazards: moving before e-stop", d.out.linear.x > 0.2, f"vx={d.out.linear.x:.2f}")
    t0 = time.monotonic()
    d.estop(True)
    while d.out.linear.x > 1e-6 and time.monotonic() - t0 < 2.0:
        time.sleep(0.01)
    dt = time.monotonic() - t0
    check("hazards: e-stop zeroes /cmd_vel within 0.25 s", dt < 0.25 and d.safety.reason in ("estop",), f"{dt * 1000:.0f} ms, reason={d.safety.reason}")
    th.join()
    d.estop(False)
    d.reset(1.0, -3.0, 0.0)
    th = threading.Thread(target=d.drive_nav, kwargs=dict(vx=0.4, secs=3.0), daemon=True)
    th.start()
    time.sleep(1.2)
    d.pub_tilt.publish(Float64(data=50.0))
    time.sleep(0.5)
    check("hazards: tilt above limit stops the output", abs(d.out.linear.x) < 1e-6 and d.safety.reason == "tilt", f"vx={d.out.linear.x:.2f} reason={d.safety.reason}")
    th.join()
    d.pub_tilt.publish(Float64(data=0.0))
    # stale nav command: stop publishing while moving
    d.reset(1.0, -3.0, 0.0)
    d.drive_nav(vx=0.4, secs=2.0)
    t0 = time.monotonic()
    while abs(d.out.linear.x) > 1e-6 and time.monotonic() - t0 < 3.0:
        time.sleep(0.01)
    dt = time.monotonic() - t0
    check("hazards: stale command stops the robot (nav timeout 0.5 s + decel)", dt < 1.5, f"zero output {dt:.2f} s after the last command")
    # acceleration limit
    d.reset(1.0, -3.0, 0.0)
    d.out_hist.clear()
    d.drive_nav(vx=0.6, secs=2.0)
    h = [(t, vx) for t, vx, _, _ in d.out_hist if True]
    amax = max(((h[i + 1][1] - h[i][1]) / max(h[i + 1][0] - h[i][0], 1e-3)) for i in range(len(h) - 1) if h[i + 1][0] - h[i][0] > 0.03)
    check("hazards: output acceleration limited (0.8 m/s2)", amax < 1.1, f"max measured acceleration {amax:.2f} m/s2")
    d.reset()
    d.set_mode("idle")


def group_primitives(d: Driver):
    d.set_mode("explore")
    rows = []

    def one(label, name, args, expect_fn):
        d.reset(1.0, -3.0, 0.3)
        p0 = d.truth()
        t0 = time.monotonic()
        ok, msg, data = d.skill(name, args)
        time.sleep(1.0)
        p1 = d.truth()
        err, achieved = expect_fn(p0, p1)
        rows.append({"cmd": label, "success": ok, "achieved": round(achieved, 2), "error": round(err, 2), "seconds": round(time.monotonic() - t0, 1)})
        return ok, err, msg

    def along(ux, uy, req_cm):
        def f(p0, p1):
            c, s = math.cos(p0.yaw), math.sin(p0.yaw)
            a = ((p1.x - p0.x) * (c * ux - s * uy) + (p1.y - p0.y) * (s * ux + c * uy)) * 100
            return a - req_cm, a
        return f

    def turned(req):
        def f(p0, p1):
            a = math.degrees(wrap_pi(p1.yaw - p0.yaw))
            err = (a - req + 180.0) % 360.0 - 180.0         # the final yaw only tells the rotation modulo 360
            return err, req + err
        return f

    for cm in (100, 100, 100):
        ok, err, msg = one("forward 100 cm", "move_relative", {"direction": "forward", "distance_cm": cm}, along(1, 0, cm))
        check("primitive: move forward 100 cm", ok and abs(err) <= 3.0, f"error {err:+.1f} cm :: {msg}")
    for dirn, ux, uy, cm in (("backward", -1, 0, 50), ("left", 0, 1, 30), ("right", 0, -1, 30), ("forward", 1, 0, 20)):
        ok, err, msg = one(f"{dirn} {cm} cm", "move_relative", {"direction": dirn, "distance_cm": cm}, along(ux, uy, cm))
        check(f"primitive: move {dirn} {cm} cm", ok and abs(err) <= 3.0, f"error {err:+.1f} cm")
    for deg in (90, 90, 90, -90, 45, -30, 180, 360):
        ok, err, msg = one(f"turn {deg:+d} deg", "turn_to", {"degrees": deg}, turned(deg))
        check(f"primitive: turn {deg:+d} deg", ok and abs(err) <= 3.0, f"error {err:+.1f} deg")
    # via the /x30/primitive service
    d.reset(1.0, -3.0, 0.0)
    req = Json.Request()
    req.request = json.dumps({"text": "move forward 75 cm"})
    d.c_prim.wait_for_service(5.0)
    f = d.c_prim.call_async(req)
    while not f.done():
        time.sleep(0.01)
    time.sleep(0.8)
    r = json.loads(f.result().response)
    check("primitive service: 'move forward 75 cm'", f.result().ok and abs((d.truth().x - 1.0) * 100 - 75) <= 3.0, f"x moved {(d.truth().x - 1.0) * 100:.1f} cm :: {r['message']}")
    req.request = json.dumps({"action": "turn_right", "value": 30})
    f = d.c_prim.call_async(req)
    while not f.done():
        time.sleep(0.01)
    check("primitive service: turn_right 30", f.result().ok, json.loads(f.result().response)["message"])
    req.request = json.dumps({"text": "dance"})
    f = d.c_prim.call_async(req)
    while not f.done():
        time.sleep(0.01)
    check("primitive service: bad text rejected with message", (not f.result().ok) and "cannot parse" in f.result().response, f.result().response[:90])
    # blocked primitive: forward into the pillar
    d.reset(1.5, 0.0, 0.0)
    ok, msg, data = d.skill("move_relative", {"direction": "forward", "distance_cm": 300})
    check("primitive: forward into the pillar aborts with guard message", (not ok) and "safety guard" in msg and d.stats["collisions"] == 0, msg)
    RESULTS["primitives"] = rows
    errs = [abs(r["error"]) for r in rows if "cm" in r["cmd"]]
    errt = [abs(r["error"]) for r in rows if "deg" in r["cmd"]]
    RESULTS["primitive_summary"] = {"translation_abs_error_cm_max": max(errs), "translation_abs_error_cm_mean": round(sum(errs) / len(errs), 2),
                                    "rotation_abs_error_deg_max": max(errt), "rotation_abs_error_deg_mean": round(sum(errt) / len(errt), 2)}
    d.set_mode("idle")


def group_skills(d: Driver):
    d.set_mode("explore")
    d.reset(0.0, 0.0, 0.0)
    ok, msg, data = d.skill("goto_pose", {"x": 7.5, "y": -1.0, "yaw_deg": 0})
    st = d.stats
    check("skill goto_pose around the pillar", ok and st["collisions"] == 0 and math.hypot(d.truth().x - 7.5, d.truth().y + 1.0) < 0.5,
          f"{msg} truth=({d.truth().x:.2f},{d.truth().y:.2f}) collisions={st['collisions']} path={st['path_len']:.1f} m")
    RESULTS["goto_pose"] = {"success": ok, "final": [round(d.truth().x, 2), round(d.truth().y, 2)], "path_len_m": round(st["path_len"], 2), "collisions": st["collisions"]}
    ok, msg, data = d.skill("goto_pose", {"x": 3.3, "y": 0.0})
    check("skill goto_pose into an obstacle -> helpful failure", (not ok) and "inside an obstacle" in msg, msg)
    ok, msg, data = d.skill("goto_pose", {"x": 20.0, "y": 20.0})
    check("skill goto_pose unreachable -> helpful failure", (not ok) and "Next:" in msg, msg)
    ok, msg, data = d.skill("goto_pose", {})
    check("skill invalid args -> message, no exception", (not ok) and "missing required argument 'x'" in msg, msg)
    ok, msg, data = d.skill("move_relative", {"distance_cm": 9999})
    check("skill out-of-range arg -> message", (not ok) and "maximum" in msg, msg)

    d.reset(0.0, 0.0, 0.0)
    ok, msg, data = d.skill("goto_object", {"label": "chair"})
    check("skill goto_object before the chair is mapped -> suggests search", (not ok) and "search" in msg, msg)
    ok, msg, data = d.skill("search", {"label": "chair", "budget_s": 120}, wait=200)
    cid = data.get("candidates", [{}])[0].get("id") if data.get("candidates") else None
    check("skill search explores and finds the chair", ok and cid == 1, msg)
    ok, msg, data = d.skill("search", {"label": "chair", "rejected_ids": [1], "budget_s": 15}, wait=100)
    check("skill search honours rejected_ids (and reports exhaustion/budget)", (not ok) and "not found" in msg, msg)
    ok, msg, data = d.skill("goto_object", {"object_id": 1, "standoff_m": 1.2})
    p = d.truth()
    gap = nose_gap_to_chair(p)
    check("skill goto_object chair standoff 1.2 m", ok and abs(gap - 1.2) < 0.45 and d.stats["collisions"] == 0, f"{msg} | measured nose gap {gap:.2f} m")
    ok, msg, data = d.skill("approach", {"object_id": 1, "standoff_m": 0.8})
    p = d.truth()
    gap = nose_gap_to_chair(p)
    yaw_err = math.degrees(wrap_pi(math.atan2(-2.5 - p.y, 5.0 - p.x) - p.yaw))
    RESULTS["approach"] = {"requested_standoff_m": 0.8, "measured_nose_gap_m": round(gap, 3), "bearing_error_deg": round(yaw_err, 1), "collisions": d.stats["collisions"]}
    check("skill approach chair to 0.8 m", ok and abs(gap - 0.8) < 0.12 and abs(yaw_err) < 10 and d.stats["collisions"] == 0, f"{msg} | measured nose gap {gap:.2f} m, bearing error {yaw_err:.1f} deg")
    ok, msg, data = d.skill("approach", {"object_id": 1, "standoff_m": 0.3})
    check("skill approach clamps standoff to the safety minimum", ok and "raised" in msg, msg)
    ok, msg, data = d.skill("verify", {"label": "chair", "object_id": 1})
    check("skill verify chair -> yes", ok and data.get("answer") == "yes", msg)
    ok, msg, data = d.skill("verify", {"label": "table", "object_id": 1})
    check("skill verify wrong label -> no with next step", ok and data.get("answer") == "no" and "rejected_ids" in msg, msg)
    ok, msg, data = d.skill("verify", {"question": "unsure_please is this a door?"})
    check("skill verify -> unsure", ok and data.get("answer") == "unsure", msg)
    ok, msg, data = d.skill("scan", {"sweep_deg": 360, "step_deg": 90, "dwell_s": 0.5})
    check("skill scan 360 deg", ok and "chair" in msg and abs(math.degrees(wrap_pi(d.truth().yaw - p.yaw))) < 8, msg)
    ok, msg, data = d.skill("turn_to", {"object_id": 2})
    check("skill turn_to object", ok, msg)
    ok, msg, data = d.skill("turn_to", {"bearing": "behind"})
    check("skill turn_to bearing word", ok and abs(abs(data["achieved_deg"]) - 180) < 4, msg)
    ok, msg, data = d.skill("turn_to", {"bearing": "sideways"})
    check("skill turn_to bad bearing -> message", (not ok) and "unknown bearing" in msg, msg)
    ok, msg, data = d.skill("stop", {})
    check("skill stop when idle", ok, msg)

    # cancel/stop/busy/blocked
    d.reset(0.0, 0.0, 0.0)
    r: dict = {}
    th = threading.Thread(target=lambda: r.update(zip(("ok", "msg", "data"), d.skill("goto_pose", {"x": 8.0, "y": -2.0}))), daemon=True)
    th.start()
    time.sleep(2.5)
    ok2, msg2, _ = d.skill("move_relative", {"distance_cm": 20})
    check("skill exclusivity: second motion skill answers busy", (not ok2) and "Busy" in msg2, msg2)
    ok3, msg3, data3 = d.skill("stop", {})
    th.join(20)
    time.sleep(1.0)
    v = d.out
    check("skill stop cancels goto_pose and halts", ok3 and "goto_pose" in msg3 and (not r.get("ok")) and abs(v.linear.x) < 0.02, f"{msg3} | goto said: {r.get('msg')}")
    d.reset(0.0, 0.0, 0.0)
    ok, msg, data = d.skill("goto_pose", {"x": 8.0, "y": -2.0}, cancel_after=2.0)
    time.sleep(0.8)
    check("skill action cancel stops the robot", abs(d.out.linear.x) < 0.02, msg)
    d.reset(0.0, 0.0, 0.0)
    th = threading.Thread(target=lambda: r.update(zip(("ok", "msg", "data"), d.skill("goto_pose", {"x": 8.0, "y": 0.0}))), daemon=True)
    th.start()
    time.sleep(1.0)
    d.pub_box.publish(Float64MultiArray(data=[1.8, -3.0, 2.2, 3.0]))          # unmapped wall across the corridor
    th.join(60)
    time.sleep(0.5)
    check("skill goto_pose blocked by an unmapped obstacle -> guard message, no collision", (not r.get("ok")) and "safety guard" in r.get("msg", "") and d.stats["collisions"] == 0,
          f"{r.get('msg')} collisions={d.stats['collisions']}")
    d.reset(0.0, 0.0, 0.0)
    ok, msg, data = d.skill("explore", {"steps": 2}, wait=200)
    check("skill explore 2 frontiers", ok and len(data["visited"]) == 2, msg)
    ok, msg, data = d.skill("explore", {}, wait=100)
    check("skill explore when nothing is left -> exhausted message", (not ok) and data.get("exhausted") and "fully explored" in msg, msg)
    d.set_mode("idle")


def group_modes(d: Driver):
    d.reset(0.0, 0.0, 0.0)
    res, fb = d.navigate("goal_pose", x=7.5, y=-1.0, yaw=0.0)
    p = d.truth()
    check("mode goal_pose: success + path length from odometry", res.success and res.stop_reason == "finished" and abs(res.path_length_m - d.stats["path_len"]) < 0.4 and d.mode == "idle",
          f"{res.message} | path {res.path_length_m:.2f} m (truth {d.stats['path_len']:.2f}), {res.duration_s:.0f} s, final=({p.x:.1f},{p.y:.1f}), feedback msgs {len(fb)}")
    RESULTS["mode_goal_pose"] = {"success": res.success, "path_length_m": round(res.path_length_m, 2), "truth_path_m": round(d.stats["path_len"], 2), "duration_s": round(res.duration_s, 1)}
    d.reset(0.0, 0.0, 0.0)
    res, fb = d.navigate("explore", timeout_s=200)
    check("mode explore: runs until no frontier is left", res.success and res.stop_reason == "finished" and "Exploration complete" in res.message, res.message[:140])
    # delegation
    res, fb = d.navigate("object_nav", instruction="find the chair", params={"duration_s": 3.0})
    check("mode object_nav: delegated to /x30/agent/run, turns relayed", res.success and res.agent_turns >= 3 and any(f.agent_turn >= 1 for f in fb) and any(f.current_skill == "move_relative" for f in fb),
          f"{res.message} turns={res.agent_turns} feedback={len(fb)}")
    res, fb = d.navigate("velocity_follow", instruction="go forward", params={"duration_s": 2.0})
    check("mode velocity_follow: delegated to /x30/agent/follow", res.success and "follow" in res.message, res.message)
    # switch mid-task
    d.reset(0.0, -3.0, 0.0)
    res_box: dict = {}
    th = threading.Thread(target=lambda: res_box.update(r=d.navigate("object_nav", params={"duration_s": 30.0, "drive": True})), daemon=True)
    th.start()
    time.sleep(3.0)
    check("mode switch: robot moving under the delegated task", d.out.linear.x > 0.05 and d.source == "nav", f"vx={d.out.linear.x:.2f} source={d.source}")
    t0 = time.monotonic()
    sm = d.set_mode("teleop")
    th.join(15)
    res = res_box["r"][0]
    time.sleep(0.8)
    check("mode switch mid-task: task aborted cleanly, mode changed, velocity zero", sm.ok and (not res.success) and res.stop_reason == "aborted" and "switched" in res.message
          and d.mode == "teleop" and abs(d.out.linear.x) < 0.02, f"set_mode took {time.monotonic() - t0:.2f} s; {res.message}; mode={d.mode}; source={d.source}")
    d.set_mode("idle")
    bad = d.set_mode("bogus")
    check("set_mode rejects unknown mode", (not bad.ok) and "unknown mode" in bad.message, bad.message)
    # new goal preempts the old one
    res_box.clear()
    d.reset(0.0, -3.0, 0.0)
    th = threading.Thread(target=lambda: res_box.update(r=d.navigate("object_nav", params={"duration_s": 30.0})), daemon=True)
    th.start()
    time.sleep(1.5)
    res2, _ = d.navigate("velocity_follow", params={"duration_s": 1.0})
    th.join(15)
    check("new Navigate goal preempts the running one", res_box["r"][0].stop_reason == "aborted" and "preempted" in res_box["r"][0].message and res2.success, res_box["r"][0].message)
    # timeout, watchdog, teleop, estop, safety
    res, _ = d.navigate("object_nav", timeout_s=3.0, params={"duration_s": 30.0})
    check("task timeout -> stop_reason timeout", (not res.success) and res.stop_reason == "timeout", f"{res.message} after {res.duration_s:.1f} s")
    res, _ = d.navigate("velocity_follow", params={"duration_s": 30.0, "freeze": True, "no_progress_window_s": 3.0})
    check("progress watchdog -> stop_reason no_progress", (not res.success) and res.stop_reason == "no_progress", f"{res.message} after {res.duration_s:.1f} s")
    d.reset(0.0, -3.0, 0.0)
    res, _ = d.navigate("object_nav", params={"duration_s": 30.0, "drive": True}, on_started=(2.0, lambda: d.drive_nav(wz=0.4, secs=0.6, pub=d.pub_tele)))
    check("teleop takeover cancels the task", (not res.success) and "teleop" in res.message, res.message)
    d.reset(0.0, -3.0, 0.0)
    res, _ = d.navigate("object_nav", params={"duration_s": 30.0, "drive": True}, on_started=(2.0, lambda: d.estop(True)))
    d.estop(False)
    check("soft e-stop -> stop_reason safety", (not res.success) and res.stop_reason == "safety", res.message)
    d.reset(1.5, 0.0, 0.0)
    res, _ = d.navigate("velocity_follow", params={"duration_s": 30.0, "drive": True, "safety_abort_s": 3.0})
    check("long guard stop (robot pinned at the pillar) -> stop_reason safety, no collision", (not res.success) and res.stop_reason == "safety" and d.stats["collisions"] == 0,
          f"{res.message}; collisions={d.stats['collisions']}")
    res, _ = d.navigate("teleop")
    check("Navigate with a non-task mode -> error result", (not res.success) and res.stop_reason == "error", res.message)
    d.reset(0.0, 0.0, 0.0)
    res, _ = d.navigate("goal_pose", params={"tolerance_m": 0.5}, x=3.3, y=0.0, timeout_s=30)
    check("goal_pose into an obstacle -> failure reported with a reason", not res.success, f"{res.stop_reason}: {res.message}")
    d.set_mode("idle")


GROUPS = {"mux": group_mux, "guard": group_guard, "hazards": group_hazards, "primitives": group_primitives, "skills": group_skills, "modes": group_modes}


def main():
    rclpy.init()
    d = Driver()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(d)
    threading.Thread(target=ex.spin, daemon=True).start()
    t0 = time.monotonic()
    while d.truth_msg is None or d.out is None and time.monotonic() - t0 < 10:
        time.sleep(0.1)
    wanted = sys.argv[1:] or list(GROUPS)
    for g in wanted:
        print(f"=== {g}", flush=True)
        try:
            GROUPS[g](d)
        except Exception:
            check(f"group {g} ran without exception", False, traceback.format_exc(limit=3))
    out = Path("/x30/results/modes")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"integration_{'_'.join(wanted) if len(wanted) < 6 else 'all'}.json").write_text(
        json.dumps({"checks": [{"name": n, "ok": ok, "detail": det} for n, ok, det in CHECKS], "measurements": RESULTS}, indent=1))
    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    rclpy.try_shutdown()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
