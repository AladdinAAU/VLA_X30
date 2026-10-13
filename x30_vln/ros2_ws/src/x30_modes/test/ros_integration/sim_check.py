"""Checks of x30_modes against the real simulator (kinematic locomotion): guard stopping, primitives, map-free skills.

Needs mux, guard, mode_manager and skill_server running with sim_overrides.yaml (run_sim_check.sh does that). goto_pose needs Nav2 and a
map and is skipped when navigate_to_pose is absent.  Writes results/modes/sim_check.json.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from x30_modes.cloud import cloud_xyz
from x30_modes.geometry import Footprint, Pose2D, wrap_pi, yaw_from_quat
from x30_modes.guard_logic import gap_to_footprint
from x30_msgs.action import Skill
from x30_msgs.msg import SafetyStatus
from x30_msgs.srv import Json, SetMode

OUT: dict = {"checks": [], "guard": [], "primitives": []}


def check(name, ok, detail=""):
    OUT["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


class D(Node):
    def __init__(self):
        super().__init__("sim_check", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.gt = self.odom = self.safety = self.cloud = None
        self.create_subscription(PoseStamped, "/gt/pose", lambda m: setattr(self, "gt", m), 10)
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", m), 10)
        self.create_subscription(SafetyStatus, "/x30/safety", lambda m: setattr(self, "safety", m), 10)
        self.create_subscription(PointCloud2, "/lidar_points", lambda m: setattr(self, "cloud", m),
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        self.pub = self.create_publisher(Twist, "/cmd_vel_nav", 10)
        self.c_mode = self.create_client(SetMode, "/x30/set_mode")
        self.c_pose = self.create_client(Json, "/sim/set_robot_pose")
        self.ac: dict = {}

    def gtpose(self):
        q = self.gt.pose.orientation
        return Pose2D(self.gt.pose.position.x, self.gt.pose.position.y, yaw_from_quat(q.x, q.y, q.z, q.w))

    def call(self, client, req):
        client.wait_for_service(10.0)
        f = client.call_async(req)
        while not f.done():
            time.sleep(0.01)
        return f.result()

    def teleport(self, x, y, yaw):
        r = Json.Request()
        r.request = json.dumps({"x": x, "y": y, "yaw": yaw})
        res = self.call(self.c_pose, r)
        time.sleep(1.5)
        return res

    def mode(self, m):
        r = SetMode.Request()
        r.mode = m
        return self.call(self.c_mode, r)

    def skill(self, name, args, wait=120.0):
        c = self.ac.setdefault(name, ActionClient(self, Skill, f"/x30/skill/{name}"))
        assert c.wait_for_server(10.0)
        g = Skill.Goal()
        g.name, g.args_json = name, json.dumps(args)
        f = c.send_goal_async(g)
        while not f.done():
            time.sleep(0.01)
        rf = f.result().get_result_async()
        t0 = time.monotonic()
        while not rf.done() and time.monotonic() - t0 < wait:
            time.sleep(0.02)
        r = rf.result().result
        return r.success, r.message, json.loads(r.data_json or "{}")

    def drive(self, vx=0.0, vy=0.0, wz=0.0, secs=1.0):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = vx, vy, wz
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            self.pub.publish(m)
            time.sleep(0.05)

    def lidar_gap(self, fp=Footprint(0.55, 0.55, 0.32)):
        """Clearance from the raw cloud (lidar_link = body frame): valid returns only, same height band as the guard."""
        xyz = cloud_xyz(self.cloud, drop_zero_intensity=True)
        xyz = xyz[(xyz[:, 2] > -0.25) & (xyz[:, 2] < 0.7)]
        ax, ay = np.abs(xyz[:, 0]), np.abs(xyz[:, 1])
        outside = ~(((ax < 0.60) & (ay < 0.37)) | ((ax < 0.95) & (ay > 0.28) & (ay < 0.45)))     # own body + legs (as the guard)
        return gap_to_footprint(xyz[outside, :2], fp), int(outside.sum())

    def drive_until_stopped(self, vx, max_wall=70.0):
        """Command vx until the robot stops moving (guard) or max_wall seconds; returns travelled distance."""
        g0 = self.gtpose()
        t0 = time.monotonic()
        last, last_t = (g0.x, g0.y), t0
        moved = False
        while time.monotonic() - t0 < max_wall:
            self.drive(vx=vx, secs=0.5)
            p = self.gtpose()
            if math.hypot(p.x - last[0], p.y - last[1]) > 0.02:
                last, last_t, moved = (p.x, p.y), time.monotonic(), True
            elif moved and time.monotonic() - last_t > 4.0:
                break
        return math.hypot(self.gtpose().x - g0.x, self.gtpose().y - g0.y)


def main():
    rclpy.init()
    d = D()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(d)
    threading.Thread(target=ex.spin, daemon=True).start()
    t0 = time.monotonic()
    while (d.gt is None or d.odom is None or d.cloud is None) and time.monotonic() - t0 < 60:
        time.sleep(0.2)
    check("sim contract topics present (/gt/pose, /odom, /lidar_points)", d.gt is not None and d.odom is not None and d.cloud is not None)
    d.mode("explore")
    gap0, n = d.lidar_gap()
    check("lidar cloud usable for the guard", n > 500, f"{n} valid points in the guard height band, current gap {gap0:.2f} m")
    time.sleep(1.0)
    check("guard publishes SafetyStatus and is not in no_lidar", d.safety is not None and d.safety.reason != "no_lidar",
          f"reason='{d.safety.reason if d.safety else None}'")
    for speed in (0.2, 0.4, 0.6):
        d.teleport(-4.0, 0.0, 0.0)
        gt0 = d.gtpose()
        d.drive_until_stopped(speed)
        time.sleep(1.5)
        gap, _ = d.lidar_gap()
        gt1 = d.gtpose()
        row = {"speed": speed, "travel_m": round(gt1.x - gt0.x, 2), "lidar_gap_m": round(gap, 3), "guard_clearance_m": round(d.safety.min_obstacle_distance, 3),
               "guard_reason": d.safety.reason, "final_x": round(gt1.x, 2)}
        OUT["guard"].append(row)
        check(f"guard stops before the wall at {speed} m/s", gap >= 0.0, json.dumps(row))
    d.teleport(0.0, 0.0, 0.0)
    for name, args, kind in (("move_relative", {"distance_cm": 100}, "cm"), ("move_relative", {"direction": "backward", "distance_cm": 50}, "cm"),
                             ("move_relative", {"direction": "left", "distance_cm": 30}, "cm"), ("turn_to", {"degrees": 90}, "deg"),
                             ("turn_to", {"degrees": -90}, "deg"), ("turn_to", {"degrees": 45}, "deg")):
        g0 = d.gtpose()
        ok, msg, data = d.skill(name, args)
        time.sleep(1.0)
        g1 = d.gtpose()
        if kind == "cm":
            ux, uy = {"forward": (1, 0), "backward": (-1, 0), "left": (0, 1), "right": (0, -1)}[args.get("direction", "forward")]
            c, s = math.cos(g0.yaw), math.sin(g0.yaw)
            got = ((g1.x - g0.x) * (c * ux - s * uy) + (g1.y - g0.y) * (s * ux + c * uy)) * 100
            req = args["distance_cm"]
        else:
            got, req = math.degrees(wrap_pi(g1.yaw - g0.yaw)), args["degrees"]
        row = {"cmd": f"{name} {json.dumps(args)}", "requested": req, "ground_truth": round(got, 1), "error": round(got - req, 1), "unit": kind, "ok": ok}
        OUT["primitives"].append(row)
        check(f"sim primitive {row['cmd']}", ok and abs(got - req) <= 4.0, json.dumps(row))
    d.teleport(2.0, 0.0, 0.0)
    ok, msg, data = d.skill("move_relative", {"distance_cm": 500})
    check("sim move_relative into the wall stops with a guard message", (not ok) and "safety guard" in msg, msg[:160])
    d.teleport(0.0, 0.0, 0.0)
    ok, msg, data = d.skill("scan", {"sweep_deg": 360, "step_deg": 90, "dwell_s": 0.3})
    check("sim scan 360", ok, msg[:160])
    from nav2_msgs.action import NavigateToPose
    if ActionClient(d, NavigateToPose, "navigate_to_pose").wait_for_server(3.0):
        ok, msg, data = d.skill("goto_pose", {"x": 0.0, "y": 0.0}, wait=120)
        check("sim goto_pose", ok, msg[:160])
    else:
        print("[SKIP] goto_pose: no Nav2 navigate_to_pose server in this run (sim verification of Nav2 skills pending)")
        OUT["goto_pose"] = "skipped: Nav2 not available"
    Path("/x30/results/modes").mkdir(parents=True, exist_ok=True)
    Path("/x30/results/modes/sim_check.json").write_text(json.dumps(OUT, indent=1))
    bad = [c for c in OUT["checks"] if not c["ok"]]
    print(f"{len(OUT['checks']) - len(bad)}/{len(OUT['checks'])} checks passed")
    rclpy.try_shutdown()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
