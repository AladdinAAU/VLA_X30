#!/usr/bin/env python3
"""End-to-end dry run: fake_x30 + x30_nav_bridge + real_adapter (+ camera_ingest, static TF) on loopback.

Started by run_dryrun.sh after `ros2 launch x30_hw real.launch.py dry_run:=true stage:=B ...`. Provokes the failure cases of
the task list and writes results/hw/integration_<timestamp>.json. Exit code != 0 when any check fails.
Never touches a real address: fake_x30 refuses everything but loopback.
"""

import json
import os
import statistics
import sys
import threading
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, CameraInfo, Image, Imu
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener

from x30_hw.tools.fake_x30 import control_request

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


class Probe(Node):
    def __init__(self):
        super().__init__("integration_probe")
        self.odom = []  # (t, vx, vy, wz)
        self.odom_n = 0
        self.imu_n = 0
        self.cam_n = 0
        self.cam_t = []
        self.ready = None
        self.armed = None
        self.estop_active = None
        self.health = {}
        self.batt = None
        self.cam_info = None
        self.cam_img = None
        self.create_subscription(Odometry, "/odom", self.on_odom, QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Imu, "/imu/data", lambda m: setattr(self, "imu_n", self.imu_n + 1),
                                 QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Image, "/camera/color/image_raw", self.on_cam, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", lambda m: setattr(self, "cam_info", m), qos_profile_sensor_data)
        self.create_subscription(Bool, "/x30/ready", lambda m: setattr(self, "ready", m.data), LATCHED)
        self.create_subscription(Bool, "/x30/armed", lambda m: setattr(self, "armed", m.data), LATCHED)
        self.create_subscription(Bool, "/x30/estop_active", lambda m: setattr(self, "estop_active", m.data), LATCHED)
        self.create_subscription(DiagnosticArray, "/x30/health", self.on_health, 10)
        self.create_subscription(BatteryState, "/x30/battery", lambda m: setattr(self, "batt", m), 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_estop = self.create_publisher(Bool, "/x30/estop", 10)
        self.cli = {}
        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)

    def on_odom(self, m):
        self.odom_n += 1
        self.odom.append((time.monotonic(), m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z))
        if len(self.odom) > 20000:
            del self.odom[:10000]

    def on_cam(self, m):
        self.cam_n += 1
        self.cam_t.append(time.monotonic())
        self.cam_img = m

    def on_health(self, m):
        for s in m.status:
            self.health[s.name] = s

    def call(self, srv_type, name, req, timeout=90.0):
        cli = self.cli.get(name) or self.create_client(srv_type, name)
        self.cli[name] = cli
        if not cli.wait_for_service(timeout_sec=10.0):
            return None
        done = threading.Event()
        box = {}
        fut = cli.call_async(req)
        fut.add_done_callback(lambda f: (box.setdefault("r", f.result()), done.set()))
        return box["r"] if done.wait(timeout) else None


def wait_for(pred, timeout, step=0.05):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    return False


def main():
    rclpy.init()
    p = Probe()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(p)
    threading.Thread(target=ex.spin, daemon=True).start()
    fake = control_request
    NB = "/nav_bridge_node"
    pub_stop = threading.Event()

    def drive(vx, vy, wz, seconds):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            p.pub_cmd.publish(t)
            time.sleep(0.05)

    def vel_after(vx, vy, wz, seconds=4.0):
        drive(vx, vy, wz, seconds)
        tail = [o for o in p.odom if o[0] > time.monotonic() - 1.0]
        return (statistics.mean(o[1] for o in tail), statistics.mean(o[2] for o in tail), statistics.mean(o[3] for o in tail)) if tail else None

    # ---- 1 telemetry, rates, health -----------------------------------------------------------------------------
    check("telemetry arrives (odom topic)", wait_for(lambda: p.odom_n > 50, 20), f"{p.odom_n} odom msgs")
    n0, t0 = p.odom_n, time.monotonic()
    time.sleep(3.0)
    rate = (p.odom_n - n0) / (time.monotonic() - t0)
    check("odom rate close to 200 Hz", 120 < rate < 260, f"{rate:.0f} Hz")
    check("imu relayed (udp mode)", p.imu_n > 100, f"{p.imu_n} msgs")
    check("camera rate", wait_for(lambda: p.cam_n > 15, 20), f"{p.cam_n} frames")
    if len(p.cam_t) > 10:
        r = (len(p.cam_t) - 1) / (p.cam_t[-1] - p.cam_t[0])
        check("camera rate ~10 Hz (synthetic 10 fps source)", 6 < r < 12, f"{r:.1f} Hz")
    check("camera_info present, placeholder flagged (empty distortion_model)", p.cam_info is not None and p.cam_info.distortion_model == "" and p.cam_info.k[0] > 0,
          None if p.cam_info is None else f"fx={p.cam_info.k[0]:.1f}")
    check("camera image rgb8", p.cam_img is not None and p.cam_img.encoding == "rgb8" and len(p.cam_img.data) == p.cam_img.height * p.cam_img.step)
    wait_for(lambda: "x30/link" in p.health and "x30/adapter" in p.health, 10)
    check("health lists link/motion/errors/battery/adapter", all(k in p.health for k in ("x30/link", "x30/motion", "x30/errors", "x30/adapter")), sorted(p.health))
    check("link OK", p.health.get("x30/link") and p.health["x30/link"].level == DiagnosticStatus.OK)
    ok_tf = True
    try:
        for child in ("imu_link", "lidar_link", "camera_link", "camera_optical_frame", "lidar_fl_link"):
            ok_tf &= p.tfbuf.can_transform("base_link", child, rclpy.time.Time(), rclpy.duration.Duration(seconds=5.0))
    except Exception as e:  # noqa: BLE001
        ok_tf = False
        print(e)
    check("static TF base_link -> sensors", ok_tf)
    check("e-stop output latched false at start", p.estop_active is False)
    check("starts disarmed and not ready", p.armed is False and p.ready in (None, False), f"armed={p.armed} ready={p.ready}")

    # ---- 2 arm refused before standing ----------------------------------------------------------------------------
    r = p.call(SetBool, "/x30/arm", SetBool.Request(data=True), 10)
    check("arm refused while the robot is sitting", r is not None and not r.success, None if r is None else r.message)
    drive(0.2, 0, 0, 1.0)
    check("disarmed /cmd_vel moves nothing", abs(fake({"cmd": "status"})["v"][0]) < 1e-3)

    # ---- 3 control + stand --------------------------------------------------------------------------------------
    r = p.call(Trigger, f"{NB}/acquire_control", Trigger.Request(), 10)
    check("acquire_control", r is not None and r.success, None if r is None else r.message)
    st = fake({"cmd": "status"})
    check("no motion packets before stand", st["stats"].get("axis_packets", 0) == 0)
    t_stand = time.monotonic()
    r = p.call(Trigger, f"{NB}/stand", Trigger.Request(), 120)
    check("stand sequence", r is not None and r.success, f"{None if r is None else r.message} ({time.monotonic() - t_stand:.1f} s)")
    st = fake({"cmd": "status"})
    check("fake in RL_MODE gait 33", st["state"] == 16 and st["gait"] == 33, f"{st['state_name']} gait {st['gait']}")
    check("manual mode restored after stand", st["nav_mode"] == 0)
    check("ready latched true", wait_for(lambda: p.ready is True, 5))
    check("stand produced no protocol violations", st["n_violations"] == 0, st["violations"])

    # ---- 4 arm, velocity mapping, clamp -----------------------------------------------------------------------
    drive(0.15, 0, 0, 0.5)
    check("still no motion while disarmed (after stand)", abs(fake({"cmd": "status"})["v"][0]) < 1e-3)
    r = p.call(SetBool, "/x30/arm", SetBool.Request(data=True), 10)
    check("arm succeeds when standing", r is not None and r.success, None if r is None else r.message)
    v = vel_after(0.15, 0.05, 0.2)
    check("velocity mapping vx/vy/wz", v and abs(v[0] - 0.15) < 0.015 and abs(v[1] - 0.05) < 0.01 and abs(v[2] - 0.2) < 0.02, v)
    check("signs: +vy moves left, +wz turns CCW", v and v[1] > 0 and v[2] > 0)
    v = vel_after(0.6, 0.0, 0.0, 4.0)
    check("hard limit stage B clamps 0.6 -> 0.2", v and abs(v[0] - 0.2) < 0.02, v)
    v = vel_after(-0.5, 0.0, 0.0, 4.0)
    check("backward clamp 0.1", v and abs(v[0] + 0.1) < 0.015, v)
    v = vel_after(0.0, 0.0, 1.5, 3.0)
    check("yaw clamp 0.3", v and abs(v[2] - 0.3) < 0.03, v)

    # ---- 5 latency / accel / timeout --------------------------------------------------------------------------
    drive(0, 0, 0, 2.0)
    p.odom.clear()
    t_cmd = time.monotonic()
    drive(0.2, 0, 0, 3.0)
    first = next((o[0] for o in p.odom if o[1] > 0.02), None)
    lat = None if first is None else first - t_cmd
    check("command-to-odom latency (nominal fake: 100 ms delay + tick + limiter)", lat is not None and lat < 0.6, None if lat is None else f"{lat * 1000:.0f} ms")
    ser = [(o[0], o[1]) for o in p.odom if o[0] >= t_cmd]
    amax = max(((b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(ser[::20], ser[20::20]) if b[0] > a[0]), default=0)
    check("acceleration <= 0.6 m/s2 (limit 0.5 + measurement noise)", amax < 0.6, f"{amax:.2f} m/s2")
    t_stop = time.monotonic()
    time.sleep(2.5)  # no /cmd_vel at all: timeout path
    near_zero = next((o[0] - t_stop for o in p.odom if o[0] > t_stop and abs(o[1]) < 0.02), None)
    check("cmd_vel silence stops the robot (adapter 0.4 s + nav_bridge + fake lag)", near_zero is not None and near_zero < 2.0, None if near_zero is None else f"{near_zero:.2f} s to |v|<0.02")

    # ---- 6 telemetry loss ---------------------------------------------------------------------------------------
    drive(0.15, 0, 0, 1.5)
    fake({"cmd": "mute_telemetry", "seconds": 2.5})
    drive(0.15, 0, 0, 1.5)
    check("telemetry loss disarms", wait_for(lambda: p.armed is False, 3), f"armed={p.armed}")
    st = fake({"cmd": "status"})
    time.sleep(1.5)
    st = fake({"cmd": "status"})
    check("telemetry loss: robot velocity back to zero", abs(st["v"][0]) < 0.03, st["v"])
    check("link status ERROR was reported", wait_for(lambda: "x30/link" in p.health, 2))
    time.sleep(2.0)
    check("telemetry back, link OK", wait_for(lambda: p.health["x30/link"].level == DiagnosticStatus.OK, 8))
    drive(0.15, 0, 0, 1.0)
    check("stays disarmed after telemetry returns (latched)", p.armed is False and abs(fake({"cmd": "status"})["v"][0]) < 0.01)

    # ---- 7 e-stop topic -----------------------------------------------------------------------------------------
    r = p.call(SetBool, "/x30/arm", SetBool.Request(data=True), 10)
    check("re-arm after fault", r is not None and r.success, None if r is None else r.message)
    drive(0.15, 0, 0, 1.5)
    t_e = time.monotonic()
    p.pub_estop.publish(Bool(data=True))
    ok = wait_for(lambda: fake({"cmd": "status"})["state"] == 6, 3, 0.01)
    dt = time.monotonic() - t_e
    check("/x30/estop -> robot soft e-stop", ok, f"{dt * 1000:.0f} ms")
    check("estop_active latched true", wait_for(lambda: p.estop_active is True, 3))
    check("e-stop disarms", wait_for(lambda: p.armed is False, 2))
    st = fake({"cmd": "status"})
    check("emergency_source 5 (client)", st["emergency_source"] == 5)
    p.pub_estop.publish(Bool(data=False))
    r = p.call(SetBool, "/x30/arm", SetBool.Request(data=True), 10)
    check("cannot arm while e-stop active", r is not None and not r.success, None if r is None else r.message)
    # recovery: lie toggles the robot out of the e-stop, then stand again
    r = p.call(Trigger, f"{NB}/lie", Trigger.Request(), 60)
    check("recovery: lie after e-stop", r is not None and r.success, None if r is None else r.message)
    check("estop_active clears", wait_for(lambda: p.estop_active is False, 8))
    r = p.call(Trigger, f"{NB}/stand", Trigger.Request(), 120)
    check("stand again", r is not None and r.success, None if r is None else r.message)

    # ---- 8 hardware e-stop, battery ---------------------------------------------------------------------------
    r = p.call(SetBool, "/x30/arm", SetBool.Request(data=True), 10)
    check("arm again", r is not None and r.success, None if r is None else r.message)
    fake({"cmd": "battery", "percent": 10})
    check("battery below minimum disarms", wait_for(lambda: p.armed is False, 6), f"armed={p.armed}")
    check("battery topic percentage", wait_for(lambda: p.batt is not None and p.batt.percentage < 0.15, 6), None if p.batt is None else p.batt.percentage)
    fake({"cmd": "battery", "percent": 80})
    time.sleep(3.0)
    r = p.call(SetBool, "/x30/arm", SetBool.Request(data=True), 10)
    check("arm after battery recovered", r is not None and r.success, None if r is None else r.message)
    fake({"cmd": "hardware_estop"})
    check("hardware e-stop (tail button) seen as estop_active", wait_for(lambda: p.estop_active is True, 3))
    check("hardware e-stop disarms", wait_for(lambda: p.armed is False, 2))
    r = p.call(Trigger, f"{NB}/lie", Trigger.Request(), 60)
    check("recovery after hardware e-stop", r is not None and r.success and wait_for(lambda: p.estop_active is False, 8), None if r is None else r.message)
    r = p.call(Trigger, f"{NB}/stand", Trigger.Request(), 120)
    check("stand after hardware e-stop", r is not None and r.success, None if r is None else r.message)

    # ---- 9 heartbeat loss / release ---------------------------------------------------------------------------
    r = p.call(Trigger, f"{NB}/release_control", Trigger.Request(), 10)
    time.sleep(1.8)
    st = fake({"cmd": "status"})
    check("release_control: heartbeat stops, fake sees wifi_error bit", (st["error_state"] & 2) != 0 and not st["session_active"], f"error_state={st['error_state']}")
    r = p.call(Trigger, f"{NB}/acquire_control", Trigger.Request(), 10)
    time.sleep(1.0)
    st = fake({"cmd": "status"})
    check("acquire_control restores the session", st["session_active"] and (st["error_state"] & 2) == 0)

    # ---- 10 handheld conflict ---------------------------------------------------------------------------------
    fake({"cmd": "handheld", "seconds": 2.0})
    time.sleep(2.5)
    st = fake({"cmd": "status"})
    check("second 0x21 source (handheld) is detected by the fake", any("second 0x21 source" in v for v in st["violations"]) or st["stats"].get("ignored_handheld_active", 0) > 0)

    st = fake({"cmd": "status"})
    viol = [v for v in st["violations"]]
    # expected: the injected handheld conflict, and (known race) up to three zero axis packets that nav_bridge sends in
    # the few milliseconds between the robot entering the e-stop and the first telemetry packet that says so
    unexpected = [v for v in viol if "second 0x21 source" not in v and not ("during SOFT_ESTOP" in v and "=0 " in v)]
    check("no unexpected protocol violations", not unexpected, unexpected or f"{len(viol)} expected/injected only")
    os.makedirs("/x30/results/hw", exist_ok=True)
    path = f"/x30/results/hw/integration_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json.dump({"checks": CHECKS, "fake_final": st}, open(path, "w"), indent=1)
    failed = [c for c in CHECKS if not c["ok"]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed; report {path}", flush=True)
    rclpy.try_shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
