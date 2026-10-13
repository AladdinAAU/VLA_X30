"""real_adapter: makes the real X30 look like the INTERFACES.md robot contract, safely.

  /x30/leg_odom (nav_bridge)      -> /odom            frame ids, reliable QoS, stamp + robot clock offset
  /x30/imu/data (nav_bridge)      -> /imu/data        only when imu_source=udp; ros1 mode only monitors the bridged topic
  /x30/battery_state, diagnostics -> /x30/battery (1 Hz), /x30/health (1 Hz), /x30/ready (latched)
  /cmd_vel (safety_guard)         -> /x30/cmd_vel_hw  ONLY when armed and every interlock holds
  /x30/estop (input)              -> disarm now, soft e-stop via nav_bridge (topic and service, belt and braces)
  /x30/arm (SetBool)              default DISARMED; arming asks nav_bridge ~/enable_motion
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Dict, Optional

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import BatteryState, Image, Imu, PointCloud2
from std_msgs.msg import Bool, Int32
from std_srvs.srv import SetBool, Trigger

from x30_hw import config as cfgmod
from x30_hw import protocol as P
from x30_hw.safety_gate import (GateLimits, Interlocks, RateMonitor, SafetyGate, clock_lag_level, rate_level)

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
RELIABLE_BIG = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)


def cdr_stamp(raw: bytes) -> Optional[float]:
    """Header stamp (s) of a CDR-serialised message that starts with std_msgs/Header (little endian)."""
    if len(raw) < 12 or raw[1] != 1:
        return None
    sec, nsec = struct.unpack_from("<iI", raw, 4)
    return sec + nsec * 1e-9


class RealAdapter(Node):
    def __init__(self):
        super().__init__("real_adapter")
        self.declare_parameter("config_file", "")
        self.declare_parameter("stage", "")
        self.declare_parameter("camera_expected", False)
        self.declare_parameter("imu_source", "")
        cfg = cfgmod.load_robot_config(self.get_parameter("config_file").value or None)
        self.cfg = cfg
        stage = self.get_parameter("stage").value or cfg.get("stage_default", "B")
        sl = cfgmod.stage_limits(cfg, stage)
        ad = cfg["adapter"]
        self.stage = stage
        self.ad = ad
        self.gate = SafetyGate(
            GateLimits(sl["vx_forward"], sl["vx_backward"], sl["vy"], sl["wz"], ad["no_lateral_above_vx"],
                       ad["cmd_timeout_s"], bool(sl["allow_motion"])),
            ad["battery_min_percent_to_arm"], ad["battery_disarm_percent"])
        self.offset = Duration(seconds=float(cfg["clock"]["robot_minus_pc_s"]))
        self.imu_source = self.get_parameter("imu_source").value or ad["imu_source"]
        self.camera_expected = bool(self.get_parameter("camera_expected").value) and cfg["camera"]["enabled"]
        self.t0 = time.monotonic()

        # state
        self.lock = threading.Lock()
        self.basic_state = -1
        self.gait_state = -1
        self.last_state_t = -1e9
        self.battery: Optional[BatteryState] = None
        self.nb_estop_active = False
        self.estop_request = False
        self.nb_status: Dict[str, DiagnosticStatus] = {}
        self.nb_status_t = -1e9
        self.last_cmd = (0.0, 0.0, 0.0)
        self.last_cmd_t = -1e9
        self.fwd_status = "disarmed"
        self.rates = {k: RateMonitor() for k in ("lidar", "imu", "odom", "camera")}
        self.lidar_lag_samples = []
        self.ready = False
        self._estop_deadline: Optional[float] = None

        cb = ReentrantCallbackGroup()
        cmd_cb = MutuallyExclusiveCallbackGroup()
        # publishers
        self.pub_odom = self.create_publisher(Odometry, "/odom", RELIABLE_BIG)
        self.pub_imu = self.create_publisher(Imu, "/imu/data", RELIABLE_BIG) if self.imu_source == "udp" else None
        self.pub_ready = self.create_publisher(Bool, "/x30/ready", LATCHED)
        self.pub_armed = self.create_publisher(Bool, "/x30/armed", LATCHED)
        self.pub_batt = self.create_publisher(BatteryState, "/x30/battery", 10)
        self.pub_health = self.create_publisher(DiagnosticArray, "/x30/health", 10)
        self.pub_cmd = self.create_publisher(Twist, ad["cmd_vel_hw"], 10)
        # subscriptions
        self.create_subscription(Odometry, "/x30/leg_odom", self.on_odom, qos_profile_sensor_data, callback_group=cb)
        self.create_subscription(Imu, "/x30/imu/data", self.on_udp_imu, qos_profile_sensor_data, callback_group=cb)
        self.create_subscription(Imu, "/imu/data", self.on_bus_imu, qos_profile_sensor_data, callback_group=cb)
        self.create_subscription(PointCloud2, "/lidar_points", self.on_lidar, qos_profile_sensor_data, callback_group=cb,
                                 raw=True)
        if self.camera_expected:
            self.create_subscription(Image, "/camera/color/image_raw", self.on_camera, qos_profile_sensor_data,
                                     callback_group=cb, raw=True)
        self.create_subscription(Int32, "/x30/basic_state", self.on_basic, 10, callback_group=cb)
        self.create_subscription(Int32, "/x30/gait_state", self.on_gait, 10, callback_group=cb)
        self.create_subscription(BatteryState, "/x30/battery_state", self.on_battery, 10, callback_group=cb)
        self.create_subscription(DiagnosticArray, "/x30/diagnostics", self.on_nb_diag, 10, callback_group=cb)
        self.create_subscription(Bool, "/x30/estop_active", self.on_nb_estop, LATCHED, callback_group=cb)
        self.create_subscription(Bool, "/x30/estop", self.on_estop, 10, callback_group=cb)
        self.create_subscription(Twist, ad["cmd_vel_in"], self.on_cmd, 10, callback_group=cmd_cb)
        # services and clients
        self.create_service(SetBool, "/x30/arm", self.on_arm, callback_group=cb)
        self.cli_enable = self.create_client(SetBool, "/nav_bridge_node/enable_motion", callback_group=cb)
        self.cli_estop = self.create_client(Trigger, "/nav_bridge_node/soft_estop", callback_group=cb)
        # timers
        self.create_timer(0.05, self.on_interlock_timer, callback_group=cb)
        self.create_timer(float(ad["health_period_s"]), self.on_health_timer, callback_group=cb)
        self.get_logger().info(
            f"real_adapter up: stage {stage} (motion {'allowed' if sl['allow_motion'] else 'NOT allowed'}), limits "
            f"+{sl['vx_forward']}/-{sl['vx_backward']} m/s vy {sl['vy']} wz {sl['wz']}, imu_source {self.imu_source}, "
            f"DISARMED (call /x30/arm to arm)")
        self._publish_armed()

    # ------------------------------------------------------------------ helpers
    def wall(self) -> float:
        return time.monotonic() - self.t0

    def shifted(self, stamp):
        return (Time.from_msg(stamp) + self.offset).to_msg()

    def interlocks(self) -> Interlocks:
        now = self.wall()
        with self.lock:
            il = Interlocks()
            il.estop = self.nb_estop_active or self.estop_request
            fresh = (now - self.last_state_t) < 1.0 and (now - self.nb_status_t) < 3.0
            link = self.nb_status.get("x30_nav_bridge/link")
            il.telemetry_ok = fresh and link is not None and link.level == DiagnosticStatus.OK
            gait_ok = self.gait_state in P.DEFAULT_NAV_BRIDGE_TABLE
            il.robot_ready = il.telemetry_ok and gait_ok and self.basic_state in (int(P.BasicState.STEPPING), int(P.BasicState.RL_MODE))
            err = self.nb_status.get("x30_nav_bridge/errors")
            il.critical_error = err is not None and err.level == DiagnosticStatus.ERROR
            il.battery_percent = None if self.battery is None else self.battery.percentage * 100.0
        return il

    def _publish_armed(self) -> None:
        self.pub_armed.publish(Bool(data=self.gate.armed))

    def _call_enable(self, value: bool) -> bool:
        if not self.cli_enable.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("nav_bridge ~/enable_motion not available")
            return False
        done = threading.Event()
        box = {}
        fut = self.cli_enable.call_async(SetBool.Request(data=value))
        fut.add_done_callback(lambda f: (box.setdefault("r", f.result()), done.set()))
        if not done.wait(2.0) or box.get("r") is None:
            return False
        if not box["r"].success:
            self.get_logger().warn(f"nav_bridge refused enable_motion={value}: {box['r'].message}")
        return bool(box["r"].success)

    def _disarm(self, reason: str) -> None:
        was = self.gate.armed
        self.gate.disarm(reason)
        self.pub_cmd.publish(Twist())
        if was:
            self.get_logger().error(f"DISARMED: {reason}")
            self.cli_enable.call_async(SetBool.Request(data=False))
        self._publish_armed()

    # ------------------------------------------------------------------ sensor relays
    def on_odom(self, msg: Odometry) -> None:
        self.rates["odom"].tick(self.wall())
        msg.header.stamp = self.shifted(msg.header.stamp)
        msg.header.frame_id = self.cfg["odometry"]["frame_id"]
        msg.child_frame_id = self.cfg["odometry"]["child_frame_id"]
        self.pub_odom.publish(msg)

    def on_udp_imu(self, msg: Imu) -> None:
        if self.pub_imu is None:
            return
        msg.header.stamp = self.shifted(msg.header.stamp)
        self.pub_imu.publish(msg)

    def on_bus_imu(self, msg: Imu) -> None:
        if self.imu_source == "ros1":
            self.rates["imu"].tick(self.wall())
            if msg.header.frame_id != self.cfg["frames"]["imu"]:
                self.get_logger().warn(
                    f"bridged /imu/data has frame_id '{msg.header.frame_id}', expected '{self.cfg['frames']['imu']}'; "
                    "add a static TF alias or switch imu_source to udp", throttle_duration_sec=30.0)
        elif self.imu_source == "udp":
            self.rates["imu"].tick(self.wall())  # our own republished messages

    def on_lidar(self, raw: bytes) -> None:
        now = self.wall()
        self.rates["lidar"].tick(now)
        st = cdr_stamp(raw)
        if st is not None:
            lag = self.get_clock().now().nanoseconds * 1e-9 - st
            self.lidar_lag_samples.append((now, lag))

    def on_camera(self, _raw: bytes) -> None:
        self.rates["camera"].tick(self.wall())

    # ------------------------------------------------------------------ robot state
    def on_basic(self, msg: Int32) -> None:
        with self.lock:
            self.basic_state = msg.data
            self.last_state_t = self.wall()

    def on_gait(self, msg: Int32) -> None:
        with self.lock:
            self.gait_state = msg.data

    def on_battery(self, msg: BatteryState) -> None:
        with self.lock:
            self.battery = msg

    def on_nb_diag(self, msg: DiagnosticArray) -> None:
        with self.lock:
            for s in msg.status:
                self.nb_status[s.name] = s
            self.nb_status_t = self.wall()

    def on_nb_estop(self, msg: Bool) -> None:
        with self.lock:
            self.nb_estop_active = msg.data
        if msg.data:
            self._disarm("nav_bridge reports e-stop active")

    def on_estop(self, msg: Bool) -> None:
        if msg.data and not self.estop_request:
            self.get_logger().error("/x30/estop requested: disarming and forwarding the soft e-stop")
            self.estop_request = True
            self._disarm("/x30/estop requested")
            self._estop_deadline = self.wall() + float(self.ad["estop_confirm_timeout_s"])
        elif not msg.data:
            self.estop_request = False
            self._estop_deadline = None

    # ------------------------------------------------------------------ command path
    def on_cmd(self, msg: Twist) -> None:
        now = self.wall()
        self.last_cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.last_cmd_t = now
        out, status = self.gate.evaluate(self.last_cmd, 0.0)
        if status != self.fwd_status:
            self.fwd_status = status
        if status in ("ok", "clamped"):
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = out
            self.pub_cmd.publish(t)
            if status == "clamped":
                self.get_logger().warn(f"/cmd_vel {self.last_cmd} clamped to stage limits -> {out}",
                                       throttle_duration_sec=2.0)

    def on_arm(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        if not req.data:
            self._disarm("operator disarm")
            res.success, res.message = True, "disarmed"
            return res
        il = self.interlocks()
        ok, msg = self.gate.arm(il)
        if ok and not self._call_enable(True):
            self.gate.disarm("nav_bridge refused to enable motion")
            ok, msg = False, "refused: nav_bridge did not enable velocity forwarding (see its log; is the robot standing and the control acquired?)"
        if ok:
            self.get_logger().warn("ARMED: /cmd_vel is now forwarded to the robot. People >= 2 m away, handheld e-stop in reach.")
        self._publish_armed()
        res.success, res.message = ok, msg
        return res

    def on_interlock_timer(self) -> None:
        now = self.wall()
        if self.gate.armed:
            why = self.gate.check_interlocks(self.interlocks())
            if why:
                self._disarm(why)
            elif now - self.last_cmd_t > self.gate.limits.cmd_timeout_s and self.fwd_status in ("ok", "clamped"):
                self.fwd_status = "stale"
                self.pub_cmd.publish(Twist())
        if self._estop_deadline is not None and now > self._estop_deadline:
            self._estop_deadline = None
            if not self.nb_estop_active and self.cli_estop.service_is_ready():
                self.get_logger().error("nav_bridge did not confirm the e-stop in time: calling ~/soft_estop")
                self.cli_estop.call_async(Trigger.Request())
        il = self.interlocks()
        ready = il.robot_ready and not il.estop and not il.critical_error
        if ready != self.ready:
            self.ready = ready
            self.pub_ready.publish(Bool(data=ready))

    # ------------------------------------------------------------------ health (1 Hz)
    def on_health_timer(self) -> None:
        now = self.wall()
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        with self.lock:
            nb = dict(self.nb_status)
            batt = self.battery
        for name, s in nb.items():
            c = DiagnosticStatus(level=s.level, name=name.replace("x30_nav_bridge/", "x30/"), message=s.message,
                                 hardware_id=s.hardware_id, values=s.values)
            arr.status.append(c)
        if not nb:
            arr.status.append(DiagnosticStatus(level=DiagnosticStatus.ERROR, name="x30/link",
                                               message="no diagnostics from nav_bridge (is it running?)"))
        exp = self.ad["rate_expectations_hz"]
        for key, topic, exp_key, needed in (("lidar", "/lidar_points", "lidar_points", True),
                                            ("imu", "/imu/data", "imu", True), ("odom", "/odom", "odom", True),
                                            ("camera", "/camera/color/image_raw", "camera", self.camera_expected)):
            if not needed:
                continue
            r = self.rates[key].rate(now)
            lvl = rate_level(r, float(exp[exp_key]))
            msg = f"{r:.1f} Hz (expected {exp[exp_key]})"
            if key == "imu" and self.imu_source == "ros1" and r == 0.0:
                msg += ": bridged /imu/data absent; start the bridge or set imu_source: udp (with ros1_bridge_topics_udp_imu.yaml)"
            arr.status.append(DiagnosticStatus(level=bytes([lvl]), name=f"x30/{key}", message=msg,
                                               values=[KeyValue(key="topic", value=topic),
                                                       KeyValue(key="rate_hz", value=f"{r:.2f}")]))
        # clock: minimum apparent lag of robot-stamped lidar data over the last 30 s
        self.lidar_lag_samples = [(t, l) for t, l in self.lidar_lag_samples if now - t < 30.0]
        if self.lidar_lag_samples:
            lag_min = min(l for _, l in self.lidar_lag_samples)
            lvl, msg = clock_lag_level(lag_min)
            arr.status.append(DiagnosticStatus(level=bytes([lvl]), name="x30/clock", message=msg,
                                               values=[KeyValue(key="lidar_lag_min_s", value=f"{lag_min:.3f}"),
                                                       KeyValue(key="configured_offset_s", value=str(self.offset.nanoseconds * 1e-9))]))
        il = self.interlocks()
        arr.status.append(DiagnosticStatus(
            level=DiagnosticStatus.OK if not il.estop else DiagnosticStatus.ERROR,
            name="x30/adapter", message=f"stage {self.stage}, {'ARMED' if self.gate.armed else 'disarmed'}, "
                                         f"forwarding: {self.fwd_status}",
            values=[KeyValue(key="armed", value=str(self.gate.armed)), KeyValue(key="ready", value=str(self.ready)),
                    KeyValue(key="estop", value=str(il.estop)),
                    KeyValue(key="last_disarm_reason", value=self.gate.last_disarm_reason),
                    KeyValue(key="clamp_events", value=str(self.gate.clamp_events))]))
        self.pub_health.publish(arr)
        if batt is not None:
            self.pub_batt.publish(batt)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealAdapter()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
