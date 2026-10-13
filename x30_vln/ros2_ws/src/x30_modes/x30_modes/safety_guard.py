"""safety_guard node: /cmd_vel_raw -> /cmd_vel at a fixed rate, limited by obstacles, e-stop, tilt, lidar freshness."""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from x30_msgs.msg import SafetyStatus

from .geometry import roll_pitch_from_quat
from .guard_logic import GuardConfig, SafetyGuardCore
from .ros_io import LATCHED_QOS, SENSOR_QOS, ObstacleSource, declare_dataclass_params, now_s


class SafetyGuard(Node):
    def __init__(self):
        super().__init__("safety_guard")
        cfg = declare_dataclass_params(self, GuardConfig())
        self.core = SafetyGuardCore(cfg)
        p = self.declare_parameter
        self.base_frame = str(p("base_frame", "base_link").value)
        source = str(p("obstacle_source", "costmap_cloud").value)
        rate = float(p("rate_hz", 20.0).value)
        self.status_every = max(int(round(rate / float(p("status_rate_hz", 10.0).value))), 1)
        self.tilt_from_accel = bool(p("tilt_from_accel", True).value)
        self.cb_main = MutuallyExclusiveCallbackGroup()
        self.cb_io = ReentrantCallbackGroup()

        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.obstacles = ObstacleSource(
            self, self.buf, self.base_frame, source, str(p("costmap_cloud_topic", "/costmap_cloud").value),
            str(p("scan_topic", "/scan").value), float(p("z_min", -0.25).value), float(p("z_max", 0.70).value),
            callback_group=self.cb_io)

        self.cmd: tuple[float, float, float] | None = None
        self.cmd_stamp: float | None = None
        self.estop = False
        self.tilt_deg: float | None = None
        self.n = 0
        self.last_reason = None
        self.pub = self.create_publisher(Twist, str(p("output_topic", "/cmd_vel").value), 10)
        self.pub_status = self.create_publisher(SafetyStatus, "/x30/safety", 10)
        self.create_subscription(Twist, str(p("input_topic", "/cmd_vel_raw").value), self.on_cmd, 10, callback_group=self.cb_io)
        # the e-stop may be published latched or volatile: listen with both so a late start still sees a latched true
        self.create_subscription(Bool, "/x30/estop", self.on_estop, 10, callback_group=self.cb_io)
        self.create_subscription(Bool, "/x30/estop", self.on_estop, LATCHED_QOS, callback_group=self.cb_io)
        self.create_subscription(Imu, "/imu/data", self.on_imu, SENSOR_QOS, callback_group=self.cb_io)
        self.last_t = now_s(self)
        self.create_timer(1.0 / rate, self.tick, callback_group=self.cb_main)
        self.get_logger().info(f"safety_guard ready: footprint f{cfg.front}/r{cfg.rear}/w{cfg.half_width} m, stop margin {cfg.stop_margin} m, "
                               f"decel {cfg.decel} m/s2, latency {cfg.latency_s} s, limits vx[{cfg.vx_min},{cfg.vx_max}] vy{cfg.vy_max} wz{cfg.wz_max}, "
                               f"obstacle source {source}")

    def on_cmd(self, m: Twist) -> None:
        self.cmd = (float(m.linear.x), float(m.linear.y), float(m.angular.z))
        self.cmd_stamp = now_s(self)

    def on_estop(self, m: Bool) -> None:
        if m.data != self.estop:
            self.get_logger().warn(f"soft e-stop {'ACTIVE' if m.data else 'released'}")
        self.estop = bool(m.data)

    def on_imu(self, m: Imu) -> None:
        q = m.orientation
        if q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w > 0.5:
            roll, pitch = roll_pitch_from_quat(q.x, q.y, q.z, q.w)
            self.tilt_deg = math.degrees(max(abs(roll), abs(pitch)))
        elif self.tilt_from_accel:
            a = m.linear_acceleration
            g = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
            if g > 1.0:
                self.tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, a.z / g))))

    def tick(self) -> None:
        now = now_s(self)
        dt = max(now - self.last_t, 1e-3)
        self.last_t = now
        pts, stamp = self.obstacles.points(self.core.cfg.obstacle_timeout_s)
        res = self.core.step(now, min(dt, 0.25), self.cmd, self.cmd_stamp, pts, stamp, self.estop, self.tilt_deg)
        out = Twist()
        out.linear.x, out.linear.y, out.angular.z = res.twist
        self.pub.publish(out)
        self.n += 1
        if self.n % self.status_every == 0 or res.reason != self.last_reason:
            st = SafetyStatus()
            st.header.stamp = self.get_clock().now().to_msg()
            st.header.frame_id = self.base_frame
            st.limiting, st.stopped = bool(res.limiting), bool(res.stopped)
            st.min_obstacle_distance = float(res.min_obstacle_distance)
            st.scale = float(res.scale)
            st.reason = res.reason
            self.pub_status.publish(st)
        if res.reason != self.last_reason:
            self.last_reason = res.reason
            self.get_logger().info(f"guard state: {res.reason or 'clear'} (scale {res.scale:.2f}, clearance {res.min_obstacle_distance:.2f} m)")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyGuard()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
