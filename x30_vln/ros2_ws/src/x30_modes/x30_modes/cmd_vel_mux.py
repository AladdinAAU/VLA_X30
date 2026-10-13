"""cmd_vel_mux: priority teleop > nav, per-source timeouts, gated by the current mode. Publishes /cmd_vel_raw at a fixed rate."""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from .mux_logic import MuxCore
from .ros_io import LATCHED_QOS, now_s


def twist3(m: Twist) -> tuple[float, float, float]:
    return float(m.linear.x), float(m.linear.y), float(m.angular.z)


class CmdVelMux(Node):
    def __init__(self):
        super().__init__("cmd_vel_mux")
        p = self.declare_parameter
        rate = float(p("rate_hz", 20.0).value)
        self.core = MuxCore(
            timeouts={"teleop": float(p("teleop_timeout_s", 0.5).value), "nav": float(p("nav_timeout_s", 0.5).value)},
            teleop_lockout_s=float(p("teleop_lockout_s", 0.5).value),
            mode=str(p("initial_mode", "idle").value))
        self.gate_by_mode = bool(p("gate_by_mode", True).value)
        if not self.gate_by_mode:
            self.core.set_mode("object_nav")             # any task mode allows teleop + nav
        self.pub = self.create_publisher(Twist, str(p("output_topic", "/cmd_vel_raw").value), 10)
        self.pub_src = self.create_publisher(String, "/x30/cmd_source", 10)
        self.create_subscription(Twist, str(p("teleop_topic", "/cmd_vel_teleop").value), self.on_teleop, 10)
        self.create_subscription(Twist, str(p("nav_topic", "/cmd_vel_nav").value), self.on_nav, 10)
        if self.gate_by_mode:
            self.create_subscription(String, "/x30/mode", self.on_mode, LATCHED_QOS)
        self.last_source = None
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"cmd_vel_mux ready (mode gating {'on' if self.gate_by_mode else 'off'}, start mode {self.core.mode})")

    def on_teleop(self, m: Twist) -> None:
        self.core.update("teleop", twist3(m), now_s(self))

    def on_nav(self, m: Twist) -> None:
        self.core.update("nav", twist3(m), now_s(self))

    def on_mode(self, m: String) -> None:
        self.core.set_mode(m.data)
        self.get_logger().info(f"mode {m.data}: sources allowed {self.core.mode}")

    def tick(self) -> None:
        res = self.core.select(now_s(self))
        out = Twist()
        out.linear.x, out.linear.y, out.angular.z = res.twist
        self.pub.publish(out)
        if res.source != self.last_source:
            self.last_source = res.source
            self.pub_src.publish(String(data=res.source))
            self.get_logger().info(f"command source: {res.source or 'none'}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
