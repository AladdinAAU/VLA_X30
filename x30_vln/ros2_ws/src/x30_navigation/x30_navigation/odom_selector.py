"""odom_selector: pick the odometry source (param odom_source: gt | leg | lidar | fused), publish /odometry/active and
the single `odom -> base_link` TF, and watch the sources for jumps, staleness and mutual disagreement.

gt     /gt/pose (sim only), re-based to start at the origin
leg    /odom (leg/proprioceptive odometry on the robot, ground truth with noise in sim)
lidar  /odometry/lidar (KISS-ICP)
fused  complementary filter: leg velocities integrated at the leg rate, corrected by lidar poses (odom_logic.py)

Switching at run time (ros2 param set /odom_selector odom_source lidar) keeps the output continuous: every
source is re-based on the last published pose when it becomes active. Health goes to /x30/health as
`navigation/odometry`.
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster

from .geometry import fill_pose, fill_transform, invert, matrix_to_rpy, pose_msg_to_matrix, rpy_to_matrix
from .odom_logic import ComplementaryFusion, DisagreementMonitor, JumpConfig, JumpDetector

SOURCES = ("gt", "leg", "lidar", "fused")


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class OdomSelector(Node):
    def __init__(self):
        super().__init__("odom_selector")
        d = self.declare_parameter
        d("odom_source", "fused")
        d("leg_topic", "/odom")
        d("lidar_topic", "/odometry/lidar")
        d("gt_topic", "/gt/pose")
        d("active_topic", "/odometry/active")
        d("odom_frame", "odom")
        d("base_frame", "base_link")
        d("publish_tf", True)
        d("max_publish_rate", 50.0)
        d("fusion.k_pos", 0.35)
        d("fusion.k_yaw", 0.35)
        d("fusion.gate_pos", 1.0)
        d("fusion.gate_yaw", 0.8)
        d("watchdog.max_speed", 3.0)
        d("watchdog.max_yaw_rate", 6.0)
        d("watchdog.max_step", 0.8)
        d("watchdog.stale_after", 1.0)
        d("watchdog.hold", 5.0)
        d("watchdog.window", 6.0)
        d("watchdog.min_travel", 0.8)
        d("watchdog.max_ratio", 1.6)
        d("watchdog.max_yaw_diff", 0.6)
        d("auto_fallback", False)            # when the active source is stale or jumping and lidar is healthy, switch to lidar
        g = self.get_parameter
        self.source = g("odom_source").value
        if self.source not in SOURCES:
            raise ValueError(f"odom_source must be one of {SOURCES}, got {self.source}")
        self.odom_frame, self.base_frame = g("odom_frame").value, g("base_frame").value
        self.publish_tf = bool(g("publish_tf").value)
        self.min_period = 1.0 / float(g("max_publish_rate").value)
        self.auto_fallback = bool(g("auto_fallback").value)
        cfg = JumpConfig(**{k: float(g("watchdog." + k).value) for k in ("max_speed", "max_yaw_rate", "max_step", "stale_after", "hold")})
        self.det = {s: JumpDetector(cfg) for s in SOURCES}
        self.disagree = DisagreementMonitor(*(float(g("watchdog." + k).value) for k in ("window", "min_travel", "max_ratio", "max_yaw_diff")))
        self.fusion = ComplementaryFusion(*(float(g("fusion." + k).value) for k in ("k_pos", "k_yaw", "gate_pos", "gate_yaw")))

        self.offset: dict[str, np.ndarray | None] = {s: None for s in SOURCES}   # source frame -> output frame
        self.last_out: np.ndarray | None = None
        self.last_pub_t = -1.0
        self.last_source_t = {s: 0.0 for s in SOURCES}
        self.disagreeing = False
        self.switches: list[str] = []

        self.pub = self.create_publisher(Odometry, g("active_topic").value, 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/x30/health", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, g("leg_topic").value, self.on_leg, qos_profile_sensor_data)
        self.create_subscription(Odometry, g("lidar_topic").value, self.on_lidar, 10)
        self.create_subscription(PoseStamped, g("gt_topic").value, self.on_gt, 10)
        self.add_on_set_parameters_callback(self.on_param)
        self.create_timer(1.0, self.health)
        self.get_logger().info(f"odom_selector: source={self.source}, auto_fallback={self.auto_fallback}")

    # ---- source handlers -------------------------------------------------------------------------------------
    def on_param(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "odom_source":
                if p.value not in SOURCES:
                    return SetParametersResult(successful=False, reason=f"odom_source must be one of {SOURCES}")
                self.set_source(p.value, "parameter")
        return SetParametersResult(successful=True)

    def set_source(self, source: str, why: str) -> None:
        if source != self.source:
            self.get_logger().warn(f"odometry source {self.source} -> {source} ({why})")
            self.switches.append(f"{self.source}->{source}: {why}")
            self.source = source
            self.offset[source] = None  # re-base on the last output when the first pose of the new source arrives

    def on_leg(self, msg: Odometry) -> None:
        t = stamp_to_sec(msg.header.stamp)
        pose = pose_msg_to_matrix(msg.pose.pose)
        self.last_source_t["leg"] = t
        self.det["leg"].update(t, pose, time.monotonic())
        self.disagree.add("a", t, pose)
        v = msg.twist.twist
        fused = self.fusion.on_leg_velocity(t, v.linear.x, v.linear.y, v.angular.z)
        if self.source == "leg":
            self.emit("leg", msg.header.stamp, pose, msg.twist.twist)
        elif self.source == "fused" and fused is not None:
            self.emit("fused", msg.header.stamp, fused.copy(), msg.twist.twist)

    def on_lidar(self, msg: Odometry) -> None:
        t = stamp_to_sec(msg.header.stamp)
        pose = pose_msg_to_matrix(msg.pose.pose)
        self.last_source_t["lidar"] = t
        self.det["lidar"].update(t, pose, time.monotonic())
        self.disagree.add("b", t, pose)
        self.disagreeing = self.disagree.check()
        fused = self.fusion.on_lidar_pose(pose)
        if self.source == "lidar":
            self.emit("lidar", msg.header.stamp, pose, msg.twist.twist)
        elif self.source == "fused" and self.det["leg"].stale(time.monotonic()):
            self.emit("fused", msg.header.stamp, fused.copy(), msg.twist.twist)  # legs silent: lidar poses drive the output

    def on_gt(self, msg: PoseStamped) -> None:
        t = stamp_to_sec(msg.header.stamp)
        pose = pose_msg_to_matrix(msg.pose)
        self.last_source_t["gt"] = t
        self.det["gt"].update(t, pose, time.monotonic())
        if self.source == "gt":
            self.emit("gt", msg.header.stamp, pose, None)

    # ---- output ------------------------------------------------------------------------------------------------
    def emit(self, source: str, stamp, pose: np.ndarray, twist) -> None:
        t = stamp_to_sec(stamp)
        if t - self.last_pub_t < self.min_period and t >= self.last_pub_t:
            return
        if self.offset[source] is None:
            self.offset[source] = self.rebase(source, pose)
        out = self.offset[source] @ pose
        self.last_out, self.last_pub_t = out, t
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        fill_pose(o.pose.pose, out)
        if twist is not None:
            o.twist.twist = twist
        self.pub.publish(o)
        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            fill_transform(tf.transform, out)
            self.tf_broadcaster.sendTransform(tf)

    def rebase(self, source: str, pose: np.ndarray) -> np.ndarray:
        """Output = offset @ source pose. Continuous with the previous output; at start, a planar re-base."""
        if self.last_out is not None:
            return self.last_out @ invert(pose)
        if source in ("gt", "leg"):  # start at the origin, keep the source's own roll/pitch/height
            roll, pitch, yaw = matrix_to_rpy(pose[:3, :3])
            planar = np.eye(4)
            planar[:3, :3] = rpy_to_matrix(0.0, 0.0, yaw)
            planar[:2, 3] = pose[:2, 3]
            return invert(planar)
        return np.eye(4)

    # ---- health ------------------------------------------------------------------------------------------------
    def health(self) -> None:
        wall = time.monotonic()
        stale = {s: self.det[s].stale(wall) for s in SOURCES}
        jumped = {s: self.det[s].flagged(self.last_source_t[s]) for s in SOURCES}
        problems = []
        if self.source != "fused" and stale[self.source]:
            problems.append(f"{self.source} odometry stale")
        if self.source == "fused" and stale["lidar"] and stale["leg"]:
            problems.append("fused: both leg and lidar stale")
        problems += [f"{s} odometry jump" for s in SOURCES if jumped[s] and s in ("leg", "lidar", "gt")]
        if self.disagreeing:
            problems.append(f"leg and lidar disagree (path ratio {self.disagree.last_ratio})")
        st = DiagnosticStatus()
        st.name = "navigation/odometry"
        st.hardware_id = "x30_navigation"
        st.level = DiagnosticStatus.OK if not problems else DiagnosticStatus.WARN
        if self.source != "fused" and (stale[self.source] or jumped[self.source]):
            st.level = DiagnosticStatus.ERROR
        st.message = "ok" if not problems else "; ".join(problems)
        st.values = [KeyValue(key="source", value=self.source),
                     KeyValue(key="jumps_leg", value=str(self.det["leg"].jumps)),
                     KeyValue(key="jumps_lidar", value=str(self.det["lidar"].jumps)),
                     KeyValue(key="rejected_lidar_updates", value=str(self.fusion.rejected_updates)),
                     KeyValue(key="disagree", value=str(self.disagreeing)),
                     KeyValue(key="switches", value=" | ".join(self.switches[-3:]))]
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [st]
        self.diag_pub.publish(arr)
        if (self.auto_fallback and self.source in ("leg", "gt") and not stale["lidar"] and not jumped["lidar"]
                and (stale[self.source] or jumped[self.source])):
            self.set_source("lidar", f"{self.source} {'stale' if stale[self.source] else 'jump'}")


def main(args=None):
    rclpy.init(args=args)
    node = OdomSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
