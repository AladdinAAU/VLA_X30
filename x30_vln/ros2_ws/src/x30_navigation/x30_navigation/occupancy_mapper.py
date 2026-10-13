"""occupancy_mapper: integrate the levelled lidar cloud with the odometry into a 2D /map (log-odds, ray cast free space).

Pose comes from TF (map -> base_footprint at the scan stamp). There is no scan-to-map correction and no loop
closure: `map -> odom` is a fixed identity published here, so the map is as good as the odometry. Use
`slam: slam_toolbox` when loop closure is needed.
"""
from __future__ import annotations

import array
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from .cloud_leveler import BODY, GROUND, OBSTACLE
from .geometry import transform_msg_to_matrix
from .grid import GridParams, LogOddsGrid
from .pointcloud import cloud_to_struct


class OccupancyMapper(Node):
    def __init__(self):
        super().__init__("occupancy_mapper")
        d = self.declare_parameter
        d("input_topic", "/nav/cloud_level")
        d("map_topic", "/map")
        d("map_frame", "map")
        d("odom_frame", "odom")
        d("publish_map_odom_tf", True)
        d("publish_period", 1.0)
        d("max_obstacle_range", 10.0)
        d("tf_timeout", 0.2)
        for k, v in vars(GridParams()).items():
            d("grid." + k, v)
        g = self.get_parameter
        self.map_frame = g("map_frame").value
        self.odom_frame = g("odom_frame").value
        self.max_range = float(g("max_obstacle_range").value)
        self.tf_timeout = float(g("tf_timeout").value)
        gp = GridParams()
        for k, v in vars(gp).items():
            setattr(gp, k, type(v)(g("grid." + k).value))
        self.grid = LogOddsGrid(gp)
        self.dirty = False
        self.scans = 0
        self.pending: deque = deque(maxlen=20)   # clouds waiting for their TF (odometry is computed after the scan arrives)
        self.times: list[float] = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        if g("publish_map_odom_tf").value:
            self.static_tf = StaticTransformBroadcaster(self)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.map_frame
            t.child_frame_id = self.odom_frame
            t.transform.rotation.w = 1.0
            self.static_tf.sendTransform(t)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(OccupancyGrid, g("map_topic").value, qos)
        self.create_subscription(PointCloud2, g("input_topic").value, lambda m: self.pending.append((m, time.monotonic())), 5)
        self.create_timer(0.02, self.process_pending)
        self.create_timer(float(g("publish_period").value), self.publish_map)
        self.create_service(Empty, "~/reset", self.on_reset)

    def on_reset(self, _req, resp):
        self.grid = LogOddsGrid(self.grid.p)
        self.dirty = False
        self.get_logger().info("map reset")
        return resp

    def process_pending(self) -> None:
        while self.pending:
            msg, queued = self.pending[0]
            try:
                tf = self.tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, msg.header.stamp)
            except Exception as exc:
                if time.monotonic() - queued > 1.0:
                    self.pending.popleft()   # its TF never came: drop the scan
                    self.get_logger().warn(f"no TF {self.map_frame} <- {msg.header.frame_id} at scan time: {exc}",
                                           throttle_duration_sec=5.0)
                return
            self.pending.popleft()
            self.integrate(msg, tf)

    def integrate(self, msg: PointCloud2, tf) -> None:
        t0 = time.perf_counter()
        t_map = transform_msg_to_matrix(tf.transform)
        pts = cloud_to_struct(msg)
        flags = pts["flags"]
        xy = np.column_stack((pts["x"], pts["y"])).astype(np.float64)
        r = np.hypot(xy[:, 0], xy[:, 1])
        hits = xy[(flags == OBSTACLE) & (r <= self.max_range)]
        ground = xy[flags == GROUND]
        rot, trans = t_map[:2, :2], t_map[:2, 3]
        body = xy[(flags == BODY) & (r <= self.max_range)]
        self.grid.integrate(trans.copy(), hits @ rot.T + trans, ground @ rot.T + trans, body @ rot.T + trans)
        self.dirty = True
        self.scans += 1
        self.times.append(time.perf_counter() - t0)
        if len(self.times) >= 100:
            a = np.array(self.times) * 1e3
            self.get_logger().info(f"occupancy_mapper: {a.mean():.1f} ms mean, {a.max():.1f} ms max per scan, grid {self.grid.shape}")
            self.times.clear()

    def publish_map(self) -> None:
        if not self.dirty or self.grid.log is None:
            return
        self.dirty = False
        h, w = self.grid.shape
        m = OccupancyGrid()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.info.resolution = self.grid.p.resolution
        m.info.width, m.info.height = w, h
        m.info.origin.position.x, m.info.origin.position.y = (float(v) for v in self.grid.origin)
        m.info.origin.orientation.w = 1.0
        m.data = array.array("b", self.grid.to_int8().tobytes())
        self.map_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
