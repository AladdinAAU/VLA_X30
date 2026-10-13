"""cloud_to_scan: /nav/cloud_level -> /scan (nearest obstacle per azimuth bin, in the levelled `base_footprint` frame).

The height slice and the ground removal were already applied by cloud_leveler (obstacle flag), so a pitched body
does not put the floor into the scan, and ceilings above the obstacle band are not seen.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2

from .cloud_leveler import OBSTACLE
from .pointcloud import cloud_to_struct


class CloudToScan(Node):
    def __init__(self):
        super().__init__("cloud_to_scan")
        d = self.declare_parameter
        d("input_topic", "/nav/cloud_level")
        d("output_topic", "/scan")
        d("num_beams", 720)
        d("range_min", 0.10)
        d("range_max", 12.0)
        d("scan_time", 0.1)
        g = self.get_parameter
        self.n = int(g("num_beams").value)
        self.rmin = float(g("range_min").value)
        self.rmax = float(g("range_max").value)
        self.scan_time = float(g("scan_time").value)
        self.pub = self.create_publisher(LaserScan, g("output_topic").value, 5)
        self.create_subscription(PointCloud2, g("input_topic").value, self.on_cloud, 5)

    def on_cloud(self, msg: PointCloud2) -> None:
        pts = cloud_to_struct(msg)
        obs = pts[pts["flags"] == OBSTACLE]
        x, y = obs["x"].astype(np.float64), obs["y"].astype(np.float64)
        r = np.hypot(x, y)
        ok = (r >= self.rmin) & (r <= self.rmax)
        ranges = np.full(self.n, np.inf, dtype=np.float32)
        if ok.any():
            az = np.arctan2(y[ok], x[ok])
            bins = np.minimum(((az + math.pi) / (2.0 * math.pi) * self.n).astype(np.int64), self.n - 1)
            np.minimum.at(ranges, bins, r[ok].astype(np.float32))
        scan = LaserScan()
        scan.header = msg.header
        inc = 2.0 * math.pi / self.n
        scan.angle_min = -math.pi + 0.5 * inc   # beam i is centred in its bin
        scan.angle_max = scan.angle_min + inc * (self.n - 1)
        scan.angle_increment = inc
        scan.scan_time = self.scan_time
        scan.time_increment = self.scan_time / self.n
        scan.range_min = self.rmin
        scan.range_max = self.rmax
        scan.ranges = ranges.tolist()
        self.pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = CloudToScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
