#!/usr/bin/env python3
"""Integration check of the navigation chain (odometry, leveling, scan, costmap cloud, map, TF, health).

Start the stack first (e.g. replay: navigation.launch.py profile:=replay nav2:=false static_lidar_tf:=true, plus
`ros2 bag play`), then run this script. Exit code 0 if everything arrived within --timeout seconds, 1 otherwise.
"""
import argparse
import sys
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from tf2_ros import Buffer, TransformListener


class Check(Node):
    def __init__(self):
        super().__init__("check_stack")
        self.got = {}
        qos_map = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, "/odometry/lidar", lambda m: self.got.setdefault("odometry/lidar", m), 10)
        self.create_subscription(Odometry, "/odometry/active", lambda m: self.got.__setitem__("odometry/active", m), 10)
        self.create_subscription(LaserScan, "/scan", lambda m: self.got.__setitem__("scan", m), 10)
        self.create_subscription(PointCloud2, "/costmap_cloud", lambda m: self.got.__setitem__("costmap_cloud", m), 10)
        self.create_subscription(OccupancyGrid, "/map", lambda m: self.got.__setitem__("map", m), qos_map)
        self.create_subscription(DiagnosticArray, "/x30/health", lambda m: self.got.__setitem__("health", m), 10)
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()
    rclpy.init()
    node = Check()
    t0 = time.monotonic()
    need = {"odometry/lidar", "odometry/active", "scan", "costmap_cloud", "map", "health"}
    while time.monotonic() - t0 < args.timeout and not need <= node.got.keys():
        rclpy.spin_once(node, timeout_sec=0.2)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    ok = True
    missing = need - node.got.keys()
    if missing:
        print("FAIL missing topics:", sorted(missing))
        ok = False
    for a, b in (("map", "odom"), ("odom", "base_link"), ("base_link", "base_footprint"), ("map", "base_footprint")):
        if not node.buf.can_transform(a, b, rclpy.time.Time()):
            print(f"FAIL no TF {a} -> {b}")
            ok = False
    if "scan" in node.got:
        r = np.array(node.got["scan"].ranges)
        print(f"scan: {len(r)} beams, {np.isfinite(r).sum()} with return, frame {node.got['scan'].header.frame_id}")
    if "costmap_cloud" in node.got:
        print(f"costmap_cloud: {node.got['costmap_cloud'].width} points, frame {node.got['costmap_cloud'].header.frame_id}")
    if "map" in node.got:
        m = node.got["map"]
        d = np.array(m.data, dtype=np.int8)
        print(f"map: {m.info.width}x{m.info.height} @ {m.info.resolution} m, free {int((d == 0).sum())}, occupied {int((d == 100).sum())}")
        ok &= (d == 0).sum() > 100
    if "odometry/active" in node.got:
        p = node.got["odometry/active"].pose.pose.position
        print(f"odometry/active: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
    if "health" in node.got:
        st = node.got["health"].status[0]
        print(f"health: {st.name} level {ord(st.level) if isinstance(st.level, bytes) else st.level} '{st.message}'")
    print("PASS" if ok else "FAIL")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
