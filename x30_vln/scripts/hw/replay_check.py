#!/usr/bin/env python3
"""Checks the replay profile (run while `ros2 launch x30_hw replay.launch.py` is up, consumers use_sim_time)."""
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

EXPECT = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7), ("tag", 16, 2), ("line", 17, 2), ("timestamp", 18, 8)]


class N(Node):
    def __init__(self):
        super().__init__("replay_check", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.lidar = []
        self.first = None
        self.odom = 0
        self.imu = 0
        self.stub = None
        self.clock = 0
        self.create_subscription(PointCloud2, "/lidar_points", self.on_l, 20)  # reliable, like the bridge
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", self.odom + 1), 10)
        self.create_subscription(Imu, "/imu/data", lambda m: setattr(self, "imu", self.imu + 1), 10)
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "clock", self.clock + 1), qos_profile_sensor_data)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        self.create_subscription(Bool, "/x30/replay_stub", lambda m: setattr(self, "stub", m.data),
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)

    def on_l(self, m):
        if self.first is None:
            self.first = m
        self.lidar.append((time.monotonic(), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9))


def main():
    rclpy.init()
    n = N()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 14:
        rclpy.spin_once(n, timeout_sec=0.1)
    fails = []

    def chk(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    m = n.first
    chk("lidar received", m is not None, f"{len(n.lidar)} scans")
    if m is not None:
        wall = (n.lidar[-1][0] - n.lidar[0][0])
        sim = (n.lidar[-1][1] - n.lidar[0][1])
        chk("lidar rate 10 Hz in sim time", abs((len(n.lidar) - 1) / sim - 10.0) < 1.0 if sim > 0 else False, f"{(len(n.lidar) - 1) / sim:.2f} Hz sim, {(len(n.lidar) - 1) / wall:.2f} Hz wall")
        chk("frame_id lidar_link", m.header.frame_id == "lidar_link", m.header.frame_id)
        got = [(f.name, f.offset, f.datatype) for f in m.fields]
        chk("fields x,y,z,intensity,tag,line,timestamp with contract offsets", got == EXPECT, got)
        chk("point_step 26, is_dense", m.point_step == 26, f"point_step={m.point_step} dense={m.is_dense}")
        chk("about 80k points", 60000 < m.width * m.height < 100000, m.width * m.height)
    chk("/clock published", n.clock > 50, n.clock)
    chk("stub /odom ~50 Hz", n.odom > 300, n.odom)
    chk("stub /imu/data ~100 Hz", n.imu > 600, n.imu)
    chk("/x30/replay_stub flag true", n.stub is True)
    ok = n.buf.can_transform("base_link", "lidar_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=2.0))
    chk("TF base_link -> lidar_link", ok)
    sys.exit(1 if fails else 0)


main()
