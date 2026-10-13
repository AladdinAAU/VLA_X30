#!/usr/bin/env python3
"""Run inside the sim container: for each pose of poses.json teleport the robot, save the RGB frame and the robot ground-truth pose;
also one lidar cloud (xyz) and the GT objects.   sim_capture.py <out_dir>   (needs ROS_DOMAIN_ID/GZ_PARTITION of the running sim)"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from x30_msgs.srv import Json


class Cap(Node):
    def __init__(self):
        super().__init__("worlds_capture")
        q = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.img = None
        self.pose = None
        self.cloud = None
        self.create_subscription(Image, "/camera/color/image_raw", lambda m: setattr(self, "img", m), q)
        self.create_subscription(PoseStamped, "/gt/pose", lambda m: setattr(self, "pose", m), q)
        self.create_subscription(PointCloud2, "/lidar_points", lambda m: setattr(self, "cloud", m), q)
        self.cli = self.create_client(Json, "/sim/set_robot_pose")

    def spin_for(self, s):
        t = time.time()
        while time.time() - t < s:
            rclpy.spin_once(self, timeout_sec=0.05)

    def teleport(self, p):
        self.cli.wait_for_service(10)
        f = self.cli.call_async(Json.Request(request=json.dumps(p)))
        while not f.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        return f.result().response


def main():
    out = Path(sys.argv[1])
    poses = json.loads((out / "poses.json").read_text())
    rclpy.init()
    n = Cap()
    n.spin_for(3)
    rec = []
    for i, p in enumerate(poses):
        n.teleport(p)
        n.img = n.pose = None
        n.spin_for(2.5)
        n.img = None
        while n.img is None or n.pose is None:
            rclpy.spin_once(n, timeout_sec=0.1)
        m = n.img
        im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        cv2.imwrite(str(out / f"frame_{i:02d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        ps = n.pose.pose
        yaw = float(np.arctan2(2 * (ps.orientation.w * ps.orientation.z + ps.orientation.x * ps.orientation.y),
                               1 - 2 * (ps.orientation.y ** 2 + ps.orientation.z ** 2)))
        rec.append({"frame": f"frame_{i:02d}.png", "x": ps.position.x, "y": ps.position.y, "z": ps.position.z, "yaw": yaw})
        if i == 0 and n.cloud is not None:
            c = n.cloud
            raw = np.frombuffer(bytes(c.data), np.uint8).reshape(-1, c.point_step)
            xyz = raw[:, :12].copy().view(np.float32).reshape(-1, 3)
            np.save(out / "lidar_xyz.npy", xyz[np.abs(xyz).sum(1) > 0])
    (out / "captured.json").write_text(json.dumps(rec))
    print("captured", len(rec))


main()
