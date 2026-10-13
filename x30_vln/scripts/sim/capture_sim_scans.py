#!/usr/bin/env python3
"""Capture N merged scans from /lidar_points of a running simulator into an .npz (run inside the sim container).
  python3 capture_sim_scans.py --n 30 --out /x30/results/sim_lidar_compare/sim_scans.npz
The file holds `scans` (N, P) with the contract dtype, `header_stamps` (N,) in seconds and `frame_id`.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

DTYPE = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'tag', 'line', 'timestamp'],
                  'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'], 'offsets': [0, 4, 8, 12, 16, 17, 18], 'itemsize': 26})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--out', required=True)
    ap.add_argument('--topic', default='/lidar_points')
    ap.add_argument('--skip', type=int, default=5, help='scans to drop first (warm-up)')
    a = ap.parse_args()
    rclpy.init()
    node = Node('capture_sim_scans')
    scans, stamps = [], []
    seen = [0]

    def cb(msg):
        seen[0] += 1
        if seen[0] > a.skip:
            scans.append(np.frombuffer(bytes(msg.data), dtype=DTYPE).copy())
            stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    node.create_subscription(PointCloud2, a.topic, cb, QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
    end = time.monotonic() + 300
    while len(scans) < a.n and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    n = min(len(s) for s in scans)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, scans=np.stack([s[:n] for s in scans]), header_stamps=np.array(stamps), frame_id='lidar_link')
    print(f'saved {len(scans)} scans of {n} points to {a.out}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
