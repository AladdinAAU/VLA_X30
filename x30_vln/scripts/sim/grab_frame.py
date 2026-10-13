#!/usr/bin/env python3
"""Save one frame of each given image topic as PNG (rgb8, mono8/16, 32FC1 scaled to 0..255). Run in the sim container.
  python3 grab_frame.py OUT_DIR /camera/color/image_raw /thermal/image_raw /thermal/temperature
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def to_png(msg: Image) -> np.ndarray:
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    if msg.encoding == 'mono8':
        return buf.reshape(msg.height, msg.width)
    if msg.encoding == '32FC1':
        f = np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width)
        fin = f[np.isfinite(f)]
        lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
        print(f'  range {lo:.2f} .. {hi:.2f}')
        return np.clip((np.nan_to_num(f, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    raise ValueError(msg.encoding)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node('grab_frame')
    got = {}
    for t in sys.argv[2:]:
        node.create_subscription(Image, t, lambda m, t=t: got.setdefault(t, m), QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE))
    end = time.monotonic() + 30
    while len(got) < len(sys.argv[2:]) and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    for t, m in got.items():
        name = t.strip('/').replace('/', '_') + '.png'
        print(name, m.encoding)
        cv2.imwrite(str(out / name), to_png(m))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
