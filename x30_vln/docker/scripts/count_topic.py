#!/usr/bin/env python3
"""Count PointCloud2 messages on a topic and report rate, inter-arrival gaps and the shape of the first message.

  count_topic.py /lidar_points --seconds 40 [--idle-exit 5] [--qos reliable|best_effort]
Exits when --seconds elapsed or no message arrived for --idle-exit seconds after the first one (end of a single bag pass).
"""
import argparse
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--idle-exit", type=float, default=0.0)
    ap.add_argument("--qos", default="reliable")
    a = ap.parse_args()
    rclpy.init()
    node = Node("count_topic")
    arrivals, sizes, first = [], [], []

    def cb(raw):
        arrivals.append(time.time())
        sizes.append(len(raw))
        if not first:
            first.append(deserialize_message(raw, PointCloud2))

    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE if a.qos == "reliable" else ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=10)
    node.create_subscription(PointCloud2, a.topic, cb, qos, raw=True)
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
        if a.idle_exit and arrivals and time.time() - arrivals[-1] > a.idle_exit:
            break
    n = len(arrivals)
    print(f"topic {a.topic}: {n} messages")
    if n > 1:
        gaps = [b - x for x, b in zip(arrivals, arrivals[1:])]
        span = arrivals[-1] - arrivals[0]
        print(f"  mean rate {((n - 1) / span):.2f} Hz over {span:.1f} s; gap mean {statistics.mean(gaps)*1e3:.1f} ms, "
              f"p95 {sorted(gaps)[int(len(gaps)*0.95)-1]*1e3:.1f} ms, max {max(gaps)*1e3:.1f} ms; gaps > 150 ms: {sum(g > 0.15 for g in gaps)}")
        print(f"  CDR size per message: mean {statistics.mean(sizes)/1e6:.2f} MB")
    if first:
        m = first[0]
        print(f"  first message: frame_id={m.header.frame_id} width={m.width} height={m.height} point_step={m.point_step} "
              f"fields={[f.name for f in m.fields]} stamp={m.header.stamp.sec}")
    rclpy.shutdown()
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
