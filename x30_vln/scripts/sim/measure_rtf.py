#!/usr/bin/env python3
"""Measure real-time factor, topic rates and process CPU/RAM of a running simulator (run inside the sim container).

  python3 measure_rtf.py --seconds 120 --interval 10 [--csv out.csv]
RTF = simulated seconds per wall second from /clock. Prints one line per interval and a summary; RSS of gz sim (ruby) is tracked so
memory growth (MB/min) can be read from the CSV.
"""
import argparse
import csv
import subprocess
import time

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2

TOPICS = {'/lidar_points': PointCloud2, '/camera/color/image_raw': Image, '/thermal/image_raw': Image}


def proc_stats():
    """(cpu_percent_sum_of_all_processes, rss_mb_by_group) from ps."""
    out = subprocess.run(['ps', '-eo', 'pcpu,rss,args', '--no-headers'], capture_output=True, text=True).stdout
    groups = {'gz': [0.0, 0.0], 'ros': [0.0, 0.0]}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        cpu, rss, args = float(parts[0]), float(parts[1]) / 1024.0, parts[2]
        key = 'gz' if ('gz sim' in args or 'ruby' in args) else 'ros' if ('/opt/ros' in args or 'x30_sim' in args or 'python3' in args) else None
        if key and 'measure_rtf' not in args:
            groups[key][0] += cpu
            groups[key][1] += rss
    return groups


class Meter(Node):
    def __init__(self):
        super().__init__('measure_rtf')
        self.sim = 0.0
        self.counts = {t: 0 for t in TOPICS}
        self.create_subscription(Clock, '/clock', lambda m: setattr(self, 'sim', m.clock.sec + m.clock.nanosec * 1e-9), 10)
        for t, typ in TOPICS.items():
            self.create_subscription(typ, t, lambda _m, t=t: self.counts.__setitem__(t, self.counts[t] + 1), 5, raw=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--interval', type=float, default=10)
    ap.add_argument('--csv', default='')
    a = ap.parse_args()
    rclpy.init()
    node = Meter()
    while node.sim == 0.0:                       # wait for the first /clock so the first interval is not inflated
        rclpy.spin_once(node, timeout_sec=0.1)
    rows = []
    t0 = time.monotonic()
    last_wall, last_sim, last_counts = t0, node.sim, dict(node.counts)
    while time.monotonic() - t0 < a.seconds:
        end = time.monotonic() + a.interval
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        wall = time.monotonic()
        dt = wall - last_wall
        g = proc_stats()
        row = {'t': round(wall - t0, 1), 'rtf': round((node.sim - last_sim) / dt, 3), 'sim_time': round(node.sim, 1),
               'gz_cpu_pct': g['gz'][0], 'gz_rss_mb': round(g['gz'][1]), 'ros_cpu_pct': g['ros'][0], 'ros_rss_mb': round(g['ros'][1])}
        for t in TOPICS:
            row['hz' + t.replace('/', '_')] = round((node.counts[t] - last_counts[t]) / dt, 2)
        rows.append(row)
        print(row, flush=True)
        last_wall, last_sim, last_counts = wall, node.sim, dict(node.counts)
    if rows:
        rtfs = [r['rtf'] for r in rows]
        print(f'SUMMARY rtf mean {sum(rtfs) / len(rtfs):.3f} min {min(rtfs):.3f} max {max(rtfs):.3f}; '
              f'gz rss {rows[0]["gz_rss_mb"]} -> {rows[-1]["gz_rss_mb"]} MB over {rows[-1]["t"]} s')
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
