#!/usr/bin/env python3
"""End-to-end check of the perception chain against the synthetic stream (or any source that publishes /synthetic/ground_truth).

Needs running: perception server, `ros2 run x30_perception synthetic_stream`, `ros2 launch x30_perception perception.launch.py`.
Collects /perception/detections for --duration seconds, compares with the ground-truth billboards and prints a table.
Exit code 0 when every ground-truth object was detected at least once with a lifted position within --tol metres of its plane
and inside its rectangle, and a diagnostics message was received; non-zero otherwise.

  ros2 run x30_perception ... or:  python3 smoke_test.py --duration 25 --json /results/perception/smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from x30_msgs.msg import Detection3DArray
from x30_msgs.srv import Json


class Collector(Node):
    def __init__(self):
        super().__init__('perception_smoke_test')
        self.gt = None
        self.arrays: list[Detection3DArray] = []
        self.diag: list[dict] = []
        self.create_subscription(
            String,
            '/synthetic/ground_truth',
            self._gt,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_subscription(Detection3DArray, '/perception/detections', self.arrays.append, 10)
        self.create_subscription(DiagnosticArray, '/perception/diagnostics', self._diag, 10)
        self.vocab = self.create_client(Json, '/perception/vocabulary')

    def _gt(self, m):
        self.gt = json.loads(m.data)

    def _diag(self, m):
        self.diag.append({kv.key: kv.value for kv in m.status[0].values})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=25.0)
    ap.add_argument('--tol', type=float, default=0.35, help='allowed distance of the lifted point from the billboard plane, m')
    ap.add_argument('--min-frames', type=int, default=5)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    rclpy.init()
    node = Collector()
    # per-task phrase through the vocabulary service ('bus' is not in the indoor base vocabulary)
    vocab_ok = False
    if node.vocab.wait_for_service(timeout_sec=10.0):
        fut = node.vocab.call_async(Json.Request(request=json.dumps({'op': 'add', 'labels': ['bus'], 'ttl_s': 600})))
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        vocab_ok = bool(fut.done() and fut.result().ok and '"bus"' in fut.result().response)
    t_end = time.time() + 3.0  # let the new phrase reach the server; discard these frames
    while time.time() < t_end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    node.arrays.clear()
    t_end = time.time() + a.duration
    while time.time() < t_end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    gt = node.gt
    arrays = node.arrays
    print(f'frames received: {len(arrays)}; diagnostics: {len(node.diag)}; ground truth: {"yes" if gt else "NO"}')
    if not gt or not arrays:
        print('FAIL: no detections or no ground truth')
        sys.exit(2)
    objs = gt['objects']
    rows = {o['id']: [] for o in objs}
    n_det = n_bearing = n_unassigned = 0
    other_labels: dict = {}
    frames = set()
    th = {o['id']: [] for o in objs}
    for arr in arrays:
        frames.add(arr.header.frame_id)
        for d in arr.detections:
            n_det += 1
            if d.bearing_only:
                n_bearing += 1
                continue
            c = np.array([d.centroid.x, d.centroid.y, d.centroid.z])

            # associate geometrically: the billboard whose rectangle contains the point (+0.15 m) and whose plane is nearest
            def plane_dist(q):
                ctr = np.array(q['center'])
                inside = abs(c[1] - ctr[1]) <= q['width'] / 2 + 0.15 and abs(c[2] - ctr[2]) <= q['height'] / 2 + 0.15
                return abs(float(np.dot(c - ctr, q['normal']))) if inside else 1e9

            o = min(objs, key=plane_dist)
            if plane_dist(o) > 1.0:
                n_unassigned += 1
                continue
            ctr = np.array(o['center'])
            if d.label != o['label']:
                other_labels[o['id']] = other_labels.get(o['id'], 0) + 1  # e.g. the passengers in the bus photo
                continue
            rows[o['id']].append((plane_dist(o), float(np.linalg.norm(c - ctr)), True, d.point_count, d.depth_source))
            if np.isfinite(d.thermal_mean_c):
                th[o['id']].append(d.thermal_mean_c)
    ok = True
    report = {
        'frames': len(arrays),
        'frame_ids': sorted(frames),
        'detections': n_det,
        'bearing_only': n_bearing,
        'unassigned_detections': n_unassigned,
        'other_label_detections_on_object': other_labels,
        'vocabulary_service_ok': vocab_ok,
        'objects': [],
    }
    print(
        f'frame ids: {sorted(frames)}; detections {n_det}, bearing_only {n_bearing}, not on any object {n_unassigned}, '
        f'other-label hits on objects {other_labels}; vocabulary service ok: {vocab_ok}'
    )
    print(
        f'{"id":>2} {"label":<8} {"GT x,y,z":<20} {"n":>3} {"plane err med/max (m)":<22} {"centre err med (m)":<19} '
        f'{"inside":<7} {"points":<7} thermal GT/meas'
    )
    for o in objs:
        r = rows[o['id']]
        if not r:
            print(f'{o["id"]:>2} {o["label"]:<8} {str(np.round(o["center"], 1).tolist()):<20}   0 NOT DETECTED')
            report['objects'].append({'id': o['id'], 'label': o['label'], 'detected': False})
            ok = False
            continue
        pe, ce = np.array([x[0] for x in r]), np.array([x[1] for x in r])
        inside = np.mean([x[2] for x in r])
        pts = int(np.median([x[3] for x in r]))
        tm = f'{o["temp_c"]:.0f}/{np.median(th[o["id"]]):.1f}' if th[o['id']] else f'{o["temp_c"]:.0f}/-'
        print(
            f'{o["id"]:>2} {o["label"]:<8} {str(np.round(o["center"], 1).tolist()):<20} {len(r):>3} '
            f'{np.median(pe):.3f} / {pe.max():.3f}        {np.median(ce):.3f}              {inside:.2f}    {pts:<7} {tm}'
        )
        report['objects'].append(
            {
                'id': o['id'],
                'label': o['label'],
                'detected': True,
                'n': len(r),
                'plane_err_median_m': float(np.median(pe)),
                'plane_err_max_m': float(pe.max()),
                'centre_err_median_m': float(np.median(ce)),
                'inside_fraction': float(inside),
                'points_median': pts,
                'thermal_median_c': float(np.median(th[o['id']])) if th[o['id']] else None,
            }
        )
        if np.median(pe) > a.tol:
            ok = False
    if node.diag:
        last = node.diag[-1]
        report['diagnostics_last'] = last
        print(
            'diagnostics (last):',
            {
                k: last[k]
                for k in (
                    'frame_id',
                    'server_rtt_ms',
                    'server_infer_ms',
                    'lift_ms',
                    'process_ms',
                    'image_to_publish_ms',
                    'scans_used',
                    'motion_compensated',
                )
                if k in last
            },
        )
    else:
        print('FAIL: no diagnostics received')
        ok = False
    if len(arrays) < a.min_frames:
        print(f'FAIL: only {len(arrays)} frames in {a.duration} s')
        ok = False
    if not vocab_ok:
        print('FAIL: /perception/vocabulary add did not answer ok')
        ok = False
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1)
    print('PASS' if ok else 'FAIL')
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
