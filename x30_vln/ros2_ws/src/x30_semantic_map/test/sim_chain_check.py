#!/usr/bin/env python3
"""Real chain check: simulator -> perception client -> semantic mapper, compared with /gt/objects.

Run inside the Humble framework container while the simulator (own container, same ROS_DOMAIN_ID), the perception server,
`perception_client` and `semantic_mapper` (both with use_sim_time) and a static TF map->odom (identity) are up.
The script drives the robot with /cmd_vel (walk forward with a heading sweep; no obstacle sensing beyond a
blocked-progress check that turns left), then reads the map through /semantic_map/query and compares it with the ground truth.

Frames: the mapper works in `map` == `odom` (identity); /gt/objects is in `world`. The rigid 2D transform world<-odom is
estimated from /gt/pose and /odom at the start.

Ground truth that counts as reachable: label known to the mapper vocabulary (canonical name equality, or same text after
normalisation), centre within --reach metres of the driven path, no occlusion test (so recall is a lower bound
for what is physically visible). A map object is correct when a same-label ground-truth object lies within --match m.
Perception reports the visible surface, so a bias of about half the object depth is expected.

    python3 sim_chain_check.py --duration 240 --out /out/semantic_map_sim_chain.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from x30_msgs.msg import Detection3DArray, SemanticObjectArray
from x30_msgs.srv import Json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from x30_semantic_map.config import load_config  # noqa: E402
from x30_semantic_map.vocabulary import Vocabulary, normalise  # noqa: E402


CLOUD = np.dtype({'names': ['x', 'y', 'z', 'i'], 'formats': ['<f4'] * 4, 'offsets': [0, 4, 8, 12], 'itemsize': 26})


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Chain(Node):
    def __init__(self):
        super().__init__('semantic_map_sim_chain_check', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.gt = None
        self.gt_pose = None
        self.odom = None
        self.n_det_msgs = 0
        self.n_det = 0
        self.dump = None
        self.objs = None
        self.create_subscription(SemanticObjectArray, '/gt/objects', lambda m: setattr(self, 'gt', m), latched)
        self.create_subscription(PoseStamped, '/gt/pose', lambda m: setattr(self, 'gt_pose', m), 10)
        self.create_subscription(Odometry, '/odom', lambda m: setattr(self, 'odom', m), 10)
        self.create_subscription(Detection3DArray, '/perception/detections', self._on_det, 10)
        self.create_subscription(SemanticObjectArray, '/semantic_map/objects', lambda m: setattr(self, 'objs', m), latched)
        self.clearance = (9.0, 9.0, 9.0)          # front, left, right free distance from the latest scan
        self.create_subscription(PointCloud2, '/lidar_points', self._on_cloud, 1)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.query = self.create_client(Json, '/semantic_map/query')
        self.summary = self.create_client(Json, '/semantic_map/summary')
        self.reset = self.create_client(Json, '/semantic_map/reset')

    def _on_cloud(self, m) -> None:
        a = np.frombuffer(m.data, CLOUD, count=m.width * m.height)
        a = a[a['i'] > 0]                                    # no-return points sit at the sensor origin with intensity 0
        x, y, z = a['x'], a['y'], a['z']
        h = (z > -0.15) & (z < 0.6)                          # body height band (lidar frame is close to base_link)

        def free(sel) -> float:
            d = np.hypot(x[h & sel], y[h & sel])
            d = d[d > 0.7]                                   # the robot's own body returns at 0.3-0.6 m
            return float(np.percentile(d, 5)) if d.size > 20 else 9.0
        ang = np.arctan2(y, x)
        self.clearance = (free((np.abs(ang) < 0.4) & (x > 0.2)), free((ang > 0.4) & (ang < 1.2)), free((ang < -0.4) & (ang > -1.2)))

    def _on_det(self, m) -> None:
        self.n_det_msgs += 1
        self.n_det += len(m.detections)
        if self.dump is not None:
            for d in m.detections:
                self.dump.write(json.dumps({'t': m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, 'label': d.label,
                                            'score': round(d.score, 3), 'xyz': [round(d.centroid.x, 3), round(d.centroid.y, 3),
                                            round(d.centroid.z, 3)], 'bearing_only': d.bearing_only,
                                            'points': d.point_count}) + '\n')

    def spin(self, seconds: float) -> None:
        t_end = time.time() + seconds
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def call(self, cli, req: dict, timeout: float = 20.0) -> dict:
        if not cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('service not available')
        r = Json.Request()
        r.request = json.dumps(req)
        fut = cli.call_async(r)
        t_end = time.time() + timeout
        while not fut.done() and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return json.loads(fut.result().response)

    def drive(self, vx: float, wz: float, sim_s: float, stop: bool = True) -> None:
        t0 = self.get_clock().now()
        while (self.get_clock().now() - t0).nanoseconds * 1e-9 < sim_s:
            m = Twist()
            m.linear.x, m.angular.z = vx, wz
            self.cmd.publish(m)
            self.spin(0.05)
        if stop:
            self.cmd.publish(Twist())

    def world_xy(self) -> np.ndarray:
        p = self.gt_pose.pose.position
        return np.array([p.x, p.y])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=300.0, help='simulated seconds of driving')
    ap.add_argument('--reach', type=float, default=4.0)
    ap.add_argument('--match', type=float, default=1.5)
    ap.add_argument('--out', default='semantic_map_sim_chain.json')
    ap.add_argument('--config', default='')
    ap.add_argument('--dump', default='', help='write every detection (stamp, label, score, centroid, extent, points) as JSON lines')
    args = ap.parse_args()
    rclpy.init()
    n = Chain()
    t_end = time.time() + 60
    while time.time() < t_end and not (n.gt and n.gt_pose and n.odom):
        n.spin(0.2)
    if not (n.gt and n.gt_pose and n.odom):
        print('FAIL: sim topics missing (/gt/objects, /gt/pose, /odom)')
        return 2
    if args.dump:
        n.dump = open(args.dump, 'w')
    vocab = Vocabulary(load_config(args.config) if args.config else load_config())

    # world <- odom (2D rigid)
    n.spin(0.5)
    o = n.odom.pose.pose
    dyaw = yaw_of(n.gt_pose.pose.orientation) - yaw_of(o.orientation)
    c, s = math.cos(dyaw), math.sin(dyaw)
    rot = np.array([[c, -s], [s, c]])
    t_w = n.world_xy() - rot @ np.array([o.position.x, o.position.y])

    def to_world(p) -> np.ndarray:
        return rot @ np.asarray(p[:2], float) + t_w

    # drive: walk forward at 0.3 m/s with a +-25 deg heading sweep (the camera sees every object for several seconds);
    # when the lidar shows an obstacle within 1 m ahead, turn in place towards the freer side until 1.5 m is free
    n.call(n.reset, {})
    path = [n.world_xy().copy()]
    t0 = n.get_clock().now()
    sim_t = lambda: (n.get_clock().now() - t0).nanoseconds * 1e-9  # noqa: E731
    steps = 0
    blocked = 0
    last_print = -10.0
    gt_xy = np.array([[g.position.x, g.position.y] for g in n.gt.objects])
    lo, hi = gt_xy.min(axis=0) - 3.0, gt_xy.max(axis=0) + 3.0
    while sim_t() < args.duration:
        if np.any(n.world_xy() < lo) or np.any(n.world_xy() > hi):
            print('robot left the object bounds, stopping the run', flush=True)
            break
        front, left, right = n.clearance
        if front < 1.0:
            blocked += 1
            sign = 1.0 if left >= right else -1.0
            n.drive(0.0, 0.5 * sign, 0.6, stop=False)
        else:
            n.drive(0.3, 0.45 * math.sin(2 * math.pi * sim_t() / 16.0), 0.5, stop=False)
        steps += 1
        path.append(n.world_xy().copy())
        if sim_t() - last_print >= 20.0:
            last_print = sim_t()
            print(f'sim_t={sim_t():.0f}s pose={n.world_xy().round(2).tolist()} clearance={np.round(n.clearance, 1).tolist()} '
                  f'detections={n.n_det}', flush=True)
    n.cmd.publish(Twist())
    n.spin(8.0)                                    # let the last frames through
    path = np.array(path)

    res = n.call(n.query, {'limit': 1000})
    mp = res.get('objects', [])
    gt = [(g.label, np.array([g.position.x, g.position.y, g.position.z])) for g in n.gt.objects]
    reach = [(lab, p) for lab, p in gt if np.min(np.linalg.norm(path - p[:2], axis=1)) <= args.reach]
    # canonical labels known to the mapper (a GT label counts when the mapper could produce it)
    mapper_labels = {vocab.canon(o['label']) for o in mp}
    known = {vocab.canon(lab) for lab, _ in gt}
    per_label: dict[str, list[int]] = {}
    matched_gt: set[int] = set()
    errors = []
    used: set[int] = set()
    for i, (lab, p) in enumerate(reach):
        cl = vocab.canon(lab)
        cands = [(float(np.linalg.norm(to_world(o['position']) - p[:2])), k) for k, o in enumerate(mp)
                 if vocab.canon(o['label']) == cl and k not in used and not o.get('bearing_only')]
        cands = [x for x in cands if x[0] <= args.match]
        stat = per_label.setdefault(cl, [0, 0])
        stat[1] += 1
        if cands:
            d, k = min(cands)
            used.add(k)
            matched_gt.add(i)
            errors.append(d)
            stat[0] += 1
    correct = 0
    for o in mp:
        if o.get('bearing_only'):
            continue
        cl = vocab.canon(o['label'])
        if any(vocab.canon(lab) == cl and np.linalg.norm(to_world(o['position']) - p[:2]) <= args.match for lab, p in gt):
            correct += 1
    nav = [o for o in mp if not o.get('bearing_only')]
    summ = n.call(n.summary, {'pose': [0, 0, 0], 'max_objects': 30})
    out = {
        'sim_seconds': round(sim_t(), 1), 'control_steps': steps, 'obstacle_turns': blocked,
        'path_length_m': round(float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))), 1),
        'detection_msgs': n.n_det_msgs, 'detections': n.n_det,
        'gt_objects': len(gt), 'gt_in_reach': len(reach),
        'gt_in_reach_with_map_label': sum(1 for lab, _ in reach if vocab.canon(lab) in mapper_labels),
        'map_objects': len(mp), 'map_bearing_only': len(mp) - len(nav),
        'recall_in_reach': round(len(matched_gt) / max(len(reach), 1), 3),
        'precision_same_label_within_match': round(correct / max(len(nav), 1), 3),
        'position_error_m': {'n': len(errors), 'mean': round(float(np.mean(errors)), 3) if errors else None,
                             'median': round(float(np.median(errors)), 3) if errors else None,
                             'p90': round(float(np.percentile(errors, 90)), 3) if errors else None},
        'per_label_matched_total': {k: v for k, v in sorted(per_label.items())},
        'map_labels': sorted(mapper_labels), 'gt_labels_not_in_vocab_text': sorted(known - mapper_labels)[:40],
        'summary_text': summ.get('text', '')[:1500],
        'params': {'reach_m': args.reach, 'match_m': args.match},
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ('summary_text', 'map_labels', 'gt_labels_not_in_vocab_text')}))
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
