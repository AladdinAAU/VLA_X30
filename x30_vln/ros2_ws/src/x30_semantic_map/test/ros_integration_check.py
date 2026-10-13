#!/usr/bin/env python3
"""ROS integration check of the `semantic_mapper` node (run inside the Humble container, node already running).

Plays the synthetic scenario (noisy duplicates, a walking person, bearing-only rays, false positives, a moved chair,
NaN thermal values) as /perception/detections together with TF, /map and camera_info, then asserts on the published
topics and on every service. Exit code 0 = all checks passed. Writes the measured numbers as JSON to --out.

    python3 ros_integration_check.py [--seed 0] [--rate 20] [--out result.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
import traceback

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped, Vector3
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray
from x30_msgs.msg import Detection3D, Detection3DArray, SemanticObjectArray
from x30_msgs.srv import Json

from x30_semantic_map import synthetic as syn


def yaw_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def optical_quat(yaw: float):
    """Rotation of the optical frame (z forward, x right, y down) for a camera looking along `yaw` (level)."""
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[s, 0.0, c], [-c, 0.0, s], [0.0, -1.0, 0.0]])
    tr = np.trace(r)
    w = math.sqrt(1 + tr) / 2
    return ((r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w)


class Check(Node):
    def __init__(self, scenario: syn.Scenario, rate: float):
        super().__init__('semantic_map_check')
        self.sc, self.rate = scenario, rate
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub_det = self.create_publisher(Detection3DArray, '/perception/detections', reliable)
        self.pub_map = self.create_publisher(OccupancyGrid, '/map', latched)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/color/camera_info', 5)
        self.tf = TransformBroadcaster(self)
        self.objects: SemanticObjectArray | None = None
        self.n_objects_msgs = 0
        self.markers: MarkerArray | None = None
        self.asm: Image | None = None
        self.create_subscription(SemanticObjectArray, '/semantic_map/objects', self._on_obj, latched)
        self.create_subscription(MarkerArray, '/semantic_map/markers', lambda m: setattr(self, 'markers', m), 1)
        self.create_subscription(Image, '/semantic_map/annotated_map', lambda m: setattr(self, 'asm', m), 1)
        self.svc = {n: self.create_client(Json, f'/semantic_map/{n}') for n in
                        ('query', 'find', 'relations', 'summary', 'reset', 'save', 'load', 'render', 'feedback', 'status')}

    def _on_obj(self, m):
        self.objects = m
        self.n_objects_msgs += 1

    # ---------------------------------------------------------------------------------------- publishing
    def publish_grid(self) -> None:
        g = self.sc.grid
        m = OccupancyGrid()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution, m.info.width, m.info.height = g.resolution, g.width, g.height
        m.info.origin.position.x, m.info.origin.position.y = g.origin_x, g.origin_y
        m.info.origin.orientation.w = 1.0
        m.data = g.data.flatten().astype(np.int8).tolist()
        self.pub_map.publish(m)
        info = CameraInfo()
        info.width, info.height = 640, 480
        fx = 320.0 / math.tan(math.radians(self.sc.cfg.hfov_deg / 2))
        fy = 240.0 / math.tan(math.radians(self.sc.cfg.vfov_deg / 2))
        info.k = [fx, 0.0, 320.0, 0.0, fy, 240.0, 0.0, 0.0, 1.0]
        info.header.frame_id = 'camera_optical_frame'
        self.pub_info.publish(info)

    def send_tf(self, pose, stamp) -> None:
        x, y, yaw = pose
        out = []
        for child, (px, py, pz), q in (('base_link', (x, y, 0.0), yaw_quat(yaw)),
                                       ('camera_optical_frame', (x, y, 0.5), optical_quat(yaw))):
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, 'map', child
            t.transform.translation = Vector3(x=px, y=py, z=pz)
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
            out.append(t)
        self.tf.sendTransform(out)

    def det_msg(self, frame: syn.Frame, stamp) -> Detection3DArray:
        arr = Detection3DArray()
        arr.header.frame_id, arr.header.stamp, arr.image_stamp = 'map', stamp, stamp
        arr.camera_position = Point(x=float(frame.camera_position[0]), y=float(frame.camera_position[1]),
                                    z=float(frame.camera_position[2]))
        for d in frame.detections:
            m = Detection3D()
            m.label, m.score = d.label, float(d.score)
            m.box_xyxy = [int(v) for v in d.box_xyxy]
            m.point_count, m.bearing_only, m.depth_source = int(d.point_count), bool(d.bearing_only), 'lidar'
            if not d.bearing_only:
                m.centroid = Point(x=float(d.centroid[0]), y=float(d.centroid[1]), z=float(d.centroid[2]))
                m.extent = Vector3(x=float(d.extent[0]), y=float(d.extent[1]), z=float(d.extent[2]))
            m.bearing = Vector3(x=float(d.bearing[0]), y=float(d.bearing[1]), z=float(d.bearing[2]))
            m.thermal_mean_c = m.thermal_max_c = float('nan')           # no thermal camera
            if d.crop:
                m.crop = CompressedImage()
                m.crop.format, m.crop.data = 'jpeg', d.crop
            arr.detections.append(m)
        return arr

    def play(self) -> None:
        self.publish_grid()
        t_end = time.time() + 2.0
        while time.time() < t_end and self.n_objects_msgs == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        period = 1.0 / self.rate
        t_next = time.time()
        for frame, pose in zip(self.sc.frames, self.sc.robot_poses):
            stamp = self.get_clock().now().to_msg()
            self.send_tf(pose, stamp)
            self.pub_det.publish(self.det_msg(frame, stamp))
            t_next += period
            while time.time() < t_next:
                rclpy.spin_once(self, timeout_sec=max(0.0, min(0.01, t_next - time.time())))

    # ---------------------------------------------------------------------------------------- services
    def call(self, name: str, req: dict | None = None, timeout: float = 20.0, expect_ok: bool = True) -> dict:
        cli = self.svc[name]
        if not cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f'service {name} not available')
        r = Json.Request()
        r.request = json.dumps(req or {})
        fut = cli.call_async(r)
        t_end = time.time() + timeout
        while not fut.done() and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not fut.done():
            raise RuntimeError(f'service {name} timed out')
        res = fut.result()
        body = json.loads(res.response)
        self.last_call = (name, req, res.response[:300])
        body['_ok'] = bool(res.ok)
        if expect_ok and not res.ok:
            raise RuntimeError(f'service {name} {req} failed: {res.response}')
        return body

    def spin_for(self, seconds: float) -> None:
        t_end = time.time() + seconds
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)


class Results:
    def __init__(self):
        self.failures: list[str] = []
        self.numbers: dict = {}

    def check(self, cond: bool, what: str) -> None:
        print(('PASS ' if cond else 'FAIL ') + what, flush=True)
        if not cond:
            self.failures.append(what)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--out', default='/tmp/semantic_map_integration.json')
    ap.add_argument('--save-dir', default='/tmp')
    ap.add_argument('--png', default='')
    args = ap.parse_args()

    sc = syn.generate(syn.ScenarioConfig(seed=args.seed))
    rclpy.init()
    node = Check(sc, args.rate)
    R = Results()
    try:
        t0 = time.time()
        node.play()
        node.spin_for(2.5)
        R.numbers['play_s'] = round(time.time() - t0, 1)
        R.numbers['frames'] = len(sc.frames)

        # ---- topics
        m = node.objects
        R.check(m is not None and len(m.objects) > 10, f'/semantic_map/objects published with {len(m.objects) if m else 0} objects')
        R.check(node.markers is not None and any(k.ns == 'labels' for k in node.markers.markers), 'markers published')
        R.check(node.asm is not None and node.asm.encoding == 'rgb8' and node.asm.width > 200, 'annotated map image published')
        labels = [o.label for o in m.objects]
        by_label = {}
        for o in m.objects:
            by_label.setdefault(o.label, []).append(o)

        # ---- accuracy vs ground truth (objects topic only)
        gt = [g for g in sc.gt_final if not g.dynamic]
        errs, found, dups = [], 0, 0
        for g in gt:
            cand = [(math.hypot(o.position.x - g.pos[0], o.position.y - g.pos[1]), o) for o in m.objects
                    if (o.label == g.label or (g.label == 'chair' and o.label in ('armchair', 'seat', 'office chair'))
                        or (g.label == 'tv' and o.label in ('monitor', 'television'))
                        or (g.label == 'couch' and o.label == 'sofa') or (g.label == 'trash can' and o.label == 'bin')
                        or (g.label == 'table' and o.label in ('desk', 'dining table')))]
            cand = sorted(c for c in cand if c[0] <= 1.0)
            if cand:
                found += 1
                errs.append(cand[0][0])
                dups += len(cand) - 1
        R.numbers.update(recall=round(found / len(gt), 3), mean_err_m=round(float(np.mean(errs)), 3),
                         median_err_m=round(float(np.median(errs)), 3), duplicates=dups, objects=len(m.objects))
        R.check(found / len(gt) >= 0.8, f'recall {found}/{len(gt)} >= 0.8')
        R.check(float(np.mean(errs)) < 0.15, f'mean position error {np.mean(errs):.3f} m < 0.15')
        R.check(dups == 0, f'no duplicate instances ({dups})')
        moved = next(g for g in sc.gt_final if g.name == 'chair_k3')
        old_chair = [o for o in m.objects if o.label in ('chair', 'armchair') and math.hypot(o.position.x - 3.0, o.position.y - 3.9) < 0.6]
        R.check(not old_chair, 'the moved chair is gone from its old place')
        persons = by_label.get('person', [])
        R.check(len(persons) <= 1, f'at most one person instance ({len(persons)})')
        R.check(all(not any(a in ('warm', 'hot') for a in o.attributes) for o in m.objects), 'no thermal attributes without thermal data')
        some = next(o for o in m.objects if o.label == 'refrigerator')
        R.check(some.room in ('kitchen',), f'refrigerator room = {some.room!r}')
        R.check(some.confidence > 0.3 and some.observations >= 4 and len(some.best_view.data) > 100, 'object record fields filled (confidence, observations, best_view)')

        # ---- services
        q = node.call('query', {'label': 'chair', 'limit': 20})
        R.check(q['_ok'] and q['count'] >= 3 and all('chair' in o['canon'] for o in q['objects']), f"query chair -> {q.get('count')}")
        q = node.call('query', {'label': 'chair', 'attributes': ['red']})
        R.check(q['_ok'] and q['count'] >= 1, 'query by attribute (red chair)')
        q = node.call('query', {'label': 'chair', 'near': [3.0, 5.0], 'radius': 2.0})
        R.check(q['_ok'] and all(math.hypot(o['position'][0] - 3.0, o['position'][1] - 5.0) <= 2.0 for o in q['objects']), 'query near/radius')
        f = node.call('find', {'text': 'the red chair near the table'})
        R.check(f['_ok'] and f['matches'] and f['matches'][0]['label'] == 'chair' and 'red' in f['matches'][0]['attributes'],
                f"find red chair -> #{f['matches'][0]['id'] if f.get('matches') else None} ({f.get('method')})")
        R.check(node.call('find', {}, expect_ok=False)['_ok'] is False, 'find without text returns ok=false')
        tid = by_label['table'][0].id
        rel = node.call('relations', {'id': tid, 'pose': [1.0, 1.0, 0.0]})
        R.check(rel['_ok'] and rel['relations'] and 'text' in rel and rel['from_robot']['range_m'] > 0, 'relations')
        R.check(node.call('relations', {'id': 99999}, expect_ok=False)['_ok'] is False, 'relations of an unknown id returns ok=false')
        s = node.call('summary', {'max_objects': 12, 'token_budget': 400})
        R.check(s['_ok'] and s['tokens_est'] <= 400 and 'Rooms:' in s['text'] and 'kitchen' in s['text'], 'summary within the token budget, with rooms')
        R.numbers['summary_tokens'] = s.get('tokens_est')
        img = node.call('render', {'mode': 'image', 'max_size_px': 700})
        png = base64.b64decode(img['png_base64'])
        R.check(img['_ok'] and png[:4] == b'\x89PNG' and img['width'] > 100, f"render image {img.get('width')}x{img.get('height')}")
        if args.png:
            open(args.png, 'wb').write(png)
        txt = node.call('render', {'mode': 'text'})
        R.check(txt['_ok'] and 'Legend:' in txt['text'], 'render text')
        st = node.call('status')
        R.check(st['_ok'] and st['stats']['frames'] == len(sc.frames) and st['grid'] and len(st['regions']) == 3, f"status frames={st['stats']['frames']} regions={st['regions']}")
        R.numbers['stats'] = st['stats']
        R.check(st['stats']['rays'] > 20 and st['stats']['triangulated'] >= 1, f"bearing-only rays {st['stats']['rays']}, triangulated objects {st['stats']['triangulated']}")
        fb = node.call('feedback', {'id': by_label['table'][0].id, 'action': 'confirm'})
        R.check(fb['_ok'] and fb['applied'], 'feedback confirm')
        path = f'{args.save_dir}/semmap_check.json'
        sv = node.call('save', {'path': path})
        n_before = sv['objects']
        R.check(sv['_ok'] and n_before > 10, f'save {n_before} objects')
        R.check(node.call('reset')['_ok'] and node.call('query', {'limit': 5})['count'] == 0, 'reset empties the map')
        ld = node.call('load', {'path': path})
        R.check(ld['_ok'] and ld['objects'] == n_before, f"load restores {ld.get('objects')} objects")
        R.check(node.call('query', {'label': 'chair', 'limit': 20})['count'] == q_count(node, 'chair') >= 3, 'queries work after load')
        R.check(node.call('load', {'path': '/nonexistent/x.json'}, expect_ok=False)['_ok'] is False, 'load of a missing file returns ok=false')
        R.check(node.call('status')['_ok'], 'node alive after a bad request')
    except Exception as ex:                                 # report, never hang
        traceback.print_exc()
        print('LAST SERVICE CALL', node.last_call if hasattr(node, 'last_call') else None, flush=True)
        R.check(False, f'exception: {type(ex).__name__}: {ex}')
    finally:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'failures': R.failures, 'numbers': R.numbers}, f, indent=1)
        node.destroy_node()
        rclpy.shutdown()
    print('NUMBERS', json.dumps(R.numbers))
    print('RESULT', 'PASS' if not R.failures else f'FAIL ({len(R.failures)}): {R.failures}')
    return 0 if not R.failures else 1


def q_count(node: Check, label: str) -> int:
    return node.call('query', {'label': label, 'limit': 20})['count']


if __name__ == '__main__':
    sys.exit(main())
