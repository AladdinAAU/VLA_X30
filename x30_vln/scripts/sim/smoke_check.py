#!/usr/bin/env python3
"""Contract check of a running X30 simulator (run inside the sim container; used by smoke_test.sh).

Checks: topics and types, rates (wall and simulated), merged-lidar layout, TF completeness, odometry against ground truth after a
3 m drive with /cmd_vel, /sim/set_robot_pose. Saves one RGB frame, one thermal frame (PNG), the lidar scan (PCD + top-down PNG) and
report.json into --out. Exit code 0 only if every check passed.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import BatteryState, CameraInfo, Image, Imu, PointCloud2
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener
from x30_msgs.msg import SemanticObjectArray
from x30_msgs.srv import Json
from x30_sim.config import load_robot_config

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
SENSOR = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)

# name, type, qos, nominal sim-time rate (Hz) or None for "just exists", latched
TOPICS = [
    ('/lidar_points', PointCloud2, SENSOR, 10.0),
    ('/imu/data', Imu, SENSOR, 100.0),
    ('/odom', Odometry, SENSOR, 30.0),
    ('/camera/color/image_raw', Image, SENSOR, 9.0),
    ('/camera/color/camera_info', CameraInfo, SENSOR, 9.0),
    ('/camera/depth/image_raw', Image, SENSOR, 9.0),
    ('/thermal/image_raw', Image, SENSOR, 4.5),
    ('/thermal/temperature', Image, SENSOR, 4.5),
    ('/thermal/camera_info', CameraInfo, SENSOR, 4.5),
    ('/x30/battery', BatteryState, SENSOR, 0.9),
    ('/x30/health', DiagnosticArray, SENSOR, 0.9),
    ('/gt/pose', PoseStamped, SENSOR, 20.0),
    ('/clock', Clock, SENSOR, None),
    ('/x30/ready', Bool, LATCHED, None),
    ('/gt/objects', SemanticObjectArray, LATCHED, None),
    ('/gt/scene_info', String, LATCHED, None),
]
EXPECTED_FIELDS = [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('intensity', 12, 7), ('tag', 16, 2), ('line', 17, 2), ('timestamp', 18, 8)]
TF_PAIRS = [('odom', 'base_link'), ('base_link', 'lidar_link'), ('base_link', 'imu_link'), ('base_link', 'camera_optical_frame'),
            ('base_link', 'thermal_optical_frame'), ('base_link', 'lidar_fl_link'), ('base_link', 'lidar_fr_link'),
            ('base_link', 'lidar_rl_link'), ('base_link', 'lidar_rr_link')]


class Checker(Node):
    def __init__(self, thermal: bool):
        super().__init__('smoke_check')
        self.thermal = thermal
        self.rec = {}
        self.last = {}
        self.sim_now = 0.0
        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)
        for name, typ, qos, _ in TOPICS:
            self.rec[name] = []
            self.create_subscription(typ, name, lambda m, n=name: self.on_msg(n, m), qos)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cli_pose = self.create_client(Json, '/sim/set_robot_pose')

    def on_msg(self, name, msg):
        if name == '/clock':
            self.sim_now = msg.clock.sec + msg.clock.nanosec * 1e-9
        stamp = None
        if hasattr(msg, 'header'):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.rec[name].append((time.monotonic(), stamp if stamp is not None else self.sim_now))
        self.last[name] = msg

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def spin_until(self, cond, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not cond():
            rclpy.spin_once(self, timeout_sec=0.05)
        return cond()


def rate(recs, window=None):
    if len(recs) < 3:
        return 0.0, 0.0
    recs = recs[-window:] if window else recs
    wall = (len(recs) - 1) / max(recs[-1][0] - recs[0][0], 1e-6)
    sim = (len(recs) - 1) / max(recs[-1][1] - recs[0][1], 1e-6)
    return wall, sim


def save_pcd(path, arr):
    pts = arr
    with open(path, 'w') as f:
        f.write('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
                f'WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA ascii\n')
        np.savetxt(f, np.c_[pts['x'], pts['y'], pts['z'], pts['intensity']], fmt='%.4f')


def save_bev(path, arr, extent=8.0, px=600):
    img = np.full((px, px, 3), 255, np.uint8)
    pts = arr
    v = (px / 2 - pts['x'] / extent * px / 2).astype(int)
    u = (px / 2 - pts['y'] / extent * px / 2).astype(int)
    ok = (u >= 0) & (u < px) & (v >= 0) & (v < px)
    z = np.clip((pts['z'][ok] + 0.6) / 3.4, 0, 1)
    colors = cv2.applyColorMap((z * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    img[v[ok], u[ok]] = colors
    cv2.circle(img, (px // 2, px // 2), 4, (0, 0, 0), -1)
    cv2.arrowedLine(img, (px // 2, px // 2), (px // 2, px // 2 - 30), (0, 0, 0), 2)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--locomotion', default='kinematic')
    ap.add_argument('--no-thermal', action='store_true')
    ap.add_argument('--drive', type=float, default=3.0)
    ap.add_argument('--speed', type=float, default=0.3)
    ap.add_argument('--measure', type=float, default=12.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Checker(not a.no_thermal)
    report = {'locomotion': a.locomotion, 'checks': {}, 'rates': {}}
    results = report['checks']

    def check(name, ok, detail=''):
        results[name] = {'ok': bool(ok), 'detail': str(detail)}
        print(f'[{"PASS" if ok else "FAIL"}] {name} {detail}', flush=True)

    ready = node.spin_until(lambda: node.last.get('/x30/ready') is not None and node.last['/x30/ready'].data, 150.0)
    check('ready', ready, '/x30/ready true')
    node.spin_for(a.measure)

    # topics, types, rates
    for name, _, _, nominal in TOPICS:
        if not a.no_thermal or not name.startswith('/thermal'):
            n = len(node.rec[name])
            wall, sim = rate(node.rec[name])
            report['rates'][name] = {'count': n, 'wall_hz': round(wall, 2), 'sim_hz': round(sim, 2)}
            ok = n > 0 and (nominal is None or sim >= 0.8 * nominal)
            check(f'topic {name}', ok, f'{n} msgs, {wall:.1f} Hz wall, {sim:.1f} Hz sim-time (nominal {nominal})')

    # lidar layout
    cloud = node.last.get('/lidar_points')
    arr = None
    if cloud is not None:
        fields = [(f.name, f.offset, f.datatype) for f in cloud.fields]
        layout_ok = (fields == EXPECTED_FIELDS and cloud.point_step == 26 and cloud.is_dense and cloud.header.frame_id == 'lidar_link')
        check('lidar layout', layout_ok, f'fields={fields} step={cloud.point_step} dense={cloud.is_dense} frame={cloud.header.frame_id}')
        arr = np.frombuffer(bytes(cloud.data), dtype=np.dtype({'names': [f[0] for f in EXPECTED_FIELDS],
                                                                'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
                                                                'offsets': [f[1] for f in EXPECTED_FIELDS], 'itemsize': 26}))
        xyz = np.c_[arr['x'], arr['y'], arr['z']].astype(float)
        mounts = np.array([m['position_mm'] for m in load_robot_config()['lidar']['mounts'].values()]) / 1000.0
        valid = np.min(np.linalg.norm(xyz[:, None, :] - mounts[None], axis=2), axis=1) > 1e-3     # invalid rays sit at the sensor origin
        rng = np.linalg.norm(xyz, axis=1)
        report['lidar'] = {'points': int(len(arr)), 'valid_fraction': round(float(valid.mean()), 3),
                           'range_p50': float(np.percentile(rng[valid], 50)) if valid.any() else None,
                           'z_range': [float(arr['z'][valid].min()), float(arr['z'][valid].max())] if valid.any() else None,
                           'lines': sorted(set(int(v) for v in np.unique(arr['line']))), 'intensity_max': float(arr['intensity'].max())}
        check('lidar points ~79872', abs(len(arr) - 79872) < 2000 and valid.mean() > 0.2, json.dumps(report['lidar']))
        save_pcd(out / 'lidar.pcd', arr[valid])
        save_bev(out / 'lidar_bev.png', arr[valid])

    # images
    rgb = node.last.get('/camera/color/image_raw')
    if rgb is not None:
        im = np.frombuffer(bytes(rgb.data), np.uint8).reshape(rgb.height, rgb.width, 3)
        cv2.imwrite(str(out / 'rgb.png'), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        check('rgb image', rgb.encoding == 'rgb8' and rgb.header.frame_id == 'camera_optical_frame' and im.std() > 5,
              f'{rgb.width}x{rgb.height} {rgb.encoding} {rgb.header.frame_id} std={im.std():.1f}')
    depth = node.last.get('/camera/depth/image_raw')
    if depth is not None:
        d = np.frombuffer(bytes(depth.data), np.float32).reshape(depth.height, depth.width)
        fin = d[np.isfinite(d)]
        check('depth image', depth.encoding == '32FC1' and fin.size > 0, f'{depth.encoding} median {np.median(fin) if fin.size else None:.2f} m')
    if not a.no_thermal:
        th = node.last.get('/thermal/image_raw')
        tt = node.last.get('/thermal/temperature')
        if th is not None and tt is not None:
            cv2.imwrite(str(out / 'thermal.png'), np.frombuffer(bytes(th.data), np.uint8).reshape(th.height, th.width))
            t = np.frombuffer(bytes(tt.data), np.float32).reshape(tt.height, tt.width)
            report['thermal'] = {'min_c': float(t.min()), 'max_c': float(t.max()), 'median_c': float(np.median(t))}
            check('thermal images', th.encoding == 'mono8' and tt.encoding == '32FC1' and np.isfinite(t).all(), json.dumps(report['thermal']))
    info = node.last.get('/camera/color/camera_info')
    check('camera_info', info is not None and info.k[0] > 1.0, f'fx={info.k[0] if info else None}')

    # TF
    missing = []
    for parent, child in TF_PAIRS:
        if a.no_thermal and 'thermal' in child:
            continue
        if not node.tfbuf.can_transform(parent, child, rclpy.time.Time(), rclpy.duration.Duration(seconds=3.0)):
            missing.append(f'{parent}->{child}')
    check('tf tree', not missing, f'missing: {missing}')

    # drive 3 m
    gt0 = node.last['/gt/pose'].pose.position
    od0 = node.last['/odom'].pose.pose.position
    t_wall_end = time.monotonic() + 300.0
    sim0 = node.sim_now
    twist = Twist()
    twist.linear.x = a.speed
    dist = reached = 0.0
    while time.monotonic() < t_wall_end and node.sim_now - sim0 < 60.0 and dist < a.drive:
        node.pub.publish(twist)
        node.spin_for(0.05)
        p = node.last['/gt/pose'].pose.position
        dist = math.hypot(p.x - gt0.x, p.y - gt0.y)
        reached = max(reached, dist)
    node.pub.publish(Twist())
    node.spin_for(2.0)
    gt1, od1 = node.last['/gt/pose'].pose.position, node.last['/odom'].pose.pose.position
    d_gt = math.hypot(gt1.x - gt0.x, gt1.y - gt0.y)
    d_od = math.hypot(od1.x - od0.x, od1.y - od0.y)
    report['drive'] = {'reached_m': round(reached, 3), 'final_gt_m': round(d_gt, 3), 'odom_m': round(d_od, 3), 'z_gt': round(gt1.z, 3), 'start': [round(gt0.x, 2), round(gt0.y, 2)],
                       'end': [round(gt1.x, 2), round(gt1.y, 2)], 'sim_s': round(node.sim_now - sim0, 1)}
    check('drive 3 m', reached >= a.drive - 0.1 and abs(d_gt - d_od) < 0.15, json.dumps(report['drive']))

    # set_robot_pose
    if node.cli_pose.wait_for_service(timeout_sec=5.0):
        fut = node.cli_pose.call_async(Json.Request(request=json.dumps({'x': 1.0, 'y': -1.0, 'yaw': 0.5})))
        node.spin_until(fut.done, 90.0)
        node.spin_for(2.0)
        p = node.last['/gt/pose'].pose.position
        err = math.hypot(p.x - 1.0, p.y + 1.0)
        check('set_robot_pose', fut.done() and fut.result().ok and err < 0.25, f'{fut.result().response if fut.done() else "timeout"} err={err:.2f}')
    else:
        check('set_robot_pose', False, 'service missing')

    # e-stop: the robot must not move while /x30/estop is true, even with a velocity command
    estop = node.create_publisher(Bool, '/x30/estop', 5)
    node.spin_for(1.0)
    p0 = node.last['/gt/pose'].pose.position
    estop.publish(Bool(data=True))
    node.spin_for(0.5)
    twist.linear.x = 0.3
    end = time.monotonic() + 8.0
    while time.monotonic() < end:
        node.pub.publish(twist)
        node.spin_for(0.05)
    p1 = node.last['/gt/pose'].pose.position
    moved = math.hypot(p1.x - p0.x, p1.y - p0.y)
    estop.publish(Bool(data=False))
    node.pub.publish(Twist())
    node.spin_for(1.0)
    check('estop holds the robot', moved < 0.15, f'moved {moved:.3f} m with estop true')

    # reset_world puts the robot back at its spawn
    spawn = (json.loads(node.last['/gt/scene_info'].data).get('spawn') or {}) if node.last.get('/gt/scene_info') else {}
    sx, sy = float(spawn.get('x', 0.0)), float(spawn.get('y', 0.0))
    cli = node.create_client(Json, '/sim/reset_world')
    if cli.wait_for_service(timeout_sec=5.0):
        fut = cli.call_async(Json.Request(request='{}'))
        node.spin_until(fut.done, 120.0)
        node.spin_for(2.0)
        p = node.last['/gt/pose'].pose.position
        check('reset_world', fut.done() and fut.result().ok and math.hypot(p.x - sx, p.y - sy) < 0.4,
              f'pose after reset ({p.x:.2f}, {p.y:.2f}), spawn ({sx:.2f}, {sy:.2f})')
    else:
        check('reset_world', False, 'service missing')
    objs = node.last.get('/gt/objects')
    info = json.loads(node.last['/gt/scene_info'].data) if node.last.get('/gt/scene_info') else {}
    check('ground truth objects', objs is not None and len(objs.objects) > 0 and bool(info.get('bounds')),
          f'{len(objs.objects) if objs else 0} objects, scene {info.get("scene")}')

    (out / 'report.json').write_text(json.dumps(report, indent=2))
    failed = [k for k, v in results.items() if not v['ok']]
    print(f'{len(results) - len(failed)}/{len(results)} checks passed' + (f'; FAILED: {failed}' if failed else ''))
    rclpy.shutdown()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
