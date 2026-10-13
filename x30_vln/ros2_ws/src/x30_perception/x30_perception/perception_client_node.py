"""ROS 2 node `perception_client`: RGB frames -> perception server -> masks -> lidar lifting -> x30_msgs/Detection3DArray.

Threading: the RGB callback hands a frame to a single worker thread (server round trip, lifting, publishing) at most `rate_hz`
times per second; frames arriving while the worker is busy are dropped and counted, so the callbacks never block and latency
never accumulates. The other callbacks only store messages.

Frames: output positions are in `map_frame` when TF provides map <- camera, else in `fallback_frame` (odom); the frame_id of
the published array says which. Lidar scans are accumulated in `accum_frame` (odom) so robot motion between the scans and the
image time is compensated; without that TF the single scan nearest to the image is used (uncompensated, a warning is logged).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, Vector3
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from tf2_ros import Buffer, TransformListener
from x30_msgs.msg import Detection3D, Detection3DArray
from x30_msgs.srv import Json

from . import ros_image
from .camera_model import CameraModel
from .config import Vocabulary, load_calibration, load_yaml
from .geometry import apply, invert, transform_from_translation_quaternion, transform_from_xyzrpy
from .lidar_projector import ProjectorParams, Scan, merge_scans, project_points, select_scans
from .lifting import add_thermal, lift_instances, mask_rle_json
from .overlay import draw_detections
from .pointcloud import decode_xyz
from .server_client import PerceptionClient, ServerError
from .thermal import mono8_to_celsius


def stamp_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _odd(n: int) -> int:
    return max(3, int(n) | 1)


class _RawScan:
    """A received point cloud, decoded lazily (only the scans actually used are parsed)."""

    def __init__(self, msg, stamp_start: float, duration: float):
        self.msg, self.stamp, self.duration, self._scan = msg, stamp_start, duration, None

    def scan(self, time_field: str, time_unit_s: float, duration: float) -> Scan:
        if self._scan is None:
            xyz, t = decode_xyz(
                self.msg.data, self.msg.fields, self.msg.point_step, self.msg.is_bigendian, time_field, time_unit_s
            )
            t_rel = None
            if t is not None and t.size:
                rel = t - t.min()
                t_rel = rel if 1e-4 < float(rel.max()) < 0.5 else None  # unusable clock: single pose per scan
            self._scan = Scan(self.stamp, xyz, t_rel, duration)
            self.msg = None
        return self._scan


class PerceptionClientNode(Node):
    def __init__(self):
        super().__init__('perception_client')
        p = self.declare_parameter
        # topics
        self.rgb_topic = p('rgb_topic', '/camera/color/image_raw').value
        self.info_topic = p('camera_info_topic', '/camera/color/camera_info').value
        self.lidar_topic = p('lidar_topic', '/lidar_points').value
        self.thermal_topic = p('thermal_image_topic', '/thermal/image_raw').value
        self.thermal_temp_topic = p('thermal_temperature_topic', '/thermal/temperature').value
        self.thermal_info_topic = p('thermal_info_topic', '/thermal/camera_info').value
        self.depth_topic = p('depth_topic', '/camera/depth/image_raw').value
        # configuration files
        cfg_path = p('config_file', '').value or os.path.join(os.environ.get('X30_CONFIGS', '/x30/configs'), 'perception.yaml')
        self.cfg = load_yaml(cfg_path) if Path(cfg_path).exists() else {}
        if not self.cfg:
            self.get_logger().warn(f'config file {cfg_path} not found or empty: using a tiny built-in vocabulary')
            self.cfg = {'vocabulary': ['chair', 'table', 'door', 'person'], 'server': {}}
        srv = self.cfg.get('server', {})
        self.vocab = Vocabulary.from_config(self.cfg)
        self._vocab_lock = threading.Lock()
        # server
        url = p('server_url', '').value or os.environ.get('PERCEPTION_URL') or srv.get('url', 'http://perception:8100')
        self.client = PerceptionClient(
            url, float(p('server_timeout_s', srv.get('timeout_s', 5.0)).value), int(srv.get('jpeg_quality', 90))
        )
        self.imgsz = int(p('imgsz', srv.get('imgsz', 640)).value)
        self.regions = srv.get('regions', [])
        self.region_imgsz = int(srv.get('region_imgsz', 1280))
        self.send_max_side = int(p('send_max_side', srv.get('send_max_side', 960)).value)
        self.rate_hz = float(p('rate_hz', 2.5).value)
        self.max_image_age_s = float(p('max_image_age_s', 1.0).value)
        # geometry
        self.depth_source = p('depth_source', 'lidar').value  # lidar | sim_depth | none
        self.map_frame = p('map_frame', 'map').value
        self.fallback_frame = p('fallback_frame', 'odom').value
        self.accum_frame = p('accum_frame', 'odom').value
        self.num_scans = int(p('num_scans', 3).value)
        self.time_bins = int(p('time_bins', 3).value)
        self.scan_duration_s = float(p('scan_duration_s', 0.1).value)
        self.scan_stamp_at = p('scan_stamp_at', 'start').value  # start | middle | end of the scan
        self.scan_time_field = p('scan_time_field', 'timestamp').value
        self.scan_time_unit_s = float(p('scan_time_unit_s', 1e-3).value)
        self.scan_wait_s = float(p('scan_wait_s', 0.15).value)
        self.max_scan_age_s = float(p('max_scan_age_s', 2.0).value)
        self.tf_timeout_s = float(p('tf_timeout_s', 0.1).value)
        self.calib_file = p('calibration_file', '').value
        self.use_override = bool(p('extrinsics_override', False).value)
        self.lidar_to_camera_xyzrpy = list(p('lidar_to_camera_xyzrpy', [0.0] * 6).value)  # camera optical frame in lidar_link
        self.pp = ProjectorParams(
            min_range_m=p('min_range_m', 0.6).value,
            max_range_m=p('max_range_m', 30.0).value,
            occlusion_window_px=p('occlusion_window_px', 5).value,
            occlusion_abs_tol_m=p('occlusion_abs_tol_m', 0.10).value,
            occlusion_rel_tol=p('occlusion_rel_tol', 0.04).value,
            mask_erode_px=p('mask_erode_px', 2).value,
            min_points=p('min_points', 5).value,
            depth_gap_abs_m=p('depth_gap_abs_m', 0.25).value,
            depth_gap_rel=p('depth_gap_rel', 0.04).value,
            cluster_keep_frac=p('cluster_keep_frac', 0.4).value,
            euclid_eps_m=p('euclid_eps_m', 0.30).value,
            euclid_eps_rel=p('euclid_eps_rel', 0.03).value,
        )
        self.min_mask_px = int(self.cfg.get('min_mask_px', 60))
        self.ignore_regions = self.cfg.get('ignore_regions', [])
        # thermal
        self.thermal_default_depth = float(p('thermal_default_depth_m', 3.0).value)
        self.thermal_max_dt = float(p('thermal_max_dt_s', 0.5).value)
        self.thermal_norm = (self.cfg.get('thermal') or {}).get('norm_range_c')
        # outputs
        self.publish_crops = bool(p('publish_crops', True).value)
        self.crop_max_side = int(p('crop_max_side', 256).value)
        self.crop_min_score = float(p('crop_min_score', 0.3).value)
        self.debug_always = bool(p('debug_image_always', False).value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._lock = threading.Lock()
        self._scans: deque[_RawScan] = deque(maxlen=40)
        self._next_submit = 0.0
        self._info: CameraInfo | None = None
        self._thermal: Image | None = None
        self._thermal_temp: Image | None = None
        self._thermal_info: CameraInfo | None = None
        self._depths: deque[Image] = deque(maxlen=6)
        self._cam_cache: CameraModel | None = None
        self._calib = load_calibration(self.calib_file) if self.calib_file else None
        self._busy = False
        self._lidar_frame = 'lidar_link'
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='perception')
        self.stats = {'frames': 0, 'dropped_busy': 0, 'dropped_stale': 0, 'server_errors': 0, 'tf_fallbacks': 0}
        self._warned: set[str] = set()

        self.create_subscription(Image, self.rgb_topic, self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.lidar_topic, self._on_lidar, qos_profile_sensor_data)
        self.create_subscription(Image, self.thermal_topic, lambda m: setattr(self, '_thermal', m), qos_profile_sensor_data)
        self.create_subscription(
            Image, self.thermal_temp_topic, lambda m: setattr(self, '_thermal_temp', m), qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, self.thermal_info_topic, lambda m: setattr(self, '_thermal_info', m), qos_profile_sensor_data
        )
        if self.depth_source == 'sim_depth':
            self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(Detection3DArray, '/perception/detections', reliable)
        self.pub_img = self.create_publisher(Image, '/perception/detections_image', 1)
        self.pub_diag = self.create_publisher(DiagnosticArray, '/perception/diagnostics', 5)
        self.create_service(Json, '/perception/vocabulary', self._on_vocab_srv)
        self.get_logger().info(
            f'perception_client: server {url}, rgb {self.rgb_topic}, lidar {self.lidar_topic}, '
            f'depth_source={self.depth_source}, rate {self.rate_hz} Hz, {len(self.vocab.phrases())} phrases'
        )

    # ------------------------------------------------------------ callbacks (store only)
    def _on_rgb(self, msg: Image) -> None:
        self._offer(msg)

    def _on_info(self, msg: CameraInfo) -> None:
        self._info = msg

    def _on_depth(self, msg: Image) -> None:
        self._depths.append(msg)

    def _on_lidar(self, msg: PointCloud2) -> None:
        t = stamp_sec(msg.header.stamp)
        start = {'start': t, 'middle': t - 0.5 * self.scan_duration_s, 'end': t - self.scan_duration_s}[self.scan_stamp_at]
        self._scans.append(_RawScan(msg, start, self.scan_duration_s))
        self._lidar_frame = msg.header.frame_id

    def _on_vocab_srv(self, req, res):
        try:
            q = json.loads(req.request or '{}')
            op = q.get('op', 'get')
            labels = [str(x) for x in q.get('labels', [])]
            with self._vocab_lock:
                if op == 'add':
                    self.vocab.add(labels, q.get('ttl_s'))
                elif op == 'remove':
                    self.vocab.remove(labels)
                elif op == 'set':
                    self.vocab.set_base(labels)
                elif op == 'reset':
                    self.vocab.reset(self.cfg.get('vocabulary', []))
                elif op not in ('get', 'stats'):
                    raise ValueError(f'unknown op {op!r} (get | add | remove | set | reset | stats)')
                out = {'phrases': self.vocab.phrases(), 'server_conf': self.vocab.server_conf()}
            if op == 'stats':
                out.update(self.stats)
            res.ok, res.response = True, json.dumps(out)
        except Exception as ex:  # noqa: BLE001 - report to the caller instead of crashing the node
            res.ok, res.response = False, json.dumps({'error': str(ex)})
        return res

    # ------------------------------------------------------------ rate limiter
    def _offer(self, msg: Image) -> None:
        """Called for every RGB frame: submit it if the rate limit allows and the worker is free, otherwise drop it."""
        now = time.monotonic()
        if now < self._next_submit:
            return
        if self._busy:
            self.stats['dropped_busy'] += 1
            return
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp_sec(msg.header.stamp)
        if age > self.max_image_age_s:
            self.stats['dropped_stale'] += 1
            self._warn_once('stale', f'RGB images are {age:.1f} s old (clock mismatch or a stalled camera?)')
            return
        self._busy = True
        self._next_submit = now + 1.0 / self.rate_hz
        self._pool.submit(self._process_safe, msg)

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warn(text)

    # ------------------------------------------------------------ TF helpers
    def _lookup(self, target: str, source: str, t: float | None, timeout: float | None = None) -> np.ndarray | None:
        """4x4 T_target_source at time t (None = latest); None when TF cannot provide it."""
        when = rclpy.time.Time() if t is None else rclpy.time.Time(nanoseconds=int(t * 1e9))
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, when, timeout=Duration(seconds=self.tf_timeout_s if timeout is None else timeout)
            )
        except Exception:  # noqa: BLE001 - tf2 raises several exception types; all mean "not available"
            return None
        tr, q = tf.transform.translation, tf.transform.rotation
        return transform_from_translation_quaternion((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))

    def _camera_model(self, img: Image) -> CameraModel | None:
        info = self._info
        if info is not None and info.k[0] > 0:
            cam = CameraModel.from_camera_info(info.width, info.height, info.k, info.d, info.distortion_model or 'plumb_bob')
        elif self._calib is not None:
            cam = self._calib['camera']
        else:
            self._warn_once(
                'nocam',
                'no valid camera_info and no calibration_file: cannot lift detections; '
                'publishing nothing until intrinsics are available',
            )
            return None
        # the image may have another resolution than the one the intrinsics refer to (resized stream)
        return (
            cam
            if (img.width, img.height) == (cam.width, cam.height)
            else cam.scaled(img.width / cam.width, img.height / cam.height)
        )

    def _T_cam_lidar(self, cam_frame: str, lidar_frame: str, t: float) -> np.ndarray | None:
        if self.use_override:
            return invert(transform_from_xyzrpy(self.lidar_to_camera_xyzrpy))
        T = self._lookup(cam_frame, lidar_frame, t) or self._lookup(cam_frame, lidar_frame, None)
        if T is None and self._calib is not None and self._calib['T_lidar_camera'] is not None:
            return invert(self._calib['T_lidar_camera'])
        return T

    # ------------------------------------------------------------ worker
    def _process_safe(self, msg: Image) -> None:
        try:
            self._process(msg)
        except ServerError as ex:
            self.stats['server_errors'] += 1
            self.get_logger().warn(f'perception server: {ex}', throttle_duration_sec=5.0)
            self._publish_diag({'error': str(ex)}, DiagnosticStatus.ERROR)
        except Exception as ex:  # noqa: BLE001 - a bad frame must not kill the worker
            self.get_logger().error(f'frame failed: {type(ex).__name__}: {ex}', throttle_duration_sec=5.0)
        finally:
            self._busy = False

    def _gather_points(self, cam_frame: str, t_img: float, info: dict) -> np.ndarray | None:
        """Lidar points around the image time in the camera frame at the image time (motion compensated), or None."""
        if self.depth_source != 'lidar':
            return None
        deadline = time.monotonic() + self.scan_wait_s
        while time.monotonic() < deadline and (not self._scans or self._scans[-1].stamp + self.scan_duration_s < t_img):
            time.sleep(0.01)  # the scan covering the image may still be arriving
        raw = [s for s in list(self._scans) if t_img - s.stamp < self.max_scan_age_s]
        if not raw:
            self._warn_once('noscan', f'no lidar scans on {self.lidar_topic}: detections will be bearing_only')
            return None
        chosen = select_scans(raw, t_img, self.num_scans)
        decoded = [r.scan(self.scan_time_field, self.scan_time_unit_s, self.scan_duration_s) for r in chosen]
        lidar_frame = getattr(self, '_lidar_frame', 'lidar_link')
        T_cam_fixed = self._lookup(cam_frame, self.accum_frame, t_img, timeout=0.2)
        if T_cam_fixed is not None and self._lookup(self.accum_frame, lidar_frame, decoded[-1].stamp) is not None:
            pts, minfo = merge_scans(
                decoded,
                lambda t: self._lookup(self.accum_frame, lidar_frame, t, timeout=0.05),
                self.time_bins,
                self.pp.min_range_m,
                self.pp.max_range_m,
            )
            info.update(minfo, compensated=True)
            if pts.shape[0]:
                return apply(T_cam_fixed, pts.astype(np.float64))
        # no odometry TF: nearest scan, static extrinsics only
        self.stats['tf_fallbacks'] += 1
        self._warn_once(
            'nocomp',
            f'no TF {self.accum_frame} -> {lidar_frame}: using the nearest single scan without motion '
            'compensation (points smear while the robot moves)',
        )
        T_cl = self._T_cam_lidar(cam_frame, lidar_frame, t_img)
        if T_cl is None:
            self._warn_once(
                'noext',
                f'no extrinsics {cam_frame} <- {lidar_frame} (TF, calibration_file or '
                'lidar_to_camera_xyzrpy): detections will be bearing_only',
            )
            return None
        s = min(decoded, key=lambda sc: abs(sc.stamp + 0.5 * sc.duration - t_img))
        pts, minfo = merge_scans([s], lambda t: np.eye(4), 1, self.pp.min_range_m, self.pp.max_range_m)
        info.update(minfo, compensated=False)
        return apply(T_cl, pts.astype(np.float64))

    def _output_frame(self, cam_frame: str, t_img: float):
        for frame in (self.map_frame, self.fallback_frame):
            T = self._lookup(frame, cam_frame, t_img, timeout=0.2)
            if T is not None:
                if frame != self.map_frame:
                    self._warn_once('nomap', f'TF {self.map_frame} not available: publishing detections in {frame}')
                return frame, T
        return None, None

    def _process(self, msg: Image) -> None:
        t_start = time.perf_counter()
        t_img = stamp_sec(msg.header.stamp)
        cam_frame = msg.header.frame_id
        cam = self._camera_model(msg)
        if cam is None:
            return
        out_frame, T_out_cam = self._output_frame(cam_frame, t_img)
        if out_frame is None:
            self._warn_once('notf', f'no TF {self.map_frame}/{self.fallback_frame} <- {cam_frame}: frames are skipped')
            return
        bgr = ros_image.to_bgr(msg)
        H, W = bgr.shape[:2]
        scale = min(1.0, self.send_max_side / max(H, W))
        sent = bgr if scale >= 1.0 else cv2.resize(bgr, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
        with self._vocab_lock:
            phrases, conf = self.vocab.phrases(), self.vocab.server_conf()
        resp, tm = self.client.segment(sent, phrases, conf, self.imgsz, self.regions, self.region_imgsz)
        wh, ww = resp['image']['height'], resp['image']['width']
        cam_w = cam.scaled(ww / cam.width, wh / cam.height)

        t_lift = time.perf_counter()
        info: dict = {}
        proj = None
        depth_img = None
        if self.depth_source == 'lidar':
            pts_cam = self._gather_points(cam_frame, t_img, info)
            if pts_cam is not None:
                pp = ProjectorParams(**{**self.pp.__dict__, 'occlusion_window_px': _odd(self.pp.occlusion_window_px * ww / 640)})
                proj = project_points(cam_w, pts_cam, pp)
                info['proj'] = proj.stats
        elif self.depth_source == 'sim_depth' and self._depths:
            d = min(self._depths, key=lambda m: abs(stamp_sec(m.header.stamp) - t_img))
            depth_img = ros_image.depth_meters(d)
            if depth_img.shape != (wh, ww):
                depth_img = cv2.resize(depth_img, (ww, wh), interpolation=cv2.INTER_NEAREST)
        dets = lift_instances(
            resp['instances'],
            self.vocab,
            cam_w,
            proj,
            self.pp,
            T_out_cam,
            self.min_mask_px,
            self.ignore_regions,
            self.depth_source,
            depth_img,
        )
        lift_ms = (time.perf_counter() - t_lift) * 1000.0

        t_th = time.perf_counter()
        thermal_used = self._thermal_stats(dets, cam_w, cam_frame, t_img)
        thermal_ms = (time.perf_counter() - t_th) * 1000.0

        arr = self._build_message(msg, dets, bgr, (ww, wh), out_frame, T_out_cam)
        self.pub.publish(arr)
        self.stats['frames'] += 1
        if self.debug_always or self.pub_img.get_subscription_count() > 0:
            self._publish_overlay(bgr, dets, (ww, wh), msg.header)
        now = self.get_clock().now().nanoseconds * 1e-9
        vals = {
            'frame_id': out_frame,
            'server_rtt_ms': tm['rtt_ms'],
            'server_ms': resp['ms'],
            'server_infer_ms': resp['timing']['infer_ms'],
            'encode_ms': tm['encode_ms'],
            'lift_ms': lift_ms,
            'thermal_ms': thermal_ms,
            'process_ms': (time.perf_counter() - t_start) * 1000.0,
            'image_to_publish_ms': (now - t_img) * 1000.0,
            'detections': len(dets),
            'bearing_only': sum(d.bearing_only for d in dets),
            'thermal_used': thermal_used,
            'scans_used': info.get('scans_used', 0),
            'motion_compensated': info.get('compensated', False),
            'lidar_points_in_image': info.get('proj', {}).get('kept', 0),
            'occluded_points': info.get('proj', {}).get('occluded', 0),
            **self.stats,
        }
        self._publish_diag(vals, DiagnosticStatus.OK)

    # ------------------------------------------------------------ thermal
    def _thermal_stats(self, dets, cam_w: CameraModel, cam_frame: str, t_img: float) -> bool:
        if not dets:
            return False
        temp_msg, th_img = self._thermal_temp, self._thermal
        info = self._thermal_info
        if info is None or info.k[0] <= 0:
            return False
        if temp_msg is not None and abs(stamp_sec(temp_msg.header.stamp) - t_img) < self.thermal_max_dt:
            temp, msg = ros_image.to_array(temp_msg).astype(np.float32), temp_msg
        elif th_img is not None and self.thermal_norm and abs(stamp_sec(th_img.header.stamp) - t_img) < self.thermal_max_dt:
            temp, msg = mono8_to_celsius(ros_image.to_array(th_img), *self.thermal_norm), th_img
        else:
            return False
        T = self._lookup(msg.header.frame_id, cam_frame, None)
        if T is None:
            self._warn_once('nothermtf', f'no TF {msg.header.frame_id} <- {cam_frame}: thermal statistics unavailable')
            return False
        cam_th = CameraModel.from_camera_info(info.width, info.height, info.k, info.d, info.distortion_model or 'plumb_bob')
        if (msg.width, msg.height) != (cam_th.width, cam_th.height):
            cam_th = cam_th.scaled(msg.width / cam_th.width, msg.height / cam_th.height)
        add_thermal(dets, cam_w, cam_th, T, temp, self.thermal_default_depth)
        return True

    # ------------------------------------------------------------ output
    def _build_message(self, msg: Image, dets, bgr, work_wh, out_frame: str, T_out_cam: np.ndarray) -> Detection3DArray:
        H, W = bgr.shape[:2]
        sx, sy = W / work_wh[0], H / work_wh[1]
        arr = Detection3DArray()
        arr.header.stamp = msg.header.stamp
        arr.header.frame_id = out_frame
        arr.image_stamp = msg.header.stamp
        arr.camera_position = Point(x=float(T_out_cam[0, 3]), y=float(T_out_cam[1, 3]), z=float(T_out_cam[2, 3]))
        R = T_out_cam[:3, :3]
        for d in dets:
            m = Detection3D()
            m.label, m.score = d.label, float(d.score)
            x1, y1, x2, y2 = d.box_xyxy
            m.box_xyxy = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
            m.mask_rle = mask_rle_json(d.mask_rle)
            m.point_count = int(d.point_count)
            b = R @ d.bearing_cam
            m.bearing = Vector3(x=float(b[0]), y=float(b[1]), z=float(b[2]))
            m.bearing_only = bool(d.bearing_only)
            m.depth_source = d.depth_source
            if d.centroid is not None:
                m.centroid = Point(x=float(d.centroid[0]), y=float(d.centroid[1]), z=float(d.centroid[2]))
                m.extent = Vector3(x=float(d.extent[0]), y=float(d.extent[1]), z=float(d.extent[2]))
            m.thermal_mean_c, m.thermal_max_c = float(d.thermal_mean_c), float(d.thermal_max_c)
            if self.publish_crops and d.score >= self.crop_min_score:
                m.crop = self._crop(bgr, (m.box_xyxy), msg.header)
            arr.detections.append(m)
        return arr

    def _crop(self, bgr: np.ndarray, box, header) -> CompressedImage:
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = box
        pad_x, pad_y = int(0.08 * (x2 - x1)), int(0.08 * (y2 - y1))
        x1, y1, x2, y2 = max(x1 - pad_x, 0), max(y1 - pad_y, 0), min(x2 + pad_x, W), min(y2 + pad_y, H)
        out = CompressedImage()
        out.header = header
        out.format = 'jpeg'
        if x2 - x1 < 4 or y2 - y1 < 4:
            return out
        crop = bgr[y1:y2, x1:x2]
        s = self.crop_max_side / max(crop.shape[:2])
        if s < 1.0:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.data = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        return out

    def _publish_overlay(self, bgr, dets, work_wh, header) -> None:
        H, W = bgr.shape[:2]
        items = []
        for d in dets:
            mask = (
                d.mask if (W, H) == work_wh else cv2.resize(d.mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            )
            sx, sy = W / work_wh[0], H / work_wh[1]
            x1, y1, x2, y2 = d.box_xyxy
            text = 'ray' if d.bearing_only else f'{d.median_depth:.1f}m n={d.point_count}'
            if np.isfinite(d.thermal_mean_c):
                text += f' {d.thermal_mean_c:.0f}C'
            items.append(
                {'label': d.label, 'score': d.score, 'box': (x1 * sx, y1 * sy, x2 * sx, y2 * sy), 'mask': mask, 'text': text}
            )
        out = draw_detections(bgr, items)
        self.pub_img.publish(ros_image.from_bgr(out, Image(), header.stamp, header.frame_id))

    def _publish_diag(self, vals: dict, level: int) -> None:
        st = DiagnosticStatus(level=level, name='x30_perception/perception_client', hardware_id='perception')
        st.message = 'ok' if level == DiagnosticStatus.OK else str(vals.get('error', 'degraded'))
        st.values = [KeyValue(key=k, value=(f'{v:.1f}' if isinstance(v, float) else str(v))) for k, v in vals.items()]
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status.append(st)
        self.pub_diag.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
