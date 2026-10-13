"""ROS 2 node `semantic_mapper`: /perception/detections + /map + TF -> persistent semantic map, services, annotated map.

Topics and services follow INTERFACES.md section 4. Extra services (documented in the README): `render`, `feedback`,
`status`. All services are x30_msgs/srv/Json (JSON request string, `ok` + JSON response string).
"""
from __future__ import annotations

import base64
import json
import math
import time
import traceback
from typing import Any, Callable

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray
from x30_msgs.msg import Detection3DArray, SemanticObjectArray
from x30_msgs.srv import Json

from . import finder, places, relations, rooms, summary
from .annotated_map import MapRenderer, to_png
from .annotator import Annotator
from .config import load_config
from .datatypes import ViewGeometry
from .gridtools import GridView
from .ros_convert import (frame_from_msg, grid_from_msg, markers_msg, objects_msg, rgb_to_image, sec_to_stamp,
                          stamp_to_sec)
from .semantic_map import SemanticMap


def _quat_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class SemanticMapperNode(Node):
    def __init__(self) -> None:
        super().__init__('semantic_mapper')
        self.declare_parameter('config_file', '')
        self.declare_parameter('vlm_enabled', 'auto', ParameterDescriptor(dynamic_typing=True))   # auto | true | false
        self.declare_parameter('assumed_hfov_deg', 0.0)         # camera field of view when no camera_info arrives (0 = off)
        self.cfg = load_config(self.get_parameter('config_file').value)
        flag = str(self.get_parameter('vlm_enabled').value).lower()
        if flag in ('true', 'false'):
            self.cfg.vlm.enabled = flag == 'true'
        mc = self.cfg.mapper
        self.smap = SemanticMap(self.cfg)
        self.grid: GridView | None = None
        self.grid_version = 0
        self.rooms_version = -1
        self.last_rooms_t = -1e9
        self.user_rooms: list[dict] = []        # operator-defined rooms (service set_rooms), replace the automatic segmentation
        self.trajectory: list[tuple[float, float]] = []
        self.hfov = math.radians(float(self.get_parameter('assumed_hfov_deg').value)) or None
        self.vfov = None
        self.clock_offset = 0.0                                  # ros time - message stamp (robust to replayed stamps)
        self.last_frame_wall = 0.0
        self._warned_frame = False
        self.renderer = MapRenderer(self.cfg.render)
        self.smap.los_fn = self._los

        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.annotator = self._make_annotator()

        g_in = MutuallyExclusiveCallbackGroup()
        g_srv = ReentrantCallbackGroup()
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=20)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Detection3DArray, mc.detections_topic, self._on_detections, be, callback_group=g_in)
        self.create_subscription(OccupancyGrid, mc.map_topic, self._on_map, latched, callback_group=g_in)
        self.create_subscription(CameraInfo, mc.camera_info_topic, self._on_info, be, callback_group=g_in)
        self.pub_objects = self.create_publisher(SemanticObjectArray, '/semantic_map/objects', latched)
        self.pub_markers = self.create_publisher(MarkerArray, '/semantic_map/markers', 1)
        self.pub_asm = self.create_publisher(Image, '/semantic_map/annotated_map', 1)
        self.create_timer(mc.publish_period_s, self._tick_objects, callback_group=g_in)
        self.create_timer(mc.markers_period_s, self._tick_markers, callback_group=g_in)
        self.create_timer(mc.annotated_period_s, self._tick_asm, callback_group=g_in)
        self.create_timer(1.0, self._tick_rooms, callback_group=g_in)
        self.create_timer(0.5, self._tick_pose, callback_group=g_in)
        if mc.autosave_period_s > 0:
            self.create_timer(mc.autosave_period_s, lambda: self._autosave(), callback_group=g_in)

        handlers: dict[str, Callable[[dict], dict]] = {
            'query': self._svc_query, 'find': self._svc_find, 'relations': self._svc_relations,
            'summary': self._svc_summary, 'reset': self._svc_reset, 'save': self._svc_save, 'load': self._svc_load,
            'set_rooms': self._svc_set_rooms, 'add_place': self._svc_add_place, 'places': self._svc_places,
            'remove_place': self._svc_remove_place, 'place_context': self._svc_place_context, 'render': self._svc_render, 'feedback': self._svc_feedback, 'status': self._svc_status}
        for name, fn in handlers.items():
            self.create_service(Json, f'/semantic_map/{name}', self._wrap(name, fn), callback_group=g_srv)
        if self.annotator is not None:
            self.annotator.start()
        self.get_logger().info(
            f'semantic_mapper up: detections={mc.detections_topic} map={mc.map_topic} categories={len(self.cfg.categories)} '
            f'vlm={"on" if self.annotator and self.annotator.available else "off"} thermal={self.cfg.thermal.enabled}')

    # ------------------------------------------------------------------------------------ setup helpers
    def _make_annotator(self) -> Annotator | None:
        client = None
        if self.cfg.vlm.enabled:
            try:
                from x30_common.vlm_client import VLMClient
                client = VLMClient.from_config('mapper', max_retries=self.cfg.vlm.retries)
                self.get_logger().info(f'mapper VLM endpoint: {client.base_url} model {client.model}')
            except Exception as ex:                              # missing package or bad config: map still works
                self.get_logger().warning(f'mapper VLM disabled: {ex}')
        return Annotator(self.smap, self.cfg.vlm, client, pose_fn=self._pose, clock=time.time)

    def _wrap(self, name: str, fn: Callable[[dict], dict]):
        def handler(req: Json.Request, res: Json.Response) -> Json.Response:
            try:
                body = json.loads(req.request) if req.request.strip() else {}
                if not isinstance(body, dict):
                    raise ValueError('request must be a JSON object')
                out = fn(body)
                res.ok, res.response = True, json.dumps(out, default=_json_default)
            except Exception as ex:                              # a bad request must not kill the node
                self.get_logger().warning(f'/semantic_map/{name} failed: {ex}')
                if not isinstance(ex, (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError)):
                    self.get_logger().debug(traceback.format_exc())
                res.ok, res.response = False, json.dumps({'error': f'{type(ex).__name__}: {ex}'})
            return res
        return handler

    # ------------------------------------------------------------------------------------ inputs
    def _on_map(self, msg: OccupancyGrid) -> None:
        if abs(_quat_to_yaw(msg.info.origin.orientation)) > 1e-3 and not self._warned_frame:
            self.get_logger().warning('/map origin has a yaw; rotated grids are not supported (treated as yaw 0)')
            self._warned_frame = True
        self.grid = grid_from_msg(msg)
        self.grid_version += 1

    def _on_info(self, msg: CameraInfo) -> None:
        if msg.k[0] > 0 and msg.width > 0:
            self.hfov = 2.0 * math.atan(msg.width / (2.0 * msg.k[0]))
            self.vfov = 2.0 * math.atan(msg.height / (2.0 * msg.k[4])) if msg.k[4] > 0 else self.hfov * 0.75

    def _transform_to_map(self, frame_id: str, stamp) -> np.ndarray | None:
        if frame_id == self.cfg.mapper.map_frame:
            return None
        tf = self.buf.lookup_transform(self.cfg.mapper.map_frame, frame_id, stamp, timeout=rclpy.duration.Duration(seconds=0.1))
        t = np.eye(4)
        t[:3, :3] = _quat_to_matrix(tf.transform.rotation)
        t[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        return t

    def _on_detections(self, msg: Detection3DArray) -> None:
        try:
            transform = self._transform_to_map(msg.header.frame_id, msg.header.stamp)
        except TransformException as ex:
            self.get_logger().warning(f'detections in {msg.header.frame_id!r} dropped, no TF to map: {ex}', throttle_duration_sec=5.0)
            return
        frame = frame_from_msg(msg, transform)
        frame.view = self._view_geometry(msg)
        self.clock_offset = self._ros_now() - frame.stamp
        self.last_frame_wall = time.time()
        res = self.smap.process_frame(frame)
        if res.removed:
            self.get_logger().debug(f'removed objects {res.removed}')

    def _view_geometry(self, msg: Detection3DArray) -> ViewGeometry | None:
        if not self.cfg.mapper.use_view_misses or self.hfov is None:
            return None
        try:
            tf = self.buf.lookup_transform(self.cfg.mapper.map_frame, self.cfg.mapper.camera_frame, msg.image_stamp,
                                           timeout=rclpy.duration.Duration(seconds=0.05))
        except TransformException:
            return None
        r = _quat_to_matrix(tf.transform.rotation)
        pos = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        return ViewGeometry(pos, r @ np.array([0.0, 0.0, 1.0]), self.hfov, self.vfov or self.hfov * 0.75)

    def _los(self, a: np.ndarray, b: np.ndarray) -> bool:
        g = self.grid
        return True if g is None else g.line_of_sight(a, b, self.cfg.rooms.occupied_threshold)

    def _ros_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _map_time(self) -> float:
        """Current time in the stamps of the detections (survives bag replay with old stamps)."""
        return self._ros_now() - self.clock_offset if self.last_frame_wall else self.smap.now

    def _pose(self) -> tuple[float, float, float] | None:
        try:
            tf = self.buf.lookup_transform(self.cfg.mapper.map_frame, self.cfg.mapper.base_frame, rclpy.time.Time())
        except TransformException:
            return None
        t = tf.transform
        return float(t.translation.x), float(t.translation.y), _quat_to_yaw(t.rotation)

    # ------------------------------------------------------------------------------------ timers
    def _tick_pose(self) -> None:
        pose = self._pose()
        if pose is None:
            return
        step = self.cfg.render.trajectory_min_step_m
        if not self.trajectory or math.hypot(pose[0] - self.trajectory[-1][0], pose[1] - self.trajectory[-1][1]) >= step:
            self.trajectory.append((pose[0], pose[1]))
            del self.trajectory[:max(0, len(self.trajectory) - self.cfg.render.trajectory_max_points)]

    def _tick_objects(self) -> None:
        if self.last_frame_wall:
            self.smap.housekeeping(self._map_time())
        self.pub_objects.publish(objects_msg(self.smap, self.cfg.mapper.map_frame, self.get_clock().now().to_msg(),
                                             self.cfg.mapper.publish_tentative))

    def _tick_markers(self) -> None:
        if self.pub_markers.get_subscription_count() == 0:
            return
        self.pub_markers.publish(markers_msg(self.smap, self.cfg.mapper.map_frame, self.get_clock().now().to_msg(),
                                             self.trajectory, self.cfg.mapper.publish_tentative))

    def _tick_asm(self) -> None:
        if self.pub_asm.get_subscription_count() == 0:
            return
        img = self.renderer.render(self.smap, self.grid, self._pose(), self.trajectory,
                                   include_tentative=self.cfg.mapper.publish_tentative)
        self.pub_asm.publish(rgb_to_image(img, self.cfg.mapper.map_frame, self.get_clock().now().to_msg()))

    def _tick_rooms(self) -> None:
        if not self.cfg.rooms.enabled or self.grid is None:
            return
        now = time.time()
        if self.user_rooms:
            if self.grid_version != self.rooms_version or not self.smap.regions:
                self.rooms_version = self.grid_version
                rooms.apply_user_rooms(self.smap, self.grid, self.cfg.rooms, self.user_rooms)
            return
        if self.grid_version != self.rooms_version and now - self.last_rooms_t >= self.cfg.rooms.period_s:
            self.rooms_version, self.last_rooms_t = self.grid_version, now
            rooms.update_rooms(self.smap, self.grid, self.cfg.rooms)
            places.apply_places(self.smap)
        elif self.smap.regions:
            rooms.name_by_rules(self.smap, self.cfg.rooms)

    def _autosave(self) -> None:
        try:
            self.smap.save(self._path('autosave.json'))
        except OSError as ex:
            self.get_logger().warning(f'autosave failed: {ex}')

    def _path(self, name: str) -> str:
        from pathlib import Path
        p = Path(name)
        base = Path(self.cfg.mapper.save_dir) if self.cfg.mapper.save_dir else Path('.')
        return str(p if p.is_absolute() else base / p)

    # ------------------------------------------------------------------------------------ services
    @staticmethod
    def _pose_from(body: dict, default) -> tuple[float, float, float] | None:
        p = body.get('pose')
        if p is None:
            return default
        if len(p) == 2:
            return float(p[0]), float(p[1]), 0.0
        return float(p[0]), float(p[1]), float(p[2])

    def _svc_query(self, b: dict) -> dict:
        pose = self._pose()
        near = b.get('near')
        if near is None and pose is not None and b.get('radius') is not None:
            near = [pose[0], pose[1]]
        objs = self.smap.query(label=b.get('label'), near=tuple(near) if near else None, radius=b.get('radius'),
                               attributes=b.get('attributes'), room=b.get('room'), limit=int(b.get('limit', 10)),
                               include_tentative=bool(b.get('include_tentative', False)),
                               min_confidence=float(b.get('min_confidence', 0.0)))
        return {'count': len(objs), 'objects': [self.smap.describe(o, pose) for o in objs],
                'robot_pose': list(pose) if pose else None}

    def _svc_find(self, b: dict) -> dict:
        text = str(b.get('text') or b.get('label') or '').strip()
        if not text:
            raise ValueError('"text" is required')
        pose = self._pose()
        limit = int(b.get('limit', 5))
        if b.get('use_vlm', True) and self.annotator is not None:
            return self.annotator.find(text, pose, limit, b.get('timeout_s'))
        out = finder.rank(self.smap, text, pose, limit)
        out['vlm_calls'] = 0
        return out

    def _svc_relations(self, b: dict) -> dict:
        if 'id' not in b:
            raise ValueError('"id" is required')
        return relations.object_relations(self.smap, int(b['id']), self._pose_from(b, self._pose()), b.get('radius'),
                                          int(b.get('limit', 12)))

    def _svc_summary(self, b: dict) -> dict:
        return summary.build_summary(self.smap, self._pose_from(b, self._pose()), self.grid, b.get('max_objects'),
                                     b.get('token_budget'), b.get('include_tentative'))

    def _svc_reset(self, b: dict) -> dict:
        self.smap.reset()
        self.smap.set_rooms(None, {})
        self.trajectory.clear()
        self.rooms_version = -1
        if self.annotator is not None:
            self.annotator.cache.clear()
            self.annotator.failures.clear()
        return {'reset': True}

    def _svc_set_rooms(self, b: dict) -> dict:
        """{"rooms": [{"name": "Room A", "rect": [x0, y0, x1, y1]}, ...]} in the map frame; an empty list returns to automatic rooms."""
        self.user_rooms = [{'name': str(r['name']), 'rect': [float(v) for v in r['rect']]} for r in b.get('rooms', [])]
        self.rooms_version = -1
        if not self.user_rooms:
            self.smap.set_rooms(None, {})
        elif self.grid is not None:
            rooms.apply_user_rooms(self.smap, self.grid, self.cfg.rooms, self.user_rooms)
            self.rooms_version = self.grid_version
        return {'rooms': len(self.user_rooms)}

    def _svc_add_place(self, b: dict) -> dict:
        """{"name", "kind": room|spot, "x", "y" (map frame), "owner"?, "function"?, "aliases"?, "utterance"?} -> {'ok', 'action', 'place', 'note'}"""
        return places.add_place(self.smap, b['name'], str(b.get('kind', 'spot')), float(b['x']), float(b['y']), b.get('owner', ''),
                                b.get('function', ''), b.get('aliases') or [], b.get('utterance', ''))

    def _svc_places(self, b: dict) -> dict:
        return {'places': places.list_places(self.smap)}

    def _svc_remove_place(self, b: dict) -> dict:
        return places.remove_place(self.smap, b.get('name', b.get('id')))

    def _svc_place_context(self, b: dict) -> dict:
        return places.context_at(self.smap, float(b['x']), float(b['y']), float(b.get('radius', 4.0)))

    def _svc_save(self, b: dict) -> dict:
        path = self.smap.save(self._path(str(b.get('path') or 'semantic_map.json')))
        return {'path': str(path), 'objects': len(self.smap.objects)}

    def _svc_load(self, b: dict) -> dict:
        self.smap.load(self._path(str(b.get('path') or 'semantic_map.json')))
        self.rooms_version = -1
        return {'objects': len(self.smap.objects), 'regions': len(self.smap.regions)}

    def _svc_render(self, b: dict) -> dict:
        pose = self._pose()
        mode = b.get('mode', 'image')
        if mode == 'text':
            return {'text': self.renderer.render_text(self.smap, self.grid, pose, self.trajectory, b.get('cols'),
                                                      bool(b.get('include_tentative', False)))}
        img = self.renderer.render(self.smap, self.grid, pose, self.trajectory, b.get('egocentric'),
                                   b.get('show_captions'), b.get('max_size_px'), bool(b.get('include_tentative', False)))
        return {'format': 'png', 'width': int(img.shape[1]), 'height': int(img.shape[0]),
                'png_base64': base64.b64encode(to_png(img)).decode('ascii'), 'frame': self.cfg.mapper.map_frame,
                'robot_pose': list(pose) if pose else None}

    def _svc_feedback(self, b: dict) -> dict:
        oid, action = int(b['id']), str(b.get('action', ''))
        if action == 'reject':
            ok = self.smap.reject_object(oid, str(b.get('reason', 'feedback')))
        elif action == 'confirm':
            ok = self.smap.confirm_object(oid)
        elif action == 'relabel':
            ok = self.smap.relabel_object(oid, str(b['label']))
        elif action == 'merge':
            into = int(b['into'])
            ok = oid in self.smap.objects and into in self.smap.objects and oid != into
            if ok:
                self.smap.merge_objects(min(oid, into), max(oid, into))
        else:
            raise ValueError('action must be reject, confirm, relabel or merge')
        return {'applied': bool(ok)}

    def _svc_status(self, b: dict) -> dict:
        with self.smap.lock:
            by_status: dict[str, int] = {}
            for o in self.smap.objects.values():
                s = self.smap.status(o)
                by_status[s] = by_status.get(s, 0) + 1
            return {'objects': by_status, 'regions': {r.id: r.name for r in self.smap.regions.values()},
                    'rays': len(self.smap.rays), 'stats': dict(self.smap.stats), 'grid': self.grid is not None,
                    'trajectory_points': len(self.trajectory), 'annotator': self.annotator.status() if self.annotator else None}


def _json_default(o: Any):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticMapperNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.annotator is not None:
            node.annotator.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
