"""Everything a skill needs from the robot and its neighbours, behind one object (so skills stay readable and testable)."""
from __future__ import annotations

import threading
import time

import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from x30_msgs.msg import SafetyStatus, SemanticObjectArray
from x30_msgs.srv import Json

from .geometry import GridMap, Pose2D
from .ros_io import LATCHED_QOS, SENSOR_QOS, ObstacleSource, call_json, lookup_pose, now_s
from .semantic import ObjectRecord, parse_query_response, record_from_msg
from .skills.base import SkillConfig


class RobotAPI:
    def __init__(self, node: Node, cfg: SkillConfig, group):
        self.node, self.cfg = node, cfg
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, node)
        self._lock = threading.Lock()
        self._objects: dict[int, ObjectRecord] = {}
        self._views: dict[int, bytes] = {}
        self._objects_stamp: float | None = None
        self._grid: GridMap | None = None
        self._costmap: tuple[GridMap, float] | None = None
        self._rgb: tuple[Image, float] | None = None
        self._safety: tuple[SafetyStatus, float] | None = None
        self._blocked_since: float | None = None
        self.pub_cmd = node.create_publisher(Twist, cfg.nav_cmd_topic, 10)
        self.obstacles = ObstacleSource(node, self.buf, cfg.base_frame, cfg.obstacle_source, cfg.costmap_cloud_topic, cfg.scan_topic,
                                        cfg.z_min, cfg.z_max, callback_group=group)
        node.create_subscription(SafetyStatus, "/x30/safety", self._on_safety, 10, callback_group=group)
        node.create_subscription(SemanticObjectArray, cfg.objects_topic, self._on_objects, LATCHED_QOS, callback_group=group)
        node.create_subscription(OccupancyGrid, cfg.map_topic, self._on_map, LATCHED_QOS, callback_group=group)
        node.create_subscription(OccupancyGrid, cfg.costmap_topic, self._on_costmap, 1, callback_group=group)
        node.create_subscription(Image, cfg.rgb_topic, self._on_rgb, SENSOR_QOS, callback_group=group)
        self.c_frontier = node.create_client(Json, cfg.frontier_service, callback_group=group)
        self.c_query = node.create_client(Json, cfg.query_service, callback_group=group)
        self.c_summary = node.create_client(Json, cfg.summary_service, callback_group=group)

    # ------------------------------------------------------------ time and poses
    def now(self) -> float:
        return now_s(self.node)

    def odom_pose(self) -> Pose2D | None:
        return lookup_pose(self.buf, self.cfg.odom_frame, self.cfg.base_frame)

    def map_pose(self) -> Pose2D | None:
        return lookup_pose(self.buf, self.cfg.map_frame, self.cfg.base_frame)

    # ------------------------------------------------------------ commands and safety
    def send_cmd(self, vx: float, vy: float, wz: float) -> None:
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(wz)
        self.pub_cmd.publish(m)

    def halt(self, repeat: int = 3) -> None:
        for _ in range(repeat):
            self.send_cmd(0.0, 0.0, 0.0)
            time.sleep(0.02)

    def _on_safety(self, m: SafetyStatus) -> None:
        now = self.now()
        with self._lock:
            self._safety = (m, now)
            if m.stopped and m.reason in ("obstacle", "no_lidar", "estop", "tilt"):
                if self._blocked_since is None:
                    self._blocked_since = now
            else:
                self._blocked_since = None

    def safety(self) -> SafetyStatus | None:
        with self._lock:
            return self._safety[0] if self._safety and self.now() - self._safety[1] < 2.0 else None

    def blocked_for(self) -> float:
        """Seconds the safety guard has continuously reported a stop for a real reason (0 if not blocked)."""
        with self._lock:
            return 0.0 if self._blocked_since is None else self.now() - self._blocked_since

    def obstacle_points(self) -> np.ndarray | None:
        pts, _ = self.obstacles.points(self.cfg.obstacle_max_age_s)
        return pts

    # ------------------------------------------------------------ maps
    @staticmethod
    def _grid_from_msg(m: OccupancyGrid) -> GridMap:
        data = np.asarray(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
        return GridMap(data, float(m.info.resolution), float(m.info.origin.position.x), float(m.info.origin.position.y))

    def _on_map(self, m: OccupancyGrid) -> None:
        g = self._grid_from_msg(m)
        with self._lock:
            self._grid = g

    def _on_costmap(self, m: OccupancyGrid) -> None:
        g = self._grid_from_msg(m)
        with self._lock:
            self._costmap = (g, self.now())

    def grid(self) -> GridMap | None:
        """Static map if present, else the latest global costmap."""
        with self._lock:
            if self._grid is not None:
                return self._grid
            return self._costmap[0] if self._costmap else None

    # ------------------------------------------------------------ semantic objects
    def _on_objects(self, m: SemanticObjectArray) -> None:
        recs, views = {}, {}
        for o in m.objects:
            r = record_from_msg(o)
            recs[r.id] = r
            if o.best_view.data:
                views[r.id] = bytes(o.best_view.data)
        with self._lock:
            self._objects, self._views, self._objects_stamp = recs, views, self.now()

    def objects(self) -> list[ObjectRecord]:
        with self._lock:
            return list(self._objects.values())

    def have_object_source(self) -> bool:
        return self._objects_stamp is not None or self.c_query.service_is_ready()

    def get_object(self, oid: int) -> ObjectRecord | None:
        with self._lock:
            o = self._objects.get(int(oid))
        if o is not None:
            return o
        res = call_json(self.node, self.c_query, {"id": int(oid)}, timeout=2.0)
        if res and res.get("ok"):
            try:
                recs = parse_query_response(_dump(res))
            except (ValueError, KeyError, TypeError):
                return None
            for r in recs:
                if r.id == int(oid):
                    return r
        return None

    def object_view(self, oid: int) -> bytes | None:
        with self._lock:
            return self._views.get(int(oid))

    def query_objects(self, label: str | None = None, near: tuple[float, float] | None = None, radius: float | None = None,
                      limit: int = 20) -> list[ObjectRecord]:
        """Map query through /semantic_map/query; falls back to the cached /semantic_map/objects topic."""
        payload: dict = {"limit": limit}
        if label:
            payload["label"] = label
        if near is not None:
            payload["near"] = [near[0], near[1]]
            if radius:
                payload["radius"] = radius
        res = call_json(self.node, self.c_query, payload, timeout=3.0)
        if res and res.get("ok"):
            try:
                return parse_query_response(_dump(res))
            except (ValueError, KeyError, TypeError) as ex:
                self.node.get_logger().warn(f"unparseable /semantic_map/query response ({ex}); using the objects topic",
                                            throttle_duration_sec=30.0)
        return self.objects()

    # ------------------------------------------------------------ camera
    def _on_rgb(self, m: Image) -> None:
        with self._lock:
            self._rgb = (m, self.now())

    def latest_rgb(self, max_age: float = 3.0) -> Image | None:
        with self._lock:
            if self._rgb and self.now() - self._rgb[1] <= max_age:
                return self._rgb[0]
        return None

    # ------------------------------------------------------------ services
    def next_frontier(self, payload: dict, cancel: threading.Event | None = None) -> dict | None:
        return call_json(self.node, self.c_frontier, payload, timeout=15.0, cancel=cancel)

    def summary(self, pose: Pose2D, max_objects: int = 20) -> str:
        res = call_json(self.node, self.c_summary, {"pose": [pose.x, pose.y, pose.yaw], "max_objects": max_objects}, timeout=3.0)
        if res and res.get("ok"):
            return str(res.get("text") or res.get("summary") or "")
        return ""


def _dump(res: dict) -> str:
    import json
    body = {k: v for k, v in res.items() if k != "ok"}
    return json.dumps(body)
