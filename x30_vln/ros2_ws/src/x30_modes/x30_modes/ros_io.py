"""ROS plumbing shared by the guard and the skill server: clocks, TF poses, obstacle points in base_link, JSON service calls."""
from __future__ import annotations

import dataclasses
import json
import math
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from .cloud import band_filter_xy, cloud_xyz, quat_to_matrix, scan_xy, transform_points
from .geometry import Pose2D, yaw_from_quat

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
LATCHED_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)


def now_s(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def declare_dataclass_params(node: Node, cfg_obj: Any) -> Any:
    """Declare one ROS parameter per dataclass field (defaults from the dataclass) and return a filled copy."""
    values = {}
    for f in dataclasses.fields(cfg_obj):
        default = getattr(cfg_obj, f.name)
        values[f.name] = node.declare_parameter(f.name, default).value
        if isinstance(default, float) and isinstance(values[f.name], int):
            values[f.name] = float(values[f.name])
    return dataclasses.replace(cfg_obj, **values)


def lookup_pose(buf: Buffer, target: str, source: str) -> Pose2D | None:
    """Latest pose of `source` in `target` as (x, y, yaw), or None while the transform is unavailable."""
    try:
        t = buf.lookup_transform(target, source, Time())
    except TransformException:
        return None
    q = t.transform.rotation
    return Pose2D(t.transform.translation.x, t.transform.translation.y, yaw_from_quat(q.x, q.y, q.z, q.w))


class ObstacleSource:
    """Latest obstacle points from /costmap_cloud and/or /scan, delivered in base_link at the time of the call.

    Points are stored in their source frame and transformed with the newest TF on every `points()` call, so an
    obstacle map expressed in `odom` or `map` stays exact relative to the current robot pose between sensor updates.
    The age of a source is its arrival time on the node clock (independent of header stamps and bag replay offsets).
    """

    def __init__(self, node: Node, buf: Buffer, base_frame: str, source: str, cloud_topic: str, scan_topic: str,
                 z_min: float, z_max: float, max_points: int = 6000, callback_group=None, drop_zero_intensity: bool = True):
        self.node, self.buf, self.base_frame = node, buf, base_frame
        self.z_min, self.z_max, self.max_points = z_min, z_max, max_points
        self.drop_zero = drop_zero_intensity
        self._lock = threading.Lock()
        self._data: dict[str, tuple[np.ndarray, str, float, bool]] = {}     # name -> (xyz in frame, frame, arrival time, is_scan)
        if source in ("costmap_cloud", "both"):
            node.create_subscription(PointCloud2, cloud_topic, self._on_cloud, SENSOR_QOS, callback_group=callback_group)
        if source in ("scan", "both"):
            node.create_subscription(LaserScan, scan_topic, self._on_scan, SENSOR_QOS, callback_group=callback_group)

    def _on_cloud(self, msg: PointCloud2) -> None:
        try:
            xyz = cloud_xyz(msg, self.drop_zero)
        except ValueError as ex:
            self.node.get_logger().warn(f"obstacle cloud ignored: {ex}", throttle_duration_sec=10.0)
            return
        if xyz.shape[0] > 4 * self.max_points:
            xyz = xyz[:: int(math.ceil(xyz.shape[0] / (4 * self.max_points)))]
        with self._lock:
            self._data["cloud"] = (xyz, msg.header.frame_id, now_s(self.node), False)

    def _on_scan(self, msg: LaserScan) -> None:
        xy = scan_xy(msg)
        xyz = np.concatenate([xy, np.zeros((xy.shape[0], 1), np.float32)], axis=1)
        with self._lock:
            self._data["scan"] = (xyz, msg.header.frame_id, now_s(self.node), True)

    def points(self, max_age: float) -> tuple[np.ndarray | None, float | None]:
        """(points (N, 2) in base_link from all sources younger than max_age, newest arrival time) or (None, None)."""
        now = now_s(self.node)
        with self._lock:
            items = list(self._data.values())
        parts, newest = [], None
        for xyz, frame, stamp, is_scan in items:
            if now - stamp > max_age:
                continue
            pts = self._to_base(xyz, frame, is_scan)
            if pts is None:
                continue
            parts.append(pts)
            newest = stamp if newest is None else max(newest, stamp)
        if not parts:
            return None, None
        return np.concatenate(parts, axis=0), newest

    def _to_base(self, xyz: np.ndarray, frame: str, scan: bool) -> np.ndarray | None:
        if frame and frame != self.base_frame:
            try:
                t = self.buf.lookup_transform(self.base_frame, frame, Time())
            except TransformException as ex:
                self.node.get_logger().warn(f"no transform {frame}->{self.base_frame}: {ex}", throttle_duration_sec=5.0)
                return None
            q, tr = t.transform.rotation, t.transform.translation
            xyz = transform_points(xyz, quat_to_matrix(q.x, q.y, q.z, q.w), (tr.x, tr.y, tr.z))
        if scan:                       # a laser scan is a height slice: its z is meaningless after the transform
            return xyz[:, :2].astype(np.float32)
        return band_filter_xy(xyz, self.z_min, self.z_max, self.max_points)


def call_json(node: Node, client, payload: dict, timeout: float, cancel: threading.Event | None = None) -> dict | None:
    """Call an x30_msgs/srv/Json service. Returns the parsed response dict (with 'ok' key) or None if unavailable/failed."""
    from x30_msgs.srv import Json
    if not client.wait_for_service(timeout_sec=min(timeout, 1.0)):
        return None
    req = Json.Request()
    req.request = json.dumps(payload)
    fut = client.call_async(req)
    t_end = time.monotonic() + timeout
    while not fut.done():
        if time.monotonic() > t_end or (cancel is not None and cancel.is_set()):
            return None
        time.sleep(0.02)
    res = fut.result()
    if res is None:
        return None
    try:
        body = json.loads(res.response) if res.response else {}
    except json.JSONDecodeError:
        body = {"raw": res.response}
    if isinstance(body, list):
        body = {"objects": body}
    body["ok"] = bool(res.ok) and body.get("ok", True)
    return body
