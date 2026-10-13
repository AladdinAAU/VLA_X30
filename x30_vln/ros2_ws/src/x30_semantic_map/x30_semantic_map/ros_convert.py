"""Conversions between ROS messages and the pure-python core types (needs ROS)."""
from __future__ import annotations

import math

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import Marker, MarkerArray
from x30_msgs.msg import Detection3DArray, SemanticObject, SemanticObjectArray

from .datatypes import Detection, Frame
from .gridtools import GridView
from .semantic_map import MapObject, SemanticMap


def stamp_to_sec(t: TimeMsg) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9


def sec_to_stamp(s: float) -> TimeMsg:
    t = TimeMsg()
    t.sec = int(math.floor(s))
    t.nanosec = int(round((s - math.floor(s)) * 1e9))
    if t.nanosec >= 1_000_000_000:
        t.sec += 1
        t.nanosec -= 1_000_000_000
    return t


def _p(p) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=float)


def frame_from_msg(msg: Detection3DArray, transform: np.ndarray | None = None) -> Frame:
    """Detection3DArray -> Frame. `transform` (4x4) maps msg.header.frame_id into the map frame (None = already there)."""
    def pt(v: np.ndarray) -> np.ndarray:
        return v if transform is None else transform[:3, :3] @ v + transform[:3, 3]

    def vec(v: np.ndarray) -> np.ndarray:
        return v if transform is None else transform[:3, :3] @ v

    dets = []
    for d in msg.detections:
        dets.append(Detection(
            label=d.label, score=float(d.score), centroid=pt(_p(d.centroid)), extent=_p(d.extent),
            bearing=vec(_p(d.bearing)), bearing_only=bool(d.bearing_only), point_count=int(d.point_count),
            thermal_mean_c=float(d.thermal_mean_c), thermal_max_c=float(d.thermal_max_c),
            crop=bytes(d.crop.data), box_xyxy=tuple(int(x) for x in d.box_xyxy), depth_source=d.depth_source))
    stamp = stamp_to_sec(msg.image_stamp) if (msg.image_stamp.sec or msg.image_stamp.nanosec) else \
        stamp_to_sec(msg.header.stamp)
    return Frame(stamp=stamp, camera_position=pt(_p(msg.camera_position)), detections=dets)


def grid_from_msg(msg: OccupancyGrid) -> GridView:
    data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    return GridView(data, float(msg.info.resolution), float(msg.info.origin.position.x), float(msg.info.origin.position.y))


def object_to_msg(smap: SemanticMap, o: MapObject) -> SemanticObject:
    m = SemanticObject()
    m.id = int(o.id)
    m.label = smap.label(o)
    m.confidence = float(smap.confidence(o))
    m.position = Point(x=float(o.position[0]), y=float(o.position[1]), z=float(o.position[2]))
    m.size = Vector3(x=float(o.size[0]), y=float(o.size[1]), z=float(o.size[2]))
    m.point_count = int(o.point_count)
    m.observations = int(o.n_obs)
    m.first_seen = sec_to_stamp(o.first_seen)
    m.last_seen = sec_to_stamp(o.last_seen)
    m.caption = o.caption
    m.attributes = smap.attribute_list(o)
    m.room = smap.room_name_of(o)
    m.bearing_only = bool(o.bearing_only)
    if o.views:
        m.best_view = CompressedImage()
        m.best_view.format = 'jpeg'
        m.best_view.data = o.views[0].jpeg
    return m


def objects_msg(smap: SemanticMap, frame_id: str, stamp: TimeMsg, include_tentative: bool) -> SemanticObjectArray:
    arr = SemanticObjectArray()
    arr.header.frame_id = frame_id
    arr.header.stamp = stamp
    with smap.lock:
        arr.objects = [object_to_msg(smap, o) for o in smap.visible_objects(include_tentative)]
    return arr


def rgb_to_image(rgb: np.ndarray, frame_id: str, stamp: TimeMsg) -> Image:
    img = Image()
    img.header.frame_id = frame_id
    img.header.stamp = stamp
    img.height, img.width = int(rgb.shape[0]), int(rgb.shape[1])
    img.encoding = 'rgb8'
    img.is_bigendian = 0
    img.step = int(rgb.shape[1] * 3)
    img.data = np.ascontiguousarray(rgb).tobytes()
    return img


def markers_msg(smap: SemanticMap, frame_id: str, stamp: TimeMsg, trajectory: list[tuple[float, float]],
                include_tentative: bool) -> MarkerArray:
    arr = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    with smap.lock:
        for o in smap.visible_objects(include_tentative):
            r, g, b = (c / 255.0 for c in smap.vocab.color(smap.label(o)))
            status = smap.status(o)
            alpha = 0.55 if status == 'confirmed' else 0.3
            cube = Marker()
            cube.header.frame_id, cube.header.stamp = frame_id, stamp
            cube.ns, cube.id = 'objects', o.id
            cube.type = Marker.SPHERE if o.bearing_only else Marker.CUBE
            cube.pose.position = Point(x=float(o.position[0]), y=float(o.position[1]), z=float(o.position[2]))
            cube.pose.orientation.w = 1.0
            s = np.maximum(o.size, 0.12)
            cube.scale = Vector3(x=float(s[0]), y=float(s[1]), z=float(s[2]))
            cube.color.r, cube.color.g, cube.color.b, cube.color.a = r, g, b, alpha
            arr.markers.append(cube)
            txt = Marker()
            txt.header.frame_id, txt.header.stamp = frame_id, stamp
            txt.ns, txt.id = 'labels', o.id
            txt.type = Marker.TEXT_VIEW_FACING
            txt.pose.position = Point(x=float(o.position[0]), y=float(o.position[1]),
                                      z=float(o.position[2] + s[2] / 2 + 0.2))
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.18
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f'#{o.id} {smap.label(o)}' + ('?' if o.bearing_only else '')
            arr.markers.append(txt)
            if o.moving:
                v = Marker()
                v.header.frame_id, v.header.stamp = frame_id, stamp
                v.ns, v.id = 'velocity', o.id
                v.type = Marker.ARROW
                v.scale = Vector3(x=0.05, y=0.1, z=0.1)
                v.color.r, v.color.a = 1.0, 0.9
                v.points = [Point(x=float(o.position[0]), y=float(o.position[1]), z=float(o.position[2])),
                            Point(x=float(o.position[0] + o.velocity[0]), y=float(o.position[1] + o.velocity[1]),
                                  z=float(o.position[2]))]
                arr.markers.append(v)
        for r in smap.regions.values():
            m = Marker()
            m.header.frame_id, m.header.stamp = frame_id, stamp
            m.ns, m.id = 'rooms', int(r.id)
            m.type = Marker.TEXT_VIEW_FACING
            m.pose.position = Point(x=float(r.centroid[0]), y=float(r.centroid[1]), z=2.2)
            m.pose.orientation.w = 1.0
            m.scale.z = 0.35
            m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.9, 0.3, 1.0
            m.text = r.name or f'area {r.id}'
            arr.markers.append(m)
    if len(trajectory) > 1:
        line = Marker()
        line.header.frame_id, line.header.stamp = frame_id, stamp
        line.ns, line.id = 'trajectory', 0
        line.type = Marker.LINE_STRIP
        line.scale.x = 0.04
        line.color.b, line.color.g, line.color.a = 1.0, 0.6, 0.9
        line.pose.orientation.w = 1.0
        line.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in trajectory[-1500:]]
        arr.markers.append(line)
    return arr
