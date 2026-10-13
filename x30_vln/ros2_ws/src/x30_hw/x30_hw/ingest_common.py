"""Shared node for camera_ingest and thermal_ingest: stream -> sensor_msgs/Image + CameraInfo."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from x30_hw import config as cfgmod
from x30_hw.rtsp_source import FrameReader


def load_intrinsics(path: str) -> Optional[Dict[str, Any]]:
    """Read a calibration file written by calibrate_camera (OpenCV/ROS camera_info layout). None when missing."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def placeholder_intrinsics(width: int, height: int, hfov_deg: float) -> Dict[str, Any]:
    import math
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return {"image_width": width, "image_height": height, "distortion_model": "",
            "camera_matrix": {"rows": 3, "cols": 3, "data": [fx, 0, width / 2.0, 0, fx, height / 2.0, 0, 0, 1]},
            "distortion_coefficients": {"rows": 1, "cols": 0, "data": []}, "calibrated": False}


def make_camera_info(intr: Dict[str, Any], width: int, height: int, frame_id: str) -> CameraInfo:
    """CameraInfo for a frame of `width`x`height`; K/P are scaled when the calibration size differs (same aspect only)."""
    ci = CameraInfo()
    ci.header.frame_id = frame_id
    ci.width, ci.height = width, height
    k = list(intr["camera_matrix"]["data"])
    sx = width / float(intr["image_width"])
    sy = height / float(intr["image_height"])
    k = [k[0] * sx, k[1], k[2] * sx, k[3], k[4] * sy, k[5] * sy, 0.0, 0.0, 1.0]
    ci.k = [float(x) for x in k]
    ci.d = [float(x) for x in intr["distortion_coefficients"]["data"]]
    ci.distortion_model = intr.get("distortion_model", "plumb_bob") if ci.d else ""
    ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.p = [float(x) for x in (k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0)]
    return ci


class IngestNode(Node):
    def __init__(self, node_name: str, section: str, image_topic: str, info_topic: str, mono: bool):
        super().__init__(node_name)
        self.declare_parameter("config_file", "")
        self.declare_parameter("url", "")
        cfg = cfgmod.load_robot_config(self.get_parameter("config_file").value or None)
        c = cfg[section]
        self.c = c
        self.mono = mono
        url = self.get_parameter("url").value or c.get("url", "")
        self.enabled = bool(c.get("enabled", False)) and bool(url)
        if not self.enabled:
            self.get_logger().info(f"{node_name}: not configured (enabled={c.get('enabled')}, url='{url}'): nothing to do")
            return
        self.frame_id = c["frame_id"]
        self.offset = float(cfg["clock"]["robot_minus_pc_s"]) + float(c.get("stamp_offset_s", 0.0))
        self.period = 1.0 / float(c.get("publish_rate_hz_max", 10.0))
        path = os.path.join(cfgmod.configs_dir(), c["intrinsics_file"]) if c.get("intrinsics_file") else ""
        self.intr = load_intrinsics(path)
        if self.intr is None:
            self.get_logger().warn(f"no calibration file '{path}': publishing PLACEHOLDER intrinsics (empty distortion model, "
                                   f"hfov {c.get('placeholder_hfov_deg', 100)} deg). Run calibrate_camera before using projections.")
        else:
            self.get_logger().info(f"intrinsics from {path}")
        self.pub_img = self.create_publisher(Image, image_topic, qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, info_topic, qos_profile_sensor_data)
        rc = c.get("reconnect", {})
        self.reader = FrameReader(url, c.get("backend", "ffmpeg"), c.get("transport", "tcp"), rc.get("initial_s", 1.0),
                                  rc.get("max_s", 10.0), log=lambda m: self.get_logger().info(m)).start()
        self.published = 0
        self.last_report = time.time()
        self.create_timer(min(self.period / 2.0, 0.02), self.tick)
        self._last_pub = 0.0

    def tick(self) -> None:
        now = time.time()
        if now - self._last_pub < self.period * 0.9:
            return
        f = self.reader.latest()
        if f is None:
            if now - self.last_report > 10.0:
                self.last_report = now
                self.get_logger().warn(f"no new frame (connected={self.reader.connected}, reconnects={self.reader.reconnects})")
            return
        self._last_pub = now
        img = f.image
        h, w = img.shape[:2]
        if self.mono:
            gray = img if img.ndim == 2 else (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299).astype(np.uint8)
            if self.c.get("invert", False):
                gray = 255 - gray
            out, enc, step = np.ascontiguousarray(gray), "mono8", w
        else:
            out, enc, step = np.ascontiguousarray(img[..., ::-1]), "rgb8", w * 3
        # stamp at arrival (PC clock), moved into the robot clock domain by the configured offsets
        stamp = (Time(nanoseconds=int(f.arrival_wall * 1e9)) + Duration(seconds=self.offset)).to_msg()
        msg = Image()
        msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
        msg.height, msg.width, msg.encoding, msg.step, msg.is_bigendian = h, w, enc, step, 0
        msg.data = out.tobytes()
        self.pub_img.publish(msg)
        intr = self.intr or placeholder_intrinsics(w, h, float(self.c.get("placeholder_hfov_deg", 100.0)))
        info = make_camera_info(intr, w, h, self.frame_id)
        info.header.stamp = stamp
        self.pub_info.publish(info)
        self.published += 1
        if now - self.last_report > 10.0:
            self.last_report = now
            self.get_logger().info(f"published {self.published}, read {self.reader.frames_read}, dropped late "
                                   f"{self.reader.frames_dropped_late}, reconnects {self.reader.reconnects}")

    def destroy_node(self):
        if getattr(self, "reader", None):
            self.reader.stop()
        return super().destroy_node()


def run(node_cls, args=None) -> None:
    rclpy.init(args=args)
    node = node_cls()
    try:
        if node.enabled:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
