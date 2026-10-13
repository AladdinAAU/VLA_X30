"""Thermal image products from the Gazebo thermal camera.

Input : /thermal/raw   sensor_msgs/Image mono16 (Gazebo thermal_camera, L16: value * kelvin_per_count = temperature in Kelvin)
Output: /thermal/image_raw    mono8  normalised display image (hot = bright; percentile auto-gain or a fixed window)
        /thermal/temperature  32FC1  degrees Celsius per pixel
Frame and stamp are those of the input (thermal_optical_frame). camera_info comes straight from the sensor via the bridge.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from . import config as cfgmod

KELVIN = 273.15


def counts_to_celsius(raw: np.ndarray, kelvin_per_count: float) -> np.ndarray:
    return raw.astype(np.float32) * np.float32(kelvin_per_count) - np.float32(KELVIN)


def to_display(temp_c: np.ndarray, th: dict) -> np.ndarray:
    """mono8 display image from a Celsius image. Auto mode: 1..99 percentile window widened to at least display_min_span_c."""
    if th['display_auto']:
        lo, hi = np.percentile(temp_c, [1, 99])
        span = max(float(hi - lo), th['display_min_span_c'])
        centre = 0.5 * (float(hi) + float(lo))
        lo, hi = centre - span / 2, centre + span / 2
    else:
        lo, hi = th['display_min_c'], th['display_max_c']
    return np.clip((temp_c - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


class ThermalProcessor(Node):
    def __init__(self):
        super().__init__('thermal_processor')
        self.declare_parameter('robot_config', '')
        self.declare_parameter('kelvin_per_count', 0.01)
        cfg = cfgmod.load_robot_config(self.get_parameter('robot_config').value or None)
        self.th = cfg['thermal']
        self.kpc = float(self.get_parameter('kelvin_per_count').value)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=3)
        self.pub8 = self.create_publisher(Image, cfg['topics']['thermal'], qos)
        self.pubt = self.create_publisher(Image, cfg['topics']['thermal_temperature'], qos)
        self.create_subscription(Image, '/thermal/raw', self.on_image, qos)

    def on_image(self, msg: Image) -> None:
        if msg.encoding not in ('mono16', '16UC1'):
            self.get_logger().error(f'unexpected thermal encoding {msg.encoding}', throttle_duration_sec=5.0)
            return
        raw = np.frombuffer(bytes(msg.data), dtype='<u2' if not msg.is_bigendian else '>u2').reshape(msg.height, msg.width)
        temp = counts_to_celsius(raw, self.kpc)
        t = Image()
        t.header, t.height, t.width, t.encoding, t.is_bigendian = msg.header, msg.height, msg.width, '32FC1', False
        t.step = msg.width * 4
        t.data = temp.astype('<f4').tobytes()
        self.pubt.publish(t)
        d = Image()
        d.header, d.height, d.width, d.encoding, d.is_bigendian = msg.header, msg.height, msg.width, 'mono8', False
        d.step = msg.width
        d.data = to_display(temp, self.th).tobytes()
        self.pub8.publish(d)


def main(args=None):
    rclpy.init(args=args)
    node = ThermalProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
