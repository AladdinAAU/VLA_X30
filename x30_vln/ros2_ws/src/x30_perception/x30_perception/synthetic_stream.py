"""ROS 2 node that publishes a synthetic robot-facing sensor stream for testing the perception chain without the simulator.

World: a room with upright photo billboards (zidane.jpg: people, bus.jpg: a bus, from test/assets), so that the real
open-vocabulary model detects them, while their 3D position is known exactly. The robot walks a small figure around the
start pose. Published (INTERFACES.md section 2 names and formats):
  /lidar_points (PointCloud2, 26-byte Livox layout, 10 Hz, per-point timestamps in ms), /camera/color/image_raw (rgb8),
  /camera/color/camera_info, /camera/depth/image_raw (32FC1, optional ground truth), /thermal/image_raw (mono8),
  /thermal/temperature (32FC1), /thermal/camera_info, TF map->odom (identity), odom->base_link (moving),
  base_link->lidar_link / camera_optical_frame / thermal_optical_frame (static),
  /synthetic/ground_truth (std_msgs/String JSON, latched) with the billboards in the map frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from . import synthetic as syn
from .camera_model import CameraModel
from .geometry import make_transform, matrix_to_quat

LIDAR_DTYPE = np.dtype(
    {
        'names': ['x', 'y', 'z', 'intensity', 'tag', 'line', 'timestamp'],
        'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
        'offsets': [0, 4, 8, 12, 16, 17, 18],
        'itemsize': 26,
    }
)
T_BASE_LIDAR = make_transform(np.eye(3), (0.0, 0.0, 0.7))
T_BASE_CAM = syn.optical_pose((0.5, 0.0, 0.6))
T_BASE_THERMAL = syn.optical_pose((0.5, 0.04, 0.6))


def _tf(stamp, parent, child, T) -> TransformStamped:
    m = TransformStamped()
    m.header.stamp, m.header.frame_id, m.child_frame_id = stamp, parent, child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in T[:3, 3])
    q = matrix_to_quat(T[:3, :3])
    m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = (float(v) for v in q)
    return m


def _info(cam: CameraModel, frame: str, stamp) -> CameraInfo:
    m = CameraInfo()
    m.header.stamp, m.header.frame_id = stamp, frame
    m.width, m.height = cam.width, cam.height
    m.distortion_model = 'plumb_bob'
    m.d = [0.0] * 5
    m.k = [float(v) for v in cam.K.ravel()]
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    m.p = [cam.fx, 0.0, cam.cx, 0.0, 0.0, cam.fy, cam.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return m


class SyntheticStream(Node):
    def __init__(self):
        super().__init__('synthetic_stream')
        p = self.declare_parameter
        assets = Path(p('assets_dir', '').value or Path(__file__).resolve().parents[1] / 'test' / 'assets')
        self.lidar_points = int(p('lidar_points', 60000).value)
        self.image_rate = float(p('image_rate_hz', 3.0).value)
        self.walk = bool(p('walk', True).value)
        self.thermal = bool(p('thermal', True).value)
        self.publish_depth = bool(p('publish_depth', True).value)
        self.cam = CameraModel.from_fov(640, 480, 1.5184)  # as the Gazebo prework camera
        self.cam_th = CameraModel.from_fov(160, 120, np.radians(60))
        person, bus = cv2.imread(str(assets / 'zidane.jpg')), cv2.imread(str(assets / 'bus.jpg'))
        if person is None or bus is None:
            raise RuntimeError(f'texture images zidane.jpg / bus.jpg not found in {assets}')
        self.temps = {}
        self.scene = syn.Scene(
            [
                syn.Billboard('person', (3.0, 0.9, 1.0), 2.0, 1.125, texture=person),
                syn.Billboard('bus', (6.0, -1.3, 1.45), 2.0, 2.67, texture=bus),
                syn.Billboard('person', (4.6, -2.4, 1.0), 2.0, 1.125, texture=person),
            ]
        )
        self.temps = {0: 34.0, 1: 21.0, 2: 34.0}  # surface temperature by billboard index
        self.rng = np.random.default_rng(0)
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

        self.pub_lidar = self.create_publisher(PointCloud2, '/lidar_points', qos_profile_sensor_data)
        self.pub_rgb = self.create_publisher(Image, '/camera/color/image_raw', qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/color/camera_info', qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image_raw', qos_profile_sensor_data)
        self.pub_th = self.create_publisher(Image, '/thermal/image_raw', qos_profile_sensor_data)
        self.pub_temp = self.create_publisher(Image, '/thermal/temperature', qos_profile_sensor_data)
        self.pub_th_info = self.create_publisher(CameraInfo, '/thermal/camera_info', qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_gt = self.create_publisher(String, '/synthetic/ground_truth', latched)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        now = self.get_clock().now().to_msg()
        self.static_tf.sendTransform(
            [
                _tf(now, 'base_link', 'lidar_link', T_BASE_LIDAR),
                _tf(now, 'base_link', 'camera_optical_frame', T_BASE_CAM),
                _tf(now, 'base_link', 'thermal_optical_frame', T_BASE_THERMAL),
                _tf(now, 'map', 'odom', np.eye(4)),
            ]
        )
        self.pub_gt.publish(
            String(
                data=json.dumps(
                    {
                        'frame': 'map',
                        'objects': [
                            {
                                'id': i,
                                'label': b.label,
                                'center': list(b.center),
                                'width': b.width,
                                'height': b.height,
                                'normal': b.normal.tolist(),
                                'temp_c': self.temps[i],
                            }
                            for i, b in enumerate(self.scene.billboards)
                        ],
                    }
                )
            )
        )
        # separate callback groups: rendering takes ~0.2 s and must not delay the TF stream
        self.create_timer(0.02, self._publish_tf, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(0.1, self._publish_scan, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0 / self.image_rate, self._publish_images, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f'synthetic stream running: {len(self.scene.billboards)} billboards, lidar {self.lidar_points} pts/scan'
        )

    # ------------------------------------------------------------ motion (analytic, so any stamp can be evaluated)
    def base_pose(self, t: float) -> np.ndarray:
        if not self.walk:
            return np.eye(4)
        s = t - self.t0
        yaw = 0.15 * np.sin(0.5 * s)
        x, y = 0.4 * np.sin(0.3 * s), 0.25 * np.sin(0.4 * s + 1.0)
        c, sn = np.cos(yaw), np.sin(yaw)
        return make_transform(np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]]), (x, y, 0.0))

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_tf(self) -> None:
        now = self.get_clock().now()
        self.tf.sendTransform(_tf(now.to_msg(), 'odom', 'base_link', self.base_pose(now.nanoseconds * 1e-9)))

    # ------------------------------------------------------------ sensors
    def _publish_scan(self) -> None:
        t_end = self._now()
        t0 = t_end - 0.1
        scan = syn.simulate_scan(self.scene, lambda t: self.base_pose(t) @ T_BASE_LIDAR, t0, self.lidar_points, rng=self.rng)
        n = scan.xyz.shape[0]
        rec = np.zeros(n, LIDAR_DTYPE)
        rec['x'], rec['y'], rec['z'] = scan.xyz[:, 0], scan.xyz[:, 1], scan.xyz[:, 2]
        rec['intensity'] = 50.0
        rec['line'] = np.arange(n) % 16
        rec['timestamp'] = (t0 + scan.t_rel) * 1e3
        msg = PointCloud2()
        sec = int(t0)
        msg.header.stamp.sec, msg.header.stamp.nanosec = sec, int((t0 - sec) * 1e9)
        msg.header.frame_id = 'lidar_link'
        msg.height, msg.width = 1, n
        msg.fields = [
            PointField(name=nm, offset=int(LIDAR_DTYPE.fields[nm][1]), datatype=dt, count=1)
            for nm, dt in (('x', 7), ('y', 7), ('z', 7), ('intensity', 7), ('tag', 2), ('line', 2), ('timestamp', 8))
        ]
        msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 26, 26 * n, True
        msg.data = rec.tobytes()
        self.pub_lidar.publish(msg)

    def _publish_images(self) -> None:
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        stamp = now.to_msg()
        T_world_base = self.base_pose(t)
        T_cam = T_world_base @ T_BASE_CAM
        rgb, depth, labels = syn.render(self.scene, self.cam, T_cam)
        img = Image()
        img.header.stamp, img.header.frame_id = stamp, 'camera_optical_frame'
        img.height, img.width, img.encoding, img.step = 480, 640, 'rgb8', 640 * 3
        img.data = np.ascontiguousarray(rgb[..., ::-1]).tobytes()
        self.pub_rgb.publish(img)
        self.pub_info.publish(_info(self.cam, 'camera_optical_frame', stamp))
        if self.publish_depth:
            d = Image()
            d.header.stamp, d.header.frame_id = stamp, 'camera_optical_frame'
            d.height, d.width, d.encoding, d.step = 480, 640, '32FC1', 640 * 4
            dd = depth.copy()
            dd[labels < 0] = np.nan  # like a real sensor: no walls beyond range
            d.data = dd.astype(np.float32).tobytes()
            self.pub_depth.publish(d)
        if self.thermal:
            ct = self.cam_th
            uu, vv = np.meshgrid(np.arange(ct.width) + 0.5, np.arange(ct.height) + 0.5)
            dirs = ct.rays(np.stack([uu.ravel(), vv.ravel()], axis=1)) @ (T_world_base @ T_BASE_THERMAL)[:3, :3].T
            _, ids = self.scene.cast((T_world_base @ T_BASE_THERMAL)[:3, 3], dirs)
            temp = np.full(ids.shape, 22.0, np.float32)
            for i, tc in self.temps.items():
                temp[ids == i] = tc
            temp = (temp + self.rng.normal(0, 0.15, temp.shape)).reshape(ct.height, ct.width).astype(np.float32)
            mono = np.clip((temp - 10.0) / 40.0 * 255.0, 0, 255).astype(np.uint8)
            for pub, arr, enc, bpp in ((self.pub_th, mono, 'mono8', 1), (self.pub_temp, temp, '32FC1', 4)):
                m = Image()
                m.header.stamp, m.header.frame_id = stamp, 'thermal_optical_frame'
                m.height, m.width, m.encoding, m.step = ct.height, ct.width, enc, ct.width * bpp
                m.data = arr.tobytes()
                pub.publish(m)
            self.pub_th_info.publish(_info(ct, 'thermal_optical_frame', stamp))


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticStream()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor(num_threads=4))
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
