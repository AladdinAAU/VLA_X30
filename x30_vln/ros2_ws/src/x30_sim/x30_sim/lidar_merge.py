"""Merge the four simulated Livox clouds into the single contract cloud `/lidar_points` (frame lidar_link).

Input : `/lidar/<id>/points` (PointCloud2 from the Gazebo gpu_lidar via ros_gz_bridge, sensor frame, organised, inf for no return)
Output: `/lidar_points` in the exact layout of the real robot (INTERFACES section 2): x,y,z,intensity f32, tag u8, line u8,
        timestamp f64 (ms), point_step 26, is_dense true, the four lidars concatenated in the real order.
Invalid rays (no echo, outside [range_min, range_max], or blocked by the robot body) become a point at the sensor origin with
intensity 0, exactly like the real cloud (config lidar.invalid_as_origin); otherwise they are dropped.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

from . import config as cfgmod
from . import lidar_model as lm

_NP_FROM_DATATYPE = {
    PointField.INT8: 'i1', PointField.UINT8: 'u1', PointField.INT16: 'i2', PointField.UINT16: 'u2',
    PointField.INT32: 'i4', PointField.UINT32: 'u4', PointField.FLOAT32: 'f4', PointField.FLOAT64: 'f8',
}

OUT_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='tag', offset=16, datatype=PointField.UINT8, count=1),
    PointField(name='line', offset=17, datatype=PointField.UINT8, count=1),
    PointField(name='timestamp', offset=18, datatype=PointField.FLOAT64, count=1),
]


def parse_cloud(msg: PointCloud2) -> tuple[np.ndarray, np.ndarray | None]:
    """xyz (N,3) float64 and the optional intensity (N,) of a PointCloud2, whatever its field layout."""
    dt = np.dtype({
        'names': [f.name for f in msg.fields],
        'formats': [_NP_FROM_DATATYPE[f.datatype] for f in msg.fields],
        'offsets': [f.offset for f in msg.fields],
        'itemsize': msg.point_step,
    })
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
    inten = arr['intensity'].astype(np.float32) if 'intensity' in arr.dtype.names else None
    return xyz, inten


class LidarMerge(Node):
    def __init__(self):
        super().__init__('lidar_merge')
        self.declare_parameter('robot_config', '')
        self.declare_parameter('seed', 1)
        self.declare_parameter('sync_tolerance_s', 0.02)
        self.declare_parameter('partial_timeout_s', 0.25)
        cfg = cfgmod.load_robot_config(self.get_parameter('robot_config').value or None)
        lc = cfg['lidar']
        self.lc = lc
        self.frame = cfg['frames']['merged_lidar']
        self.order = lc['merged_order']
        self.mounts = {k: lc['mounts'][k] for k in self.order}
        self.T = {k: cfgmod.lidar_mount_matrix(m) for k, m in self.mounts.items()}
        self.rng = np.random.default_rng(int(self.get_parameter('seed').value))
        self.tol = float(self.get_parameter('sync_tolerance_s').value)
        self.timeout = float(self.get_parameter('partial_timeout_s').value)
        self.pending: dict[str, PointCloud2] = {}
        self.n_points: dict[str, int] = {}
        self.published = 0
        self.partial = 0

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=3)
        self.pub = self.create_publisher(PointCloud2, cfg['topics']['lidar'], qos)
        for key in self.order:
            self.create_subscription(PointCloud2, f'/lidar/{key}/points', lambda m, k=key: self.on_cloud(k, m), qos)
        self.create_timer(0.05, self.on_timer)
        self.get_logger().info(f'merging lidars {self.order} into {cfg["topics"]["lidar"]} ({self.frame})')

    @staticmethod
    def stamp_s(msg: PointCloud2) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def on_cloud(self, key: str, msg: PointCloud2) -> None:
        self.pending[key] = msg
        if len(self.pending) == len(self.order):
            stamps = [self.stamp_s(m) for m in self.pending.values()]
            if max(stamps) - min(stamps) <= self.tol:
                self.publish_merged()

    def on_timer(self) -> None:
        if not self.pending:
            return
        oldest = min(self.stamp_s(m) for m in self.pending.values())
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - oldest > self.timeout:
            self.partial += 1
            self.get_logger().warn(f'publishing partial scan, missing {[k for k in self.order if k not in self.pending]}',
                                   throttle_duration_sec=5.0)
            self.publish_merged()

    def lidar_block(self, key: str, msg: PointCloud2 | None, t0_ms: float, index: int) -> np.ndarray:
        lc = self.lc
        if msg is None:
            n = self.n_points.get(key) or lc['horizontal_samples'] * lc['vertical_samples']
            xyz = np.full((n, 3), np.inf)
            retro = None
        else:
            xyz, retro = parse_cloud(msg)
            self.n_points[key] = len(xyz)
        n = len(xyz)
        T = self.T[key]
        origin = T[:3, 3]
        r = np.linalg.norm(xyz, axis=1)
        valid = np.isfinite(r) & (r >= lc['range_min_m']) & (r <= lc['range_max_m'])
        body = np.full((n, 3), np.nan)
        body[valid] = xyz[valid] @ T[:3, :3].T + origin
        sm = lc['self_mask']
        if valid.any():
            dirs = (body[valid] - origin) / r[valid][:, None]
            blocked = lm.self_mask(origin, dirs, r[valid], sm['boxes'], sm['min_chord_m'])
            valid[np.flatnonzero(valid)[blocked]] = False
        pts = np.where(valid[:, None], body, origin)
        inten = np.zeros(n, dtype=np.float32)
        if valid.any():
            retro_v = retro[valid] if retro is not None else None
            inten[valid] = lm.synth_intensity(self.rng, r[valid], lc['intensity'], retro_v)
        if lc['emulate_tags']:
            tag = lm.sample_tags(self.rng, valid, lc['tag_distribution'])
        else:
            tag = np.zeros(n, dtype=np.uint8)
        line = (index * lc['lines_per_lidar'] + (np.arange(n) % lc['lines_per_lidar'])).astype(np.uint8)
        ts = t0_ms + lc['scan_time_spread_s'] * 1000.0 * np.arange(n) / max(n, 1)
        block = lm.assemble(pts, inten, tag, line, ts)
        if not lc['invalid_as_origin']:
            block = block[valid]
        return block

    def publish_merged(self) -> None:
        stamps = [m.header.stamp for m in self.pending.values()]
        first = min(stamps, key=lambda s: (s.sec, s.nanosec))
        t0_ms = first.sec * 1e3 + first.nanosec * 1e-6
        blocks = [self.lidar_block(k, self.pending.get(k), t0_ms, i) for i, k in enumerate(self.order)]
        data = np.concatenate(blocks)
        out = PointCloud2()
        out.header.stamp = first
        out.header.frame_id = self.frame
        out.height = 1
        out.width = len(data)
        out.fields = OUT_FIELDS
        out.is_bigendian = False
        out.point_step = lm.POINT_DTYPE.itemsize
        out.row_step = out.point_step * out.width
        out.is_dense = True
        out.data = data.tobytes()
        self.pub.publish(out)
        self.published += 1
        self.pending.clear()


def main(args=None):
    rclpy.init(args=args)
    node = LidarMerge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
