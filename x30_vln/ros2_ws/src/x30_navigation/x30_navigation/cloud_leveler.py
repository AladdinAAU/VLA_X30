"""cloud_leveler: /lidar_points -> gravity-aligned, ground-classified cloud (/nav/cloud_level) + TF base_link -> base_footprint.

The body of a walking quadruped pitches and rolls. This node estimates the local ground plane from every scan
(seeded by the IMU when available), publishes `base_footprint` (on the ground below `base_link`, level, same
heading) and the cloud expressed in it: z is the height above the local ground. Points are flagged ground (1)
or obstacle (2) with the rules in `HeightBand`; everything else (ceilings, high structures, the robot body) is dropped.
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from .cloud_ops import HeightBand, body_hits, classify
from .geometry import fill_transform, quat_to_matrix, transform_msg_to_matrix, rpy_to_matrix
from .ground import GroundEstimate, GroundEstimator, GroundParams, UP, footprint_transform, to_level_frame
from .pointcloud import cloud_to_struct, make_cloud, xyz_of

GROUND, OBSTACLE, BODY = 1, 2, 4      # BODY: hit inside the robot's body box (stops free-space carving only)


class CloudLeveler(Node):
    def __init__(self):
        super().__init__("cloud_leveler")
        d = self.declare_parameter
        d("cloud_topic", "/lidar_points")
        d("imu_topic", "/imu/data")
        d("output_topic", "/nav/cloud_level")
        d("base_frame", "base_link")
        d("level_frame", "base_footprint")
        d("tilt_source", "auto")             # auto: IMU gives the prior up direction, ground fit refines; ground: fit only; none: level body
        d("fixed_sensor_height", 0.45)       # m above ground, used when tilt_source is none or the fit never succeeds
        d("imu_max_age", 0.5)
        d("extrinsic_xyzrpy", [0.0] * 6)     # base_link -> cloud frame, used only while TF has no such transform
        d("max_abs_z", 20.0)                 # sensor-frame outlier gate before anything else
        for k, v in vars(HeightBand()).items():
            d("band." + k, float(v))
        for k, v in vars(GroundParams()).items():
            d("ground." + k, v)
        g = self.get_parameter
        self.base_frame = g("base_frame").value
        self.level_frame = g("level_frame").value
        self.tilt_source = g("tilt_source").value
        self.fixed_height = float(g("fixed_sensor_height").value)
        self.imu_max_age = float(g("imu_max_age").value)
        self.max_abs_z = float(g("max_abs_z").value)
        self.band = HeightBand(**{k: float(g("band." + k).value) for k in vars(HeightBand())})
        gp = GroundParams()
        for k, v in vars(gp).items():
            setattr(gp, k, type(v)(g("ground." + k).value))
        self.estimator = GroundEstimator(gp)
        x = g("extrinsic_xyzrpy").value
        self.fallback_ext = np.eye(4)
        self.fallback_ext[:3, :3] = rpy_to_matrix(*x[3:])
        self.fallback_ext[:3, 3] = x[:3]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.up_in_base: np.ndarray | None = None
        self.up_wall: float = 0.0
        self.t_base_cloud: np.ndarray | None = None
        self.pub = self.create_publisher(PointCloud2, g("output_topic").value, 5)
        self.create_subscription(PointCloud2, g("cloud_topic").value, self.on_cloud, qos_profile_sensor_data)
        if self.tilt_source == "auto":
            self.create_subscription(Imu, g("imu_topic").value, self.on_imu, qos_profile_sensor_data)
        self.times: list[float] = []
        self.create_timer(10.0, self.report)
        self.get_logger().info(f"cloud_leveler: tilt_source={self.tilt_source}, band={self.band}")

    def on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        if q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w < 0.5 or msg.orientation_covariance[0] == -1.0:
            return
        r_world_imu = quat_to_matrix(q.x, q.y, q.z, q.w)
        up_imu = r_world_imu.T @ UP
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, rclpy.time.Time())
            up = transform_msg_to_matrix(tf.transform)[:3, :3] @ up_imu
        except Exception:
            up = up_imu  # IMU axes are aligned with the body (vendor manual): identity extrinsic
        self.up_in_base = up / np.linalg.norm(up)
        self.up_wall = time.monotonic()

    def cloud_to_base(self, frame: str) -> np.ndarray | None:
        """4x4 pose of the cloud frame in base_link (cached once found)."""
        if self.t_base_cloud is not None:
            return self.t_base_cloud
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame, rclpy.time.Time())
            self.t_base_cloud = transform_msg_to_matrix(tf.transform)
        except Exception:
            if not np.allclose(self.fallback_ext, np.eye(4)):
                self.t_base_cloud = self.fallback_ext
            elif frame == self.base_frame:
                self.t_base_cloud = np.eye(4)
            else:
                self.get_logger().warn(f"no TF {self.base_frame} <- {frame} yet", throttle_duration_sec=5.0)
        return self.t_base_cloud

    def on_cloud(self, msg: PointCloud2) -> None:
        t0 = time.perf_counter()
        t_bc = self.cloud_to_base(msg.header.frame_id)
        if t_bc is None:
            return
        pts = cloud_to_struct(msg)
        if len(pts) == 0:
            return
        xyz = xyz_of(pts)
        keep = np.isfinite(xyz).all(axis=1) & (np.abs(xyz[:, 2]) < self.max_abs_z)
        xyz = xyz[keep]
        inten = pts["intensity"][keep] if "intensity" in pts.dtype.names else np.zeros(len(xyz), np.float32)
        xyz_b = xyz @ t_bc[:3, :3].T + t_bc[:3, 3]

        est = self.estimate(xyz_b)
        lev = to_level_frame(xyz_b, est)
        ground, obstacle = classify(lev, self.band)
        body = body_hits(lev, self.band)
        sel = ground | obstacle | body
        flags = np.select([ground, obstacle], [GROUND, OBSTACLE], BODY).astype(np.uint8)[sel]

        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = self.base_frame
        tf.child_frame_id = self.level_frame
        fill_transform(tf.transform, footprint_transform(est))
        self.tf_broadcaster.sendTransform(tf)

        header = msg.header
        header.frame_id = self.level_frame
        out = make_cloud(header, {
            "x": lev[sel, 0].astype(np.float32), "y": lev[sel, 1].astype(np.float32),
            "z": lev[sel, 2].astype(np.float32), "intensity": inten[sel].astype(np.float32), "flags": flags})
        self.pub.publish(out)
        self.times.append(time.perf_counter() - t0)
        if not est.valid:
            self.get_logger().warn("ground fit failed, reusing the previous plane", throttle_duration_sec=5.0)

    def estimate(self, xyz_b: np.ndarray) -> GroundEstimate:
        if self.tilt_source == "none":
            return GroundEstimate(normal=UP.copy(), offset=-self.fixed_height, inliers=0, valid=True)
        prior = None
        if self.tilt_source == "auto" and self.up_in_base is not None and time.monotonic() - self.up_wall < self.imu_max_age:
            prior = self.up_in_base
        est = self.estimator.fit(xyz_b, prior)
        if not est.valid and not self.estimator.last.valid:
            est = GroundEstimate(normal=prior if prior is not None else UP.copy(), offset=-self.fixed_height, valid=False)
        return est

    def report(self) -> None:
        if self.times:
            a = np.array(self.times) * 1e3
            self.get_logger().info(f"leveler: {len(a)} scans, {a.mean():.1f} ms mean, {a.max():.1f} ms max; "
                                   f"height above ground {self.estimator.last.height_above_ground:.3f} m")
            self.times.clear()


def main(args=None):
    rclpy.init(args=args)
    node = CloudLeveler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
