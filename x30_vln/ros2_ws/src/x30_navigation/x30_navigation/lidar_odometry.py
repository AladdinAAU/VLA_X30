"""lidar_odometry: KISS-ICP on the merged 4-lidar cloud -> /odometry/lidar (base_link pose in `odom`).

The per-point `timestamp` field (any unit; normalised per scan) drives KISS-ICP's constant-velocity deskewing.
The odom frame is gravity aligned: the first scan's ground plane gives the initial roll/pitch and height of
`base_link`, so the output is a full 6-DoF pose with z up. This node publishes no TF; odom_selector owns
`odom -> base_link`.
"""
from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener

from .geometry import fill_pose, invert, make_transform, matrix_to_rpy, rpy_to_matrix, transform_msg_to_matrix
from .ground import GroundEstimator, GroundParams, level_rotation
from .pointcloud import cloud_to_struct, make_cloud, xyz_of, voxel_downsample


def rotvec(r: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    c = max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.zeros(3)
    return axis / n * ang


class LidarOdometry(Node):
    def __init__(self):
        super().__init__("lidar_odometry")
        d = self.declare_parameter
        d("cloud_topic", "/lidar_points")
        d("odom_topic", "/odometry/lidar")
        d("odom_frame", "odom")
        d("base_frame", "base_link")
        d("extrinsic_xyzrpy", [0.0] * 6)    # base_link -> cloud frame, used while TF has no such transform
        d("max_range", 30.0)                # m; far points of the merged cloud are sparse and noisy
        d("min_range", 0.9)                 # m; removes the robot's own legs/body seen by the four sensors
        d("voxel_size", 0.4)                # m; KISS-ICP map voxel (indoor: 0.3-0.5, rule of thumb max_range / 100)
        d("max_points_per_voxel", 20)
        d("deskew", True)
        d("min_motion_th", 0.05)            # m; adaptive threshold floor, small for slow quadruped motion
        d("initial_threshold", 1.0)         # m; first correspondence threshold (default 2.0 is for cars)
        d("max_num_iterations", 100)
        d("convergence_criterion", 1e-4)
        d("gravity_align", True)            # initial roll/pitch/height of base_link from the first ground fit
        d("fixed_sensor_height", 0.45)
        d("pose_cov_xyz", 0.0004)
        d("pose_cov_rpy", 0.0004)
        d("publish_map_3d", False)
        d("map_3d_topic", "/map_3d")
        d("map_3d_period", 5.0)
        d("map_3d_voxel", 0.15)
        g = self.get_parameter
        self.odom_frame = g("odom_frame").value
        self.base_frame = g("base_frame").value
        self.deskew = bool(g("deskew").value)
        self.gravity_align = bool(g("gravity_align").value)
        self.fixed_height = float(g("fixed_sensor_height").value)
        self.cov_xyz, self.cov_rpy = float(g("pose_cov_xyz").value), float(g("pose_cov_rpy").value)
        x = g("extrinsic_xyzrpy").value
        self.fallback_ext = make_transform(rpy_to_matrix(*x[3:]), x[:3])
        self.min_range, self.max_range = float(g("min_range").value), float(g("max_range").value)

        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP
        cfg = KISSConfig()
        cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = self.max_range, self.min_range, self.deskew
        cfg.mapping.voxel_size = float(g("voxel_size").value)
        cfg.mapping.max_points_per_voxel = int(g("max_points_per_voxel").value)
        cfg.adaptive_threshold.min_motion_th = float(g("min_motion_th").value)
        cfg.adaptive_threshold.initial_threshold = float(g("initial_threshold").value)
        cfg.registration.max_num_iterations = int(g("max_num_iterations").value)
        cfg.registration.convergence_criterion = float(g("convergence_criterion").value)
        self._kiss_cls, self._cfg = KissICP, cfg
        self.kiss = KissICP(cfg)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.t_base_lidar: np.ndarray | None = None   # pose of the cloud frame in base_link
        self.t_odom_base0: np.ndarray | None = None
        self.prev: tuple[float, np.ndarray] | None = None
        self.times: list[float] = []
        self.map_vox: dict[int, np.ndarray] = {}
        self.map_pts = np.zeros((0, 3), np.float32)
        self.pub = self.create_publisher(Odometry, g("odom_topic").value, 10)
        self.create_subscription(PointCloud2, g("cloud_topic").value, self.on_cloud, qos_profile_sensor_data)
        self.create_service(Empty, "~/reset", self.on_reset)
        if g("publish_map_3d").value:
            self.map_voxel = float(g("map_3d_voxel").value)
            self.map_pub = self.create_publisher(PointCloud2, g("map_3d_topic").value, 1)
            self.create_timer(float(g("map_3d_period").value), self.publish_map_3d)
        self.map_enabled = bool(g("publish_map_3d").value)
        self.create_timer(10.0, self.report)

    def on_reset(self, _req, resp):
        self.kiss = self._kiss_cls(self._cfg)
        self.t_odom_base0 = None
        self.prev = None
        self.map_pts = np.zeros((0, 3), np.float32)
        self.get_logger().info("lidar odometry reset")
        return resp

    def lookup_extrinsic(self, frame: str) -> np.ndarray | None:
        if self.t_base_lidar is not None:
            return self.t_base_lidar
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame, rclpy.time.Time())
            self.t_base_lidar = transform_msg_to_matrix(tf.transform)
        except Exception:
            if frame == self.base_frame:
                self.t_base_lidar = np.eye(4)
            elif not np.allclose(self.fallback_ext, np.eye(4)):
                self.t_base_lidar = self.fallback_ext
            else:
                self.get_logger().warn(f"no TF {self.base_frame} <- {frame}; waiting", throttle_duration_sec=5.0)
        return self.t_base_lidar

    def initial_pose(self, xyz_lidar: np.ndarray, t_bl: np.ndarray) -> np.ndarray:
        """T_odom_base at the first scan: level (z up), standing at the origin, height from the ground fit."""
        if not self.gravity_align:
            return np.eye(4)
        xyz_b = xyz_lidar @ t_bl[:3, :3].T + t_bl[:3, 3]
        est = GroundEstimator(GroundParams(smooth=1.0)).fit(xyz_b)
        if not est.valid:
            self.get_logger().warn("no ground plane in the first scan: starting level at the fixed sensor height")
            return make_transform(np.eye(3), [0.0, 0.0, self.fixed_height])
        r_base_in_level = level_rotation(est.normal).T
        roll, pitch, _ = matrix_to_rpy(r_base_in_level)
        self.get_logger().info(f"initial tilt roll={math.degrees(roll):.1f} pitch={math.degrees(pitch):.1f} deg, "
                               f"height above ground {est.height_above_ground:.3f} m ({est.inliers} inliers)")
        return make_transform(r_base_in_level, [0.0, 0.0, est.height_above_ground])

    def on_cloud(self, msg: PointCloud2) -> None:
        t_bl = self.lookup_extrinsic(msg.header.frame_id)
        if t_bl is None:
            return
        t0 = time.perf_counter()
        pts = cloud_to_struct(msg)
        xyz = xyz_of(pts)
        stamps = pts["timestamp"].astype(np.float64) if "timestamp" in pts.dtype.names else None
        ok = np.isfinite(xyz).all(axis=1)
        xyz = xyz[ok]
        if len(xyz) < 100:
            return
        if self.t_odom_base0 is None:
            self.t_odom_base0 = self.initial_pose(xyz, t_bl)
        tn = np.full(len(xyz), 0.5)
        if stamps is not None:
            s = stamps[ok]
            span = s.max() - s.min()
            if span > 0.0:
                tn = (s - s.min()) / span
        frame, _ = self.kiss.register_frame(xyz, tn)
        t_l0_lt = self.kiss.last_pose
        t_ob = self.t_odom_base0 @ t_bl @ t_l0_lt @ invert(t_bl)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.publish(msg.header.stamp, stamp, t_ob)
        if self.map_enabled:
            self.accumulate(frame, self.t_odom_base0 @ t_bl @ t_l0_lt)
        self.times.append(time.perf_counter() - t0)

    def publish(self, stamp_msg, stamp: float, t_ob: np.ndarray) -> None:
        o = Odometry()
        o.header.stamp = stamp_msg
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        fill_pose(o.pose.pose, t_ob)
        cov = np.zeros(36)
        cov[0] = cov[7] = cov[14] = self.cov_xyz
        cov[21] = cov[28] = cov[35] = self.cov_rpy
        o.pose.covariance = cov.tolist()
        if self.prev is not None and stamp > self.prev[0]:
            dt = stamp - self.prev[0]
            rel = invert(self.prev[1]) @ t_ob
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = (float(v) for v in rel[:3, 3] / dt)
            o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = (float(v) for v in rotvec(rel[:3, :3]) / dt)
        tc = np.zeros(36)
        tc[0] = tc[7] = tc[14] = tc[21] = tc[28] = tc[35] = 0.01
        o.twist.covariance = tc.tolist()
        self.prev = (stamp, t_ob)
        self.pub.publish(o)

    def accumulate(self, frame: np.ndarray, t_o_lidar: np.ndarray) -> None:
        """Add the deskewed scan (in the lidar frame) to a voxel map in the odom frame."""
        world = frame @ t_o_lidar[:3, :3].T + t_o_lidar[:3, 3]
        keep = world[:, 2] > -5.0
        pts = voxel_downsample(world[keep], self.map_voxel)[0].astype(np.float32)
        merged = np.vstack((self.map_pts, pts))
        self.map_pts = voxel_downsample(merged, self.map_voxel)[0]

    def publish_map_3d(self) -> None:
        if len(self.map_pts) == 0:
            return
        from std_msgs.msg import Header
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self.odom_frame
        self.map_pub.publish(make_cloud(h, {"x": self.map_pts[:, 0], "y": self.map_pts[:, 1], "z": self.map_pts[:, 2]}))

    def report(self) -> None:
        if self.times:
            a = np.array(self.times) * 1e3
            self.get_logger().info(f"lidar_odometry: {len(a)} scans, {a.mean():.1f} ms mean, {a.max():.1f} ms max")
            self.times.clear()


def main(args=None):
    rclpy.init(args=args)
    node = LidarOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
