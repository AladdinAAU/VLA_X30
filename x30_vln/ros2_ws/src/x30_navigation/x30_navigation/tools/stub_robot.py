"""nav_stub_robot: minimal stand-in for the simulator/robot so the navigation stack can be tested alone.

Publishes the robot contract the navigation stack needs: /lidar_points (merged-4-lidar style cloud, same fields),
/imu/data, /odom (leg-like planar odometry with configurable scale error and yaw drift), /gt/pose, static TF
base_link -> lidar_link/imu_link; consumes a Twist. The robot is a kinematic holonomic body with acceleration limits
that cannot pass through walls (contact is counted in /stub/collisions) and that wobbles in roll/pitch while it moves
(to exercise the ground levelling). Not a physics simulation: use x30_sim for that.
"""
from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Int32
from tf2_ros import StaticTransformBroadcaster

from ..geometry import fill_pose, fill_transform, make_transform, rpy_to_matrix
from ..pointcloud import make_cloud
from .stub_world import make_world


class StubRobot(Node):
    def __init__(self):
        super().__init__("nav_stub_robot")
        d = self.declare_parameter
        d("world", "rooms")
        d("seed", 0)
        d("start", [2.0, 2.0, 0.0])
        d("cmd_topic", "/cmd_vel")
        d("body_height", 0.45)
        d("footprint", [1.0, 0.46])   # length, width (width not documented for the X30: assumed)
        d("rate_hz", 50.0)
        d("lidar_rate_hz", 10.0)
        d("lidar_azimuth", 720)
        d("lidar_elevations_deg", [-50.0, -42.0, -34.0, -26.0, -20.0, -15.0, -10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0, 15.0, 22.0, 30.0])
        d("lidar_noise", 0.01)
        d("lidar_max_range", 30.0)
        d("body_blind_radius", 0.1)
        d("sensor_x", 0.45)               # front and rear sensors at +-sensor_x (vendor: +-0.44 / -0.46 m), each covers +-110 deg of its side
        d("accel", [1.0, 1.0, 2.0])
        d("cmd_timeout", 0.5)
        d("wobble_amp", 0.03)             # rad of roll/pitch while walking
        d("wobble_hz", 2.0)
        d("static_tilt", [0.0, 0.0])      # roll, pitch (rad) always present
        d("leg_scale", 1.0)               # leg odometry distance scale error
        d("leg_yaw_drift", 0.0)           # rad/s constant yaw-rate bias of the leg odometry
        d("leg_noise", 0.01)
        d("leg_freeze_after", -1.0)       # s; after this the leg odometry stops updating (simulates a stuck estimator)
        d("leg_jump_at", -1.0)            # s; leg odometry pose jumps by 2 m once
        g = self.get_parameter
        self.world = make_world(g("world").value, int(g("seed").value))
        x, y, yaw = g("start").value
        self.pos = np.array([float(x), float(y)])
        self.yaw = float(yaw)
        self.vel = np.zeros(3)
        self.cmd = np.zeros(3)
        self.cmd_wall = 0.0
        self.p = {k: g(k).value for k in ("body_height", "footprint", "rate_hz", "lidar_azimuth", "lidar_noise", "lidar_max_range",
                                           "body_blind_radius", "sensor_x", "accel", "cmd_timeout", "wobble_amp", "wobble_hz", "static_tilt",
                                           "leg_scale", "leg_yaw_drift", "leg_noise", "leg_freeze_after", "leg_jump_at")}
        el = np.radians(np.array(g("lidar_elevations_deg").value))
        az = np.linspace(-math.pi, math.pi, self.p["lidar_azimuth"], endpoint=False)
        aa, ee = np.meshgrid(az, el)
        self.az_grid = aa.reshape(-1)
        self.rays_body = np.column_stack((np.cos(ee.reshape(-1)) * np.cos(self.az_grid), np.cos(ee.reshape(-1)) * np.sin(self.az_grid),
                                          np.sin(ee.reshape(-1))))
        self.rng = np.random.default_rng(int(g("seed").value) + 1)
        self.rp = np.zeros(2)  # roll, pitch
        self.t_start = time.monotonic()
        self.collisions = 0
        self.in_contact = False
        self.distance = 0.0
        # leg odometry state (planar, starts at the origin with the robot's yaw as heading 0)
        self.leg_pose = np.zeros(3)
        self.leg_yaw_bias = 0.0
        self.leg_frozen = False
        self.leg_jumped = False

        self.cloud_pub = self.create_publisher(PointCloud2, "/lidar_points", qos_profile_sensor_data)
        self.imu_pub = self.create_publisher(Imu, "/imu/data", qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, "/odom", qos_profile_sensor_data)
        self.gt_pub = self.create_publisher(PoseStamped, "/gt/pose", 10)
        self.col_pub = self.create_publisher(Int32, "/stub/collisions", 10)
        self.create_subscription(Twist, g("cmd_topic").value, self.on_cmd, 10)
        self.static_tf = StaticTransformBroadcaster(self)
        tfs = []
        for child in ("lidar_link", "imu_link"):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = child
            t.transform.rotation.w = 1.0
            tfs.append(t)
        self.static_tf.sendTransform(tfs)   # one call: the latched static topic keeps only the last message
        self.last_t = time.monotonic()
        self.create_timer(1.0 / self.p["rate_hz"], self.step)
        self.create_timer(1.0 / float(g("lidar_rate_hz").value), self.publish_lidar)
        self.get_logger().info(f"stub robot in world '{self.world.name}' at {self.pos}, {len(self.rays_body)} lidar rays")

    def on_cmd(self, msg: Twist) -> None:
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.cmd_wall = time.monotonic()

    def step(self) -> None:
        now = time.monotonic()
        dt = min(now - self.last_t, 0.1)
        self.last_t = now
        cmd = self.cmd if now - self.cmd_wall < self.p["cmd_timeout"] else np.zeros(3)
        acc = np.array(self.p["accel"])
        dv = np.clip(cmd - self.vel, -acc * dt, acc * dt)
        self.vel = self.vel + dv
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx = (c * self.vel[0] - s * self.vel[1]) * dt
        dy = (s * self.vel[0] + c * self.vel[1]) * dt
        new_yaw = self.yaw + self.vel[2] * dt
        fl, fw = self.p["footprint"]
        blocked = self.world.collides(self.pos[0] + dx, self.pos[1] + dy, new_yaw, fl, fw)
        if blocked:
            if not self.in_contact:
                self.collisions += 1
            self.in_contact = True
            dx = dy = 0.0
            new_yaw = self.yaw if self.world.collides(self.pos[0], self.pos[1], new_yaw, fl, fw) else new_yaw
        else:
            self.in_contact = False
        self.pos += (dx, dy)
        self.yaw = new_yaw
        self.distance += math.hypot(dx, dy)
        speed = min(1.0, float(np.hypot(self.vel[0], self.vel[1])) / 0.4 + abs(self.vel[2]) / 0.8)
        t = now - self.t_start
        amp, hz = self.p["wobble_amp"], self.p["wobble_hz"]
        self.rp = np.array([self.p["static_tilt"][0] + amp * speed * math.sin(2 * math.pi * hz * t),
                            self.p["static_tilt"][1] + amp * speed * math.sin(2 * math.pi * hz * t + 1.0)])
        self.update_leg(dt, t)
        stamp = self.get_clock().now().to_msg()
        self.publish_state(stamp)
        msg = Int32()
        msg.data = self.collisions
        self.col_pub.publish(msg)

    def update_leg(self, dt: float, t: float) -> None:
        if self.p["leg_freeze_after"] >= 0 and t > self.p["leg_freeze_after"]:
            self.leg_frozen = True
        if self.leg_frozen:
            return
        vx, vy = self.vel[0] * self.p["leg_scale"], self.vel[1] * self.p["leg_scale"]
        wz = self.vel[2] + self.p["leg_yaw_drift"]
        yaw = self.leg_pose[2]
        self.leg_pose[0] += (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt + self.rng.normal(0, self.p["leg_noise"]) * dt
        self.leg_pose[1] += (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt + self.rng.normal(0, self.p["leg_noise"]) * dt
        self.leg_pose[2] += wz * dt
        if self.p["leg_jump_at"] >= 0 and t > self.p["leg_jump_at"] and not self.leg_jumped:
            self.leg_jumped = True
            self.leg_pose[0] += 2.0

    def body_rotation(self) -> np.ndarray:
        return rpy_to_matrix(self.rp[0], self.rp[1], self.yaw)

    def publish_state(self, stamp) -> None:
        z = self.p["body_height"]
        pose = make_transform(self.body_rotation(), [self.pos[0], self.pos[1], z])
        gt = PoseStamped()
        gt.header.stamp, gt.header.frame_id = stamp, "world"
        fill_pose(gt.pose, pose)
        self.gt_pub.publish(gt)
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, "odom", "base_link"
        leg = make_transform(rpy_to_matrix(0.0, 0.0, self.leg_pose[2]), [self.leg_pose[0], self.leg_pose[1], 0.0])
        fill_pose(o.pose.pose, leg)
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.angular.z = (
            0.0 if self.leg_frozen else self.vel[0] * self.p["leg_scale"], 0.0 if self.leg_frozen else self.vel[1] * self.p["leg_scale"],
            0.0 if self.leg_frozen else self.vel[2] + self.p["leg_yaw_drift"])
        self.odom_pub.publish(o)
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, "imu_link"
        q = make_transform(self.body_rotation(), [0, 0, 0])
        from ..geometry import matrix_to_quat
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = matrix_to_quat(q[:3, :3])
        imu.orientation_covariance[0] = 0.01
        self.imu_pub.publish(imu)

    def publish_lidar(self) -> None:
        r_body = self.body_rotation()
        chunks, times = [], []
        for sign in (1.0, -1.0):   # front sensor looks forward, rear sensor backward; the body hides the rest
            off = np.array([sign * self.p["sensor_x"], 0.0, 0.0])
            facing = np.cos(self.az_grid - (0.0 if sign > 0 else math.pi)) > math.cos(math.radians(110.0))
            origin = np.array([self.pos[0], self.pos[1], self.p["body_height"]]) + r_body @ off
            dirs = self.rays_body[facing]
            rng = self.world.raycast(origin, dirs @ r_body.T, self.p["lidar_max_range"])
            ok = np.isfinite(rng) & (rng > self.p["body_blind_radius"])
            rng = rng + self.rng.normal(0.0, self.p["lidar_noise"], len(rng))
            chunks.append(off + dirs[ok] * rng[ok, None])
            times.append(self.az_grid[facing][ok])
        pts = np.vstack(chunks)
        az = np.concatenate(times)
        n = len(pts)
        stamp = self.get_clock().now()
        t_ms = stamp.nanoseconds * 1e-6 + (az + math.pi) / (2 * math.pi) * 100.0
        from std_msgs.msg import Header
        header = Header()
        header.stamp = stamp.to_msg()
        header.frame_id = "lidar_link"
        msg = make_cloud(header, {"x": pts[:, 0].astype(np.float32), "y": pts[:, 1].astype(np.float32), "z": pts[:, 2].astype(np.float32),
                                  "intensity": np.full(n, 30.0, np.float32), "tag": np.zeros(n, np.uint8),
                                  "line": np.zeros(n, np.uint8), "timestamp": t_ms})
        self.cloud_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StubRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
