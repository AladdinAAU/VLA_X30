"""Robot state outputs of the simulator: /odom + TF odom->base_link, /gt/pose, /x30/battery, /x30/health, /x30/ready (kinematic).

Input: /gt/odom_raw (nav_msgs/Odometry from the Gazebo OdometryPublisher through ros_gz_bridge: model pose in the world frame).
/odom is planar leg-odometry-like (x, y, yaw), origin = pose at start / last /sim/reset_odom, error scaled by param odom_noise.
"""
from __future__ import annotations

import math
import time

import rclpy
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import BatteryState, Image, Imu, PointCloud2
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from . import config as cfgmod
from .odometry import LegOdometry


def quat_to_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class SimState(Node):
    def __init__(self):
        super().__init__('sim_state')
        self.declare_parameter('robot_config', '')
        self.declare_parameter('locomotion', 'kinematic')
        self.declare_parameter('odom_noise', 0.0)
        self.declare_parameter('seed', 1)
        self.declare_parameter('thermal', True)
        cfg = cfgmod.load_robot_config(self.get_parameter('robot_config').value or None)
        self.cfg = cfg
        self.locomotion = self.get_parameter('locomotion').value
        self.odom_model = LegOdometry(float(self.get_parameter('odom_noise').value), cfg['odometry']['nominal'],
                                      int(self.get_parameter('seed').value))
        self.base = cfg['robot']['base_frame']
        self.odom_frame = cfg['robot']['odom_frame']
        self.world_frame = cfg['robot']['world_frame']
        self.rate = float(cfg['odometry']['rate_hz'])
        self.last_pub = 0.0
        self.t_start = self.get_clock().now()
        self.estop = False
        self.ready_sent = False
        self.last_pose = None

        self.tf = TransformBroadcaster(self)
        self.pub_odom = self.create_publisher(Odometry, cfg['topics']['odom'], 20)
        self.pub_gt = self.create_publisher(PoseStamped, '/gt/pose', 20)
        self.pub_batt = self.create_publisher(BatteryState, '/x30/battery', 5)
        self.pub_health = self.create_publisher(DiagnosticArray, '/x30/health', 5)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_ready = self.create_publisher(Bool, '/x30/ready', latched)
        if self.locomotion == 'kinematic':
            self.pub_ready.publish(Bool(data=False))
        self.create_subscription(Odometry, '/gt/odom_raw', self.on_gt, 20)
        self.create_subscription(Bool, '/x30/estop', self.on_estop, 5)
        self.create_service(Trigger, '/sim/reset_odom', self.on_reset)

        # message counters for /x30/health (raw subscriptions: no deserialisation cost)
        self.counts: dict[str, int] = {}
        self.expected = {
            'lidar': (cfg['topics']['lidar'], PointCloud2, cfg['lidar']['rate_hz']),
            'imu': (cfg['topics']['imu'], Imu, cfg['imu']['rate_hz']),
            'camera': (cfg['topics']['rgb'], Image, cfg['camera']['rate_hz']),
            'odom': (cfg['topics']['odom'], Odometry, self.rate),
        }
        if self.get_parameter('thermal').value:
            self.expected['thermal'] = (cfg['topics']['thermal'], Image, cfg['thermal']['rate_hz'])
        for name, (topic, typ, _) in self.expected.items():
            self.counts[name] = 0
            self.create_subscription(typ, topic, lambda _m, n=name: self.counts.__setitem__(n, self.counts[n] + 1), 10, raw=True)
        self.last_counts = dict(self.counts)
        self.wall_last = time.monotonic()
        self.sim_last = 0.0
        self.sim_now = 0.0
        self.create_subscription(Clock, '/clock', self.on_clock, 10)
        self.create_timer(1.0, self.on_health)

    def on_clock(self, msg: Clock) -> None:
        self.sim_now = msg.clock.sec + msg.clock.nanosec * 1e-9

    def on_estop(self, msg: Bool) -> None:
        self.estop = bool(msg.data)

    def on_reset(self, _req, resp):
        self.odom_model.reset()
        resp.success, resp.message = True, 'odom origin reset at the next ground-truth pose'
        return resp

    def on_gt(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = quat_to_yaw(q)
        now = self.get_clock().now().to_msg()
        gt = PoseStamped()
        gt.header.stamp, gt.header.frame_id = now, self.world_frame
        gt.pose = msg.pose.pose
        self.pub_gt.publish(gt)
        self.last_pose = (p.x, p.y, yaw)
        t = now.sec + now.nanosec * 1e-9
        if t - self.last_pub < 1.0 / self.rate - 1e-6:
            return
        self.last_pub = t
        ox, oy, oyaw = self.odom_model.update(p.x, p.y, yaw)
        if self.locomotion == 'kinematic' and not self.ready_sent and t > 1.0:
            self.pub_ready.publish(Bool(data=True))
            self.ready_sent = True
        self.publish_odom(now, ox, oy, oyaw, msg)

    def publish_odom(self, stamp: Time, x: float, y: float, yaw: float, gt: Odometry) -> None:
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, self.odom_frame, self.base
        tf.transform.translation.x, tf.transform.translation.y = x, y
        tf.transform.rotation.z, tf.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.tf.sendTransform(tf)
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = stamp, self.odom_frame, self.base
        o.pose.pose.position.x, o.pose.pose.position.y = x, y
        o.pose.pose.orientation = tf.transform.rotation
        # body-frame velocity from the ground-truth twist (planar), scaled by the run's scale error
        s = self.odom_model.scale
        o.twist.twist.linear.x = gt.twist.twist.linear.x * s
        o.twist.twist.linear.y = gt.twist.twist.linear.y * s
        o.twist.twist.angular.z = gt.twist.twist.angular.z * self.odom_model.yaw_scale
        cov = (1e-3, 1e-3, 1e-1, 1e-1, 1e-1, 1e-2)
        for i, v in enumerate(cov):
            o.pose.covariance[i * 7] = v
            o.twist.covariance[i * 7] = v
        self.pub_odom.publish(o)

    def on_health(self) -> None:
        wall = time.monotonic()
        dt = max(wall - self.wall_last, 1e-3)
        rtf = (self.sim_now - self.sim_last) / dt
        self.wall_last, self.sim_last = wall, self.sim_now
        # battery
        b = self.cfg['battery']
        hours = (self.get_clock().now() - self.t_start).nanoseconds * 1e-9 / 3600.0
        pct = max(0.0, b['start_percent'] - b['drain_percent_per_hour'] * hours)
        bat = BatteryState()
        bat.header.stamp = self.get_clock().now().to_msg()
        bat.voltage = float(b['nominal_voltage_v'] + (b['full_voltage_v'] - b['nominal_voltage_v']) * pct / 100.0)
        bat.percentage = float(pct / 100.0)
        bat.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        bat.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD
        bat.present = True
        bat.charge = bat.capacity = bat.design_capacity = bat.current = float('nan')
        self.pub_batt.publish(bat)
        # diagnostics
        arr = DiagnosticArray()
        arr.header.stamp = bat.header.stamp
        for name, (topic, _, nominal) in self.expected.items():
            hz = (self.counts[name] - self.last_counts[name]) / dt
            st = DiagnosticStatus(name=f'x30_sim/{name}', hardware_id='sim')
            st.level = DiagnosticStatus.OK if hz >= 0.5 * nominal * max(rtf, 0.05) else (
                DiagnosticStatus.WARN if hz > 0 else DiagnosticStatus.ERROR)
            st.message = f'{hz:.1f} Hz (nominal {nominal} Hz at real-time)'
            st.values = [KeyValue(key='topic', value=topic), KeyValue(key='rate_hz', value=f'{hz:.2f}')]
            arr.status.append(st)
        self.last_counts = dict(self.counts)
        loco = DiagnosticStatus(name='x30_sim/locomotion', hardware_id='sim')
        loco.level = DiagnosticStatus.ERROR if self.estop else DiagnosticStatus.OK
        loco.message = 'e-stop active' if self.estop else f'{self.locomotion} ok'
        loco.values = [KeyValue(key='backend', value=self.locomotion), KeyValue(key='real_time_factor', value=f'{rtf:.2f}')]
        arr.status.append(loco)
        batt = DiagnosticStatus(name='x30_sim/battery', hardware_id='sim', level=DiagnosticStatus.OK if pct > 15 else DiagnosticStatus.WARN)
        batt.message = f'{pct:.1f} %'
        arr.status.append(batt)
        self.pub_health.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = SimState()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
