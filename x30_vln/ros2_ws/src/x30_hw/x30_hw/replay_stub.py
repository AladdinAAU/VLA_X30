"""replay_stub: SYNTHETIC stand-ins for what the recorded bag lacks (only /lidar_points was recorded).

  /odom       zero-motion identity odometry (frame odom -> base_link), 50 Hz
  /imu/data   gravity-only IMU, 100 Hz, orientation_covariance[0] = -1 (orientation NOT provided)
  /x30/ready  true (latched), /x30/battery 100 %, /x30/health OK with the message 'REPLAY STUB'
  /x30/replay_stub  Bool true (latched): lets any node or operator detect that these topics are fake
Nothing here is robot data. With use_sim_time and a bag played with --clock the stamps follow /clock.
"""

import math

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState, Imu
from std_msgs.msg import Bool

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class ReplayStub(Node):
    def __init__(self):
        super().__init__("replay_stub")
        self.pub_odom = self.create_publisher(Odometry, "/odom", 50)
        self.pub_imu = self.create_publisher(Imu, "/imu/data", 200)
        self.pub_batt = self.create_publisher(BatteryState, "/x30/battery", 10)
        self.pub_health = self.create_publisher(DiagnosticArray, "/x30/health", 10)
        ready = self.create_publisher(Bool, "/x30/ready", LATCHED)
        flag = self.create_publisher(Bool, "/x30/replay_stub", LATCHED)
        ready.publish(Bool(data=True))
        flag.publish(Bool(data=True))
        self.create_timer(0.02, self.odom)
        self.create_timer(0.01, self.imu)
        self.create_timer(1.0, self.slow)
        self.get_logger().warn("REPLAY STUB ACTIVE: /odom (zero motion), /imu/data (gravity only), /x30/ready, /x30/battery and "
                               "/x30/health are SYNTHETIC. Do not draw conclusions about odometry or IMU from this run.")

    def odom(self):
        m = Odometry()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = "odom", "base_link"
        m.pose.pose.orientation.w = 1.0
        for i in (0, 7, 14, 21, 28, 35):
            m.pose.covariance[i] = 1e-3
            m.twist.covariance[i] = 1e-3
        self.pub_odom.publish(m)

    def imu(self):
        m = Imu()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "imu_link"
        m.orientation.w = 1.0
        m.orientation_covariance[0] = -1.0  # REP-145: orientation not provided
        m.linear_acceleration.z = 9.81
        for i in (0, 4, 8):
            m.angular_velocity_covariance[i] = 1e-4
            m.linear_acceleration_covariance[i] = 1e-2
        self.pub_imu.publish(m)

    def slow(self):
        b = BatteryState()
        b.header.stamp = self.get_clock().now().to_msg()
        b.percentage, b.voltage, b.present = 1.0, math.nan, True
        self.pub_batt.publish(b)
        h = DiagnosticArray()
        h.header.stamp = b.header.stamp
        h.status.append(DiagnosticStatus(level=DiagnosticStatus.OK, name="x30/replay", message="REPLAY STUB (synthetic)"))
        self.pub_health.publish(h)


def main(args=None):
    rclpy.init(args=args)
    node = ReplayStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
