"""Kinematic locomotion backend: shapes /cmd_vel and feeds the Gazebo VelocityControl plugin of the (leg-frozen, gravity-free) model.

Shaping, in order: dead zone (2 % of the limit), clamp to the velocity limits, command timeout (zero when /cmd_vel is silent),
latency (the command takes effect `latency_s` after it was received), acceleration limits, e-stop (instant zero).
Output: geometry_msgs/Twist on /x30/cmd_vel_gz (bridged to /model/x30/cmd_vel), body frame, 50 Hz.
"""
from __future__ import annotations

from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool

from . import config as cfgmod


def shape_axis(value: float, lo: float, hi: float, deadband: float) -> float:
    """Dead zone then clamp. deadband is the absolute dead-zone half width."""
    if abs(value) < deadband:
        return 0.0
    return max(lo, min(hi, value))


def slew(current: float, target: float, max_step: float) -> float:
    return current + max(-max_step, min(max_step, target - current))


class KinematicDriver(Node):
    def __init__(self):
        super().__init__('kinematic_driver')
        self.declare_parameter('robot_config', '')
        self.declare_parameter('latency_s', -1.0)         # negative: use the config value
        self.declare_parameter('rate_hz', 50.0)
        cfg = cfgmod.load_robot_config(self.get_parameter('robot_config').value or None)
        lim = cfg['limits']['kinematic']
        self.lim = {'x': lim['vx'], 'y': lim['vy'], 'z': lim['wz']}
        self.acc_lin, self.acc_ang = lim['accel_linear'], lim['accel_angular']
        lat = float(self.get_parameter('latency_s').value)
        self.latency = lim['latency_s'] if lat < 0 else lat
        self.deadband = cfg['limits']['deadband_fraction']
        self.timeout = cfg['limits']['cmd_timeout_s']
        self.rate = float(self.get_parameter('rate_hz').value)
        self.queue: deque[tuple[float, tuple[float, float, float]]] = deque()
        self.target = (0.0, 0.0, 0.0)
        self.cur = [0.0, 0.0, 0.0]
        self.last_cmd_t = None
        self.estop = False
        self.create_subscription(Twist, cfg['topics']['cmd_vel'], self.on_cmd, 10)
        self.create_subscription(Bool, '/x30/estop', self.on_estop, 5)
        self.pub = self.create_publisher(Twist, '/x30/cmd_vel_gz', 10)
        self.create_timer(1.0 / self.rate, self.step)
        self.get_logger().info(f'kinematic locomotion: limits {self.lim}, latency {self.latency:.3f} s, '
                               f'accel {self.acc_lin}/{self.acc_ang}')

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_estop(self, msg: Bool) -> None:
        self.estop = bool(msg.data)

    def on_cmd(self, msg: Twist) -> None:
        t = self.now_s()
        v = (shape_axis(msg.linear.x, *self.lim['x'], self.deadband * self.lim['x'][1]),
             shape_axis(msg.linear.y, *self.lim['y'], self.deadband * self.lim['y'][1]),
             shape_axis(msg.angular.z, *self.lim['z'], self.deadband * self.lim['z'][1]))
        self.queue.append((t + self.latency, v))
        self.last_cmd_t = t

    def step(self) -> None:
        t = self.now_s()
        while self.queue and self.queue[0][0] <= t:
            self.target = self.queue.popleft()[1]
        if self.last_cmd_t is None or t - self.last_cmd_t > self.timeout + self.latency:
            self.target = (0.0, 0.0, 0.0)
        dt = 1.0 / self.rate
        if self.estop:
            self.cur = [0.0, 0.0, 0.0]
        else:
            self.cur[0] = slew(self.cur[0], self.target[0], self.acc_lin * dt)
            self.cur[1] = slew(self.cur[1], self.target[1], self.acc_lin * dt)
            self.cur[2] = slew(self.cur[2], self.target[2], self.acc_ang * dt)
        out = Twist()
        out.linear.x, out.linear.y, out.angular.z = self.cur
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = KinematicDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
