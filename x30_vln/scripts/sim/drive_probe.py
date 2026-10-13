#!/usr/bin/env python3
"""Drive the simulated robot with a constant /cmd_vel and print ground-truth pose, odom and height once per simulated second.
  python3 drive_probe.py --vx 0.3 --vy 0 --wz 0 --seconds 20
"""
import argparse
import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vx', type=float, default=0.3)
    ap.add_argument('--vy', type=float, default=0.0)
    ap.add_argument('--wz', type=float, default=0.0)
    ap.add_argument('--seconds', type=float, default=20.0)
    a = ap.parse_args()
    rclpy.init()
    node = Node('drive_probe')
    state = {'sim': 0.0, 'gt': None, 'odom': None}
    node.create_subscription(Clock, '/clock', lambda m: state.__setitem__('sim', m.clock.sec + m.clock.nanosec * 1e-9), 10)
    node.create_subscription(PoseStamped, '/gt/pose', lambda m: state.__setitem__('gt', m.pose), 10)
    node.create_subscription(Odometry, '/odom', lambda m: state.__setitem__('odom', m.pose.pose), 10)
    pub = node.create_publisher(Twist, '/cmd_vel', 10)
    cmd = Twist()
    cmd.linear.x, cmd.linear.y, cmd.angular.z = a.vx, a.vy, a.wz
    while state['gt'] is None or state['odom'] is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    t0, g0, next_print = state['sim'], state['gt'], 0.0
    while state['sim'] - t0 < a.seconds:
        pub.publish(cmd)
        rclpy.spin_once(node, timeout_sec=0.05)
        if state['sim'] - t0 >= next_print:
            g, o = state['gt'], state['odom']
            yaw = math.atan2(2 * (g.orientation.w * g.orientation.z + g.orientation.x * g.orientation.y),
                             1 - 2 * (g.orientation.y ** 2 + g.orientation.z ** 2))
            print(f't={state["sim"] - t0:5.1f} gt=({g.position.x:6.2f},{g.position.y:6.2f},{g.position.z:5.3f}) yaw={yaw:5.2f} '
                  f'dist={math.hypot(g.position.x - g0.position.x, g.position.y - g0.position.y):5.2f} odom=({o.position.x:6.2f},{o.position.y:6.2f})', flush=True)
            next_print += 2.0
    pub.publish(Twist())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
