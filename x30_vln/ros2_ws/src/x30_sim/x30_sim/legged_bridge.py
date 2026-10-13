"""Legged locomotion backend: /cmd_vel -> unitree_guide_controller (control_input_msgs/Inputs) and its stand-up state machine.

Ported from the prework cmd_vel_bridge. Adds: dead zone, clamp to the controller limits, command timeout, e-stop, and a
hold/resume handshake (/sim/legged/hold) used by /sim/set_robot_pose so the robot is standing (FIXEDSTAND) while it is moved.
The controller state is read from its "Switched from A to B" log lines on /rosout.
FSM (controller sources): PASSIVE -2-> FIXEDDOWN -2-> FIXEDSTAND -4-> TROTTING; TROTTING -2-> FIXEDSTAND.
"""
from __future__ import annotations

import re

import rclpy
from control_input_msgs.msg import Inputs
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from . import config as cfgmod

SWITCH_RE = re.compile(r'Switched from (.+?) to (.+)$')
# state -> (command to send, state it should switch to); each step is retried because ramp states ignore early commands
STARTUP_STEPS = {'passive': (2, 'fixed down'), 'fixed down': (2, 'fixed stand'), 'fixed stand': (4, 'trotting')}


def normalise(value: float, limit: float) -> float:
    return 0.0 if limit <= 0.0 else max(-1.0, min(1.0, value / limit))


class LeggedBridge(Node):
    def __init__(self):
        super().__init__('legged_bridge')
        self.declare_parameter('robot_config', '')
        self.declare_parameter('retry_period', 1.0)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('controller_node', 'unitree_guide_controller')
        cfg = cfgmod.load_robot_config(self.get_parameter('robot_config').value or None)
        lim = cfg['limits']['legged']
        self.vx, self.vy, self.wz = lim['vx'][1], lim['vy'][1], lim['wz'][1]
        self.deadband = cfg['limits']['deadband_fraction']
        self.timeout = cfg['limits']['cmd_timeout_s']
        self.retry = float(self.get_parameter('retry_period').value)
        self.controller = self.get_parameter('controller_node').value
        self.fsm = 'passive'
        self.hold = False
        self.estop = False
        self.twist = Twist()
        self.last_cmd = None
        self.last_step = None
        self.pub = self.create_publisher(Inputs, '/control_input', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.ready = self.create_publisher(Bool, '/x30/ready', latched)
        self.ready.publish(Bool(data=False))
        self.create_subscription(Twist, cfg['topics']['cmd_vel'], self.on_cmd, 10)
        self.create_subscription(Bool, '/x30/estop', self.on_estop, 5)
        self.create_subscription(Bool, '/sim/legged/hold', self.on_hold, 5)
        self.create_subscription(Log, '/rosout', self.on_log, 100)
        self.create_timer(1.0 / float(self.get_parameter('publish_rate').value), self.step)
        self.get_logger().info('legged bridge started: waiting for the controller, will stand up')

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg: Twist) -> None:
        self.twist, self.last_cmd = msg, self.now_s()

    def on_estop(self, msg: Bool) -> None:
        self.estop = bool(msg.data)

    def on_hold(self, msg: Bool) -> None:
        self.hold = bool(msg.data)
        self.last_step = None
        self.get_logger().info(f'hold {"on" if self.hold else "off"}')

    def on_log(self, msg: Log) -> None:
        if msg.name != self.controller:
            return
        m = SWITCH_RE.search(msg.msg)
        if not m:
            return
        self.fsm = m.group(2).strip()
        self.last_step = None
        self.get_logger().info(f'controller state: {self.fsm}')
        self.ready.publish(Bool(data=self.fsm == 'trotting'))

    def step(self) -> None:
        msg = Inputs()
        if self.hold and self.fsm == 'trotting':
            target = (2, 'fixed stand')            # back to the standing state
        elif self.hold and self.fsm == 'fixed stand':
            return
        else:
            target = STARTUP_STEPS.get(self.fsm)
        if target is not None:
            now = self.now_s()
            if self.last_step is None or now - self.last_step >= self.retry:
                msg.command = target[0]
                self.pub.publish(msg)
                self.last_step = now
            return
        if self.fsm != 'trotting':
            return
        t = self.now_s()
        fresh = self.last_cmd is not None and t - self.last_cmd <= self.timeout
        vx = vy = wz = 0.0
        if fresh and not self.estop:
            tw = self.twist
            vx = tw.linear.x if abs(tw.linear.x) >= self.deadband * self.vx else 0.0
            vy = tw.linear.y if abs(tw.linear.y) >= self.deadband * self.vy else 0.0
            wz = tw.angular.z if abs(tw.angular.z) >= self.deadband * self.wz else 0.0
        msg.command = 0
        msg.ly = normalise(vx, self.vx)           # controller mapping (StateTrotting.cpp): ly = vx/limit, lx = -vy/limit, rx = -wz/limit
        msg.lx = normalise(-vy, self.vy)
        msg.rx = normalise(-wz, self.wz)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LeggedBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
