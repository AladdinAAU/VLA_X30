#!/usr/bin/env python3
"""
Bridges geometry_msgs/Twist on /cmd_vel into the control_input_msgs/Inputs
message unitree_guide_controller listens for on /control_input, and drives
its FSM from PASSIVE -> FIXEDSTAND -> TROTTING on startup.

Field mapping derived from unitree_guide_controller's StateTrotting.cpp
(invNormalize with symmetric limits reduces to a simple division):
  v_x_limit  = [-0.4, 0.4] m/s  -> ly =  vx / 0.4
  v_y_limit  = [-0.3, 0.3] m/s  -> lx = -vy / 0.3
  w_yaw_limit= [-0.5, 0.5] rad/s -> rx = -wz / 0.5

Startup FSM sequence (state-driven, see STARTUP_STEPS), per the actual
checkChange() transitions in
controller_common/FSM/{StatePassive,BaseFixedStand}.cpp and
unitree_guide_controller/FSM/{StateFixedDown,StateFixedStand,StateTrotting}.cpp
(the controller activates into PASSIVE, which only reacts to command=2):
  PASSIVE --(2)--> FIXEDDOWN --(2)--> FIXEDSTAND --(4)--> TROTTING
Each ramp state (FixedDown/FixedStand) ignores new commands until its stand-up
interpolation reaches percent_ >= 1.5, i.e. ctrl_interfaces_.frequency_ * 1.2
control cycles -> ~1.8s at the controller's 200Hz update rate. A command sent
earlier is ignored, so the bridge retries each step until the controller reports
the switch.
"""
import re

import rclpy
from rcl_interfaces.msg import Log
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool
from control_input_msgs.msg import Inputs

VX_LIMIT = 0.4
VY_LIMIT = 0.3
WZ_LIMIT = 0.5


def clamp(value: float, limit: float) -> float:
    if limit <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, value / limit))


# Startup is driven by the controller's own state, which it reports on /rosout as
# "Switched from <a> to <b>". Timed one-shot commands are lost whenever the
# controller is slow to consume them (the ramp states also ignore commands until
# their stand-up interpolation finishes), so each step is retried until the
# controller confirms the switch.
#   state -> (command to send, state it should switch to)
STARTUP_STEPS = {
    'passive': (2, 'fixed down'),
    'fixed down': (2, 'fixed stand'),
    'fixed stand': (4, 'trotting'),
}
SWITCH_RE = re.compile(r'Switched from (.+?) to (.+)$')


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        self.declare_parameter('retry_period', 1.0)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('controller_node', 'unitree_guide_controller')

        self.pub_ = self.create_publisher(Inputs, '/control_input', 10)
        # Latched: true once the FSM is in TROTTING and /cmd_vel is actually
        # acted on. Goal senders can wait for this.
        ready_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                               reliability=ReliabilityPolicy.RELIABLE)
        self.ready_pub_ = self.create_publisher(Bool, '/x30/ready', ready_qos)
        self.ready_pub_.publish(Bool(data=False))
        self.sub_ = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)
        self.create_subscription(Log, '/rosout', self.rosout_cb, 100)

        self.latest_twist_ = Twist()
        self.controller_node_ = self.get_parameter('controller_node').value
        self.retry_period_ = float(self.get_parameter('retry_period').value)
        self.fsm_state_ = 'passive'  # controller activates into PASSIVE
        self.last_cmd_time_ = None

        rate = self.get_parameter('publish_rate').value
        self.cmd_timer_ = self.create_timer(1.0 / rate, self.publish_cmd)

        self.get_logger().info('cmd_vel_bridge started: waiting for controller, will stand up')

    def cmd_vel_cb(self, msg: Twist):
        self.latest_twist_ = msg

    def rosout_cb(self, msg: Log):
        if msg.name != self.controller_node_:
            return
        m = SWITCH_RE.search(msg.msg)
        if not m:
            return
        self.fsm_state_ = m.group(2).strip()
        self.last_cmd_time_ = None
        self.get_logger().info('Controller state: %s' % self.fsm_state_)
        if self.fsm_state_ == 'trotting':
            self.ready_pub_.publish(Bool(data=True))
            self.get_logger().info('Robot is trotting - ready for cmd_vel')
        else:
            self.ready_pub_.publish(Bool(data=False))

    def publish_cmd(self):
        msg = Inputs()
        step = STARTUP_STEPS.get(self.fsm_state_)
        if step is not None:
            now = self.get_clock().now()
            if (self.last_cmd_time_ is None
                    or (now - self.last_cmd_time_).nanoseconds * 1e-9 >= self.retry_period_):
                msg.command = step[0]
                self.pub_.publish(msg)
                self.last_cmd_time_ = now
                self.get_logger().info(
                    'In %s: sending command=%d (-> %s)' % (self.fsm_state_, step[0], step[1]))
            return
        if self.fsm_state_ != 'trotting':
            return  # unexpected state (e.g. swing test): don't drive it
        msg.command = 0
        msg.ly = clamp(self.latest_twist_.linear.x, VX_LIMIT)
        msg.lx = clamp(-self.latest_twist_.linear.y, VY_LIMIT)
        msg.rx = clamp(-self.latest_twist_.angular.z, WZ_LIMIT)
        msg.ry = 0.0
        self.pub_.publish(msg)


def main():
    rclpy.init()
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
