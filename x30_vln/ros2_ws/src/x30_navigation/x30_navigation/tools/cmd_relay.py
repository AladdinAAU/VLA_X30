"""cmd_relay: forward Nav2's /cmd_vel_nav to /cmd_vel. Only for running navigation against the simulator without x30_modes
(whose mux and safety guard own /cmd_vel in the full system). Not used on the real robot."""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class Relay(Node):
    def __init__(self) -> None:
        super().__init__("cmd_relay")
        self.declare_parameter("input", "/cmd_vel_nav")
        self.declare_parameter("output", "/cmd_vel")
        self.pub = self.create_publisher(Twist, self.get_parameter("output").value, 10)
        self.create_subscription(Twist, self.get_parameter("input").value, self.pub.publish, 10)


def main(args=None) -> None:
    rclpy.init(args=args)
    rclpy.spin(Relay())


if __name__ == "__main__":
    main()
