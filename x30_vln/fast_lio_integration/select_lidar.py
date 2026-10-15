"""Keep a single Livox device when /livox/lidar multiplexes several IDs."""
import os

import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.qos import QoSProfile, ReliabilityPolicy


def main():
    rclpy.init()
    node = rclpy.create_node("fast_lio_select_lidar")
    selected = os.environ.get("LIVOX_LIDAR_ID", "auto")
    if selected != "auto":
        selected = int(selected)
        if not 0 <= selected <= 255:
            raise ValueError("LIVOX_LIDAR_ID must be 0..255")
    pub = node.create_publisher(CustomMsg, "/fast_lio/lidar", QoSProfile(depth=5))

    def receive(msg):
        nonlocal selected
        if selected == "auto":
            selected = msg.lidar_id
            node.get_logger().warning(f"Using first Livox lidar_id={selected}; set LIVOX_LIDAR_ID explicitly for repeatable runs")
        if msg.lidar_id == selected:
            pub.publish(msg)

    node.create_subscription(
        CustomMsg, "/livox/lidar", receive,
        QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE),
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
