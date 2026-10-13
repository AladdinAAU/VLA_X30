"""static_tf_publisher: publishes the sensor extrinsics of configs/robot_real.yaml as static TF.

Optional calibration overlay (YAML written by calibrate_lidar_camera): {camera_link: {xyz: [...], rpy_deg: [...]}, ...}
replaces the config values for those frames. Frames whose status is 'assumed' or 'calibrate' are reported at start-up.
"""

import yaml
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from x30_hw import config as cfgmod
from x30_hw.transforms import build_transforms


class StaticTfPublisher(Node):
    def __init__(self):
        super().__init__("static_tf_publisher")
        self.declare_parameter("config_file", "")
        self.declare_parameter("overlay_file", "")
        cfg = cfgmod.load_robot_config(self.get_parameter("config_file").value or None)
        overlay = None
        ov_path = self.get_parameter("overlay_file").value
        if ov_path:
            with open(ov_path, "r", encoding="utf-8") as f:
                overlay = yaml.safe_load(f)
            self.get_logger().info(f"calibration overlay {ov_path}: {sorted(overlay)}")
        self.br = StaticTransformBroadcaster(self)
        msgs = []
        for parent, child, xyz, q, status in build_transforms(cfg, overlay):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id, t.child_frame_id = parent, child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
            msgs.append(t)
            if status in ("assumed", "calibrate"):
                self.get_logger().warn(f"TF {parent} -> {child}: status '{status}' (not measured/calibrated)")
        self.br.sendTransform(msgs)
        self.get_logger().info(f"published {len(msgs)} static transforms")


def main(args=None):
    rclpy.init(args=args)
    node = StaticTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
