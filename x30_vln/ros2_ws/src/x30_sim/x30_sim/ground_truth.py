"""Latched ground truth for evaluation: /gt/objects (SemanticObjectArray, world frame) and /gt/scene_info (JSON String).

Objects come from the world SDF (labels by the convention in sdf_scene.py); they are static, so they are published once with
TRANSIENT_LOCAL durability and again on /sim/reset_world.
"""
from __future__ import annotations

import json

import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from x30_msgs.msg import SemanticObject, SemanticObjectArray

from . import config as cfgmod
from . import worlds
from .sdf_scene import parse_scene


class GroundTruth(Node):
    def __init__(self):
        super().__init__('ground_truth')
        self.declare_parameter('robot_config', '')
        self.declare_parameter('world', worlds.BUILTIN_WORLD)
        self.declare_parameter('registry', '')
        cfg = cfgmod.load_robot_config(self.get_parameter('robot_config').value or None)
        spec = worlds.resolve_world(self.get_parameter('world').value, self.get_parameter('registry').value or None)
        pkg = worlds.worlds_package_dir()
        labels = pkg / 'config' / 'labels.yaml' if pkg else None
        objects, info = parse_scene(spec, labels)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_obj = self.create_publisher(SemanticObjectArray, '/gt/objects', qos)
        self.pub_info = self.create_publisher(String, '/gt/scene_info', qos)
        arr = SemanticObjectArray()
        arr.header.frame_id = cfg['robot']['world_frame']
        for i, o in enumerate(objects, start=1):
            m = SemanticObject()
            m.id = i
            m.label = o.label
            m.confidence = float(cfg['ground_truth']['object_confidence'])
            m.position.x, m.position.y, m.position.z = o.position
            m.size.x, m.size.y, m.size.z = o.size
            m.caption = o.name                       # the instance name (chair_3) rides in the caption field
            m.attributes = list(o.attributes)
            m.first_seen = m.last_seen = Time()
            arr.objects.append(m)
        self.pub_obj.publish(arr)
        self.pub_info.publish(String(data=json.dumps(info)))
        self.get_logger().info(f'ground truth: {len(objects)} objects in "{spec.world_id}" ({info["n_models"]} models)')


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
