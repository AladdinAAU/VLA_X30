"""Validate native Livox fields and the existing cloud/IMU/TF paths in Humble."""
import json
import struct
import time

import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

rclpy.init()
node = rclpy.create_node("livox_bridge_proof_receiver")
imu_topics = ["/imu/data"] + [f"/livox/imu_192_168_2_{i}" for i in range(202, 206)]
counts = {topic: 0 for topic in ["/livox/lidar", "/lidar_points", "/tf", "/tf_static"] + imu_topics}

def on_custom(msg):
    assert msg.header.frame_id == "livox_frame"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (1790000000, 123456789)
    assert msg.timebase == 1790000000123456789
    assert msg.lidar_id == 7 and list(msg.rsvd) == [0, 1, 255]
    assert msg.point_num == len(msg.points) == 2
    expected = [(0, 1.25, -2.5, 0.5, 255, 16, 0), (100000001, -3.5, 4.25, 2.0, 17, 0, 15)]
    for point, value in zip(msg.points, expected):
        assert (point.offset_time, point.x, point.y, point.z, point.reflectivity, point.tag, point.line) == value
    counts["/livox/lidar"] += 1

def on_cloud(msg):
    assert msg.header.frame_id == "lidar_link" and msg.width == 2
    assert struct.unpack("<8f", bytes(msg.data)) == (1, 2, 3, 4, 5, 6, 7, 8)
    counts["/lidar_points"] += 1

def on_imu(msg, topic):
    assert msg.header.frame_id == "imu_link"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (1790000000, 123456789)
    assert (msg.orientation.w, msg.angular_velocity.x, msg.linear_acceleration.z) == (1.0, 0.25, 9.75)
    counts[topic] += 1

def on_tf(msg, topic):
    counts[topic] += 1

node.create_subscription(CustomMsg, "/livox/lidar", on_custom, 5)
node.create_subscription(PointCloud2, "/lidar_points", on_cloud, 5)
for topic in imu_topics:
    node.create_subscription(Imu, topic, lambda msg, topic=topic: on_imu(msg, topic), 200)
node.create_subscription(TFMessage, "/tf", lambda msg: on_tf(msg, "/tf"), 100)
node.create_subscription(TFMessage, "/tf_static",
                         lambda msg: on_tf(msg, "/tf_static"),
                         QoSProfile(history=HistoryPolicy.KEEP_ALL, reliability=ReliabilityPolicy.RELIABLE,
                                    durability=DurabilityPolicy.TRANSIENT_LOCAL))
buffer = Buffer()
listener = TransformListener(buffer, node)
deadline = time.monotonic() + 20
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)
    if all(value >= (1 if topic == "/tf_static" else 3) for topic, value in counts.items()):
        if buffer.can_transform("odom", "lidar_link", rclpy.time.Time()):
            break
assert all(value >= (1 if topic == "/tf_static" else 3) for topic, value in counts.items()), counts
transform = buffer.lookup_transform("odom", "lidar_link", rclpy.time.Time())
assert transform.transform.translation.x == 1.25
assert "/cmd_vel" not in dict(node.get_topic_names_and_types())
print(json.dumps({"status": "passed", "received": counts, "custom_fields_preserved": True,
                  "late_static_tf": True}, indent=2))
node.destroy_node()
rclpy.shutdown()
