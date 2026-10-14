"""Synthetic ROS 1 inputs for proof_livox_bridge.sh; never commands motion."""
import struct
import time

import rospy
from geometry_msgs.msg import TransformStamped
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from sensor_msgs.msg import Imu, PointCloud2, PointField
from tf2_msgs.msg import TFMessage

rospy.init_node("livox_bridge_proof_source")
publishers = {
    "custom": rospy.Publisher("/livox/lidar", CustomMsg, queue_size=5),
    "cloud": rospy.Publisher("/lidar_points", PointCloud2, queue_size=5),
    "tf": rospy.Publisher("/tf", TFMessage, queue_size=100),
    "static": rospy.Publisher("/tf_static", TFMessage, queue_size=100, latch=True),
}
imu_topics = ["/imu/data"] + [f"/livox/imu_192_168_2_{i}" for i in range(202, 206)]
imu_publishers = [rospy.Publisher(topic, Imu, queue_size=200) for topic in imu_topics]
time.sleep(3)

static = TransformStamped()
static.header.frame_id = "base_link"
static.child_frame_id = "lidar_link"
static.transform.translation.x = 0.25
static.transform.rotation.w = 1.0
publishers["static"].publish(TFMessage([static]))
print("Published static TF once", flush=True)

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    custom = CustomMsg()
    custom.header.stamp = rospy.Time(1790000000, 123456789)
    custom.header.frame_id = "livox_frame"
    custom.timebase = 1790000000123456789
    custom.lidar_id = 7
    custom.rsvd = [0, 1, 255]
    custom.points = [
        CustomPoint(offset_time=0, x=1.25, y=-2.5, z=0.5, reflectivity=255, tag=16, line=0),
        CustomPoint(offset_time=100000001, x=-3.5, y=4.25, z=2.0, reflectivity=17, tag=0, line=15),
    ]
    custom.point_num = len(custom.points)
    publishers["custom"].publish(custom)

    cloud = PointCloud2()
    cloud.header.stamp = rospy.Time.now()
    cloud.header.frame_id = "lidar_link"
    cloud.height, cloud.width = 1, 2
    cloud.fields = [PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(("x", "y", "z", "intensity"))]
    cloud.point_step, cloud.row_step = 16, 32
    cloud.is_dense = True
    cloud.data = struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 8)
    publishers["cloud"].publish(cloud)

    imu = Imu()
    imu.header.stamp = custom.header.stamp
    imu.header.frame_id = "imu_link"
    imu.orientation.w = 1.0
    imu.angular_velocity.x = 0.25
    imu.linear_acceleration.z = 9.75
    for publisher in imu_publishers:
        publisher.publish(imu)

    dynamic = TransformStamped()
    dynamic.header.stamp = cloud.header.stamp
    dynamic.header.frame_id = "odom"
    dynamic.child_frame_id = "base_link"
    dynamic.transform.translation.x = 1.0
    dynamic.transform.rotation.w = 1.0
    publishers["tf"].publish(TFMessage([dynamic]))
    rate.sleep()
