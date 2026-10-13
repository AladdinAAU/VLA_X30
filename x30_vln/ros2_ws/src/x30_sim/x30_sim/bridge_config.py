"""ros_gz_bridge configuration (yaml text) for the sensor and state topics of the simulator."""
from __future__ import annotations

import yaml


def _entry(ros: str, gz: str, ros_type: str, gz_type: str, direction: str = 'GZ_TO_ROS') -> dict:
    return {'ros_topic_name': ros, 'gz_topic_name': gz, 'ros_type_name': ros_type, 'gz_type_name': gz_type, 'direction': direction}


def bridge_entries(cfg: dict, locomotion: str, thermal: bool, model: str = 'x30') -> list[dict]:
    t = cfg['topics']
    e = [
        _entry('/clock', '/clock', 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock'),
        _entry(t['imu'], '/imu_ros', 'sensor_msgs/msg/Imu', 'gz.msgs.IMU'),
        _entry(t['rgb'], '/camera/image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'),
        _entry(t['rgb_info'], '/camera/camera_info', 'sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo'),
        _entry('/gt/odom_raw', '/gt/odom_raw', 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry'),
    ]
    for key in cfg['lidar']['mounts']:
        e.append(_entry(f'/lidar/{key}/points', f'/lidar/{key}/scan/points', 'sensor_msgs/msg/PointCloud2', 'gz.msgs.PointCloudPacked'))
    if cfg['camera']['depth']:
        e.append(_entry(t['depth'], '/camera/depth_image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'))
    if thermal:
        e.append(_entry('/thermal/raw', '/thermal/image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'))
        e.append(_entry(t['thermal_info'], '/thermal/camera_info', 'sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo'))
    if locomotion == 'kinematic':
        e.append(_entry('/x30/cmd_vel_gz', f'/model/{model}/cmd_vel', 'geometry_msgs/msg/Twist', 'gz.msgs.Twist', 'ROS_TO_GZ'))
    return e


def bridge_yaml(cfg: dict, locomotion: str, thermal: bool, model: str = 'x30') -> str:
    return yaml.safe_dump(bridge_entries(cfg, locomotion, thermal, model), sort_keys=False)
