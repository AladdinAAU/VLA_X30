"""
Records a rosbag2 of the topics you'd want for understanding a run and
replaying/analysing it later offline: sensor data, localization/mapping
output, planned paths, velocity commands and the full TF tree.

    ros2 launch x30_data_capture record_bag.launch.py
    ros2 launch x30_data_capture record_bag.launch.py bag_name:=arena_run_01
"""
import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


TOPICS = [
    '/scan',
    '/imu',
    '/camera/image',
    '/camera/depth_image',
    '/camera/camera_info',
    '/camera/points',
    '/odom',
    '/tf',
    '/tf_static',
    '/map',
    '/plan',
    '/cmd_vel',
    '/control_input',
    '/global_costmap/costmap',
    '/local_costmap/costmap',
]


def generate_launch_description():
    default_name = 'x30_run_' + datetime.now().strftime('%Y%m%d_%H%M%S')

    bag_name_arg = DeclareLaunchArgument('bag_name', default_value=default_name)
    output_dir_arg = DeclareLaunchArgument(
        'output_dir', default_value=os.path.join(os.path.expanduser('~'), 'x30_bags'))

    record_process = ExecuteProcess(
        cmd=['ros2', 'bag', 'record',
             '-o', PathJoinSubstitution([LaunchConfiguration('output_dir'), LaunchConfiguration('bag_name')]),
             '--use-sim-time'] + TOPICS,
        output='screen',
    )

    return LaunchDescription([
        bag_name_arg,
        output_dir_arg,
        record_process,
    ])
