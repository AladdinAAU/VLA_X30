"""Standalone launch of x30_nav_bridge with the package parameter file (x30_hw real.launch.py is the normal entry)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(get_package_share_directory('x30_nav_bridge'), 'config', 'x30_params.yaml')
    return LaunchDescription([
        Node(package='x30_nav_bridge', executable='nav_bridge_node', name='nav_bridge_node', output='screen',
             parameters=[params_file], emulate_tty=True),
    ])
