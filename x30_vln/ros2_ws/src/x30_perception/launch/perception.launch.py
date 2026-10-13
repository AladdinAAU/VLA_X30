"""ros2 launch x30_perception perception.launch.py [use_sim_time:=true] [server_url:=...] [depth_source:=lidar]"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('x30_perception')
    default_cfg = os.path.join(os.environ.get('X30_CONFIGS', '/x30/configs'), 'perception.yaml')
    args = [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('params_file', default_value=os.path.join(share, 'config', 'perception_client.yaml')),
        DeclareLaunchArgument('config_file', default_value=default_cfg, description='configs/perception.yaml'),
        DeclareLaunchArgument(
            'server_url',
            default_value=os.environ.get('PERCEPTION_URL', ''),
            description="'' = use PERCEPTION_URL or configs/perception.yaml",
        ),
        DeclareLaunchArgument('depth_source', default_value='lidar', description='lidar | sim_depth | none'),
        DeclareLaunchArgument('rate_hz', default_value='2.5'),
    ]
    node = Node(
        package='x30_perception',
        executable='perception_client',
        name='perception_client',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'config_file': LaunchConfiguration('config_file'),
                'server_url': LaunchConfiguration('server_url'),
                'depth_source': LaunchConfiguration('depth_source'),
                'rate_hz': LaunchConfiguration('rate_hz'),
            },
        ],
    )
    return LaunchDescription(args + [node])
