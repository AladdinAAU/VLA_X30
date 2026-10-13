from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value='', description='configs/semantic_map.yaml (empty = search)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('vlm_enabled', default_value='auto', description='auto | true | false'),
        Node(package='x30_semantic_map', executable='semantic_mapper', name='semantic_mapper', output='screen',
             parameters=[{'config_file': LaunchConfiguration('config_file'),
                          'use_sim_time': LaunchConfiguration('use_sim_time'),
                          'vlm_enabled': LaunchConfiguration('vlm_enabled')}]),
    ])
