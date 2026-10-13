import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    nav2_params_arg = DeclareLaunchArgument(
        'nav2_params',
        default_value=os.path.join(
            get_package_share_directory('x30_navigation'), 'config', 'nav2_params.yaml'),
        description='Path to the Nav2 param file')

    nav2_bringup_launch_dir = os.path.join(
        get_package_share_directory('nav2_bringup'), 'launch')

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_launch_dir, 'navigation_launch.py')),
        launch_arguments={
            'params_file': LaunchConfiguration('nav2_params'),
            'use_sim_time': 'true',
        }.items(),
    )

    return LaunchDescription([
        nav2_params_arg,
        nav2_launch,
    ])
