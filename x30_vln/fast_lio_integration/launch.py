"""Run one Livox device's native points through FAST-LIO."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        ExecuteProcess(cmd=["python3", "/opt/fast_lio/select_lidar.py"], output="screen"),
        Node(
            package="fast_lio", executable="fastlio_mapping", output="screen",
            parameters=["/etc/fast_lio.yaml", {"use_sim_time": sim_time}],
        ),
    ])
