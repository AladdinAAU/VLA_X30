from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        Node(package="x30_teach", executable="teach_node", name="teach", output="screen",
             parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}]),
    ])
