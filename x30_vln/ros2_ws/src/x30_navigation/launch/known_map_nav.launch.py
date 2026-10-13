"""Known-map navigation (no SLAM / Nav2): known_map_nav + waypoint_runner (+ cmd_relay when no x30_modes mux owns /cmd_vel).

ros2 launch x30_navigation known_map_nav.launch.py world:=lab_01 [use_sim_time:=true] [relay:=true] [align:=gt|spawn]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def generate_launch_description():
    args = [("world", "lab_01"), ("use_sim_time", "true"), ("relay", "true"), ("align", "gt"), ("margin", "0.05")]
    sim = {"use_sim_time": LC("use_sim_time")}
    return LaunchDescription([DeclareLaunchArgument(n, default_value=v) for n, v in args] + [
        Node(package="x30_navigation", executable="known_map_nav", output="screen",
             parameters=[sim, {"world": LC("world"), "align": LC("align"), "margin": LC("margin")}]),
        Node(package="x30_navigation", executable="waypoint_runner", output="screen",
             parameters=[sim, {"planner_backend": "known_map"}]),
        Node(package="x30_navigation", executable="next_viewpoint", output="screen", parameters=[sim]),
        Node(package="x30_navigation", executable="cmd_relay", condition=IfCondition(LC("relay")), parameters=[sim]),
    ])
