"""cloud_leveler + cloud_to_scan + obstacle_cloud + mapping backend (slam: occupancy | slam_toolbox | none)."""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from x30_navigation.launch_utils import config_dir, node_param_files


def _setup(context):
    cfg = config_dir(LaunchConfiguration("config_dir").perform(context))
    profile = LaunchConfiguration("profile").perform(context)
    slam = LaunchConfiguration("slam").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    params = node_param_files(cfg, profile) + [{"use_sim_time": use_sim_time}]
    nodes = [Node(package="x30_navigation", executable=exe, name=exe, output="screen", parameters=params)
             for exe in ("cloud_leveler", "cloud_to_scan", "obstacle_cloud")]
    if slam == "occupancy":
        nodes.append(Node(package="x30_navigation", executable="occupancy_mapper", name="occupancy_mapper",
                          output="screen", parameters=params))
    elif slam == "slam_toolbox":
        nodes.append(Node(package="slam_toolbox", executable="async_slam_toolbox_node", name="slam_toolbox", output="screen",
                          parameters=[os.path.join(cfg, "slam_toolbox.yaml"), {"use_sim_time": use_sim_time}]))
    elif slam == "none":
        nodes.append(Node(package="tf2_ros", executable="static_transform_publisher", name="map_odom_identity",
                       arguments=["0", "0", "0", "0", "0", "0", "map", "odom"]))
    else:
        raise RuntimeError(f"slam must be occupancy|slam_toolbox|none, got {slam}")
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="sim", description="sim | real | replay"),
        DeclareLaunchArgument("slam", default_value="occupancy", description="occupancy | slam_toolbox | none"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("config_dir", default_value="", description="configs/navigation (default $X30_CONFIGS/navigation)"),
        OpaqueFunction(function=_setup),
    ])
