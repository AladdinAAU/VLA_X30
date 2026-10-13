"""frontier_explorer (service /nav/next_frontier always; autonomous loop with autonomous:=true)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from x30_navigation.launch_utils import config_dir, node_param_files


def _setup(context):
    cfg = config_dir(LaunchConfiguration("config_dir").perform(context))
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    autonomous = LaunchConfiguration("autonomous").perform(context).lower() == "true"
    params = node_param_files(cfg, LaunchConfiguration("profile").perform(context)) + [
        {"use_sim_time": use_sim_time, "autonomous": autonomous}]
    return [Node(package="x30_navigation", executable="frontier_explorer", name="frontier_explorer", output="screen", parameters=params)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="sim"),
        DeclareLaunchArgument("autonomous", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("config_dir", default_value=""),
        OpaqueFunction(function=_setup),
    ])
