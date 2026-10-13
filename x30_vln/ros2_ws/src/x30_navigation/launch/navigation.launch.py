"""Full navigation stack: odometry + selector, mapping, (Nav2), (frontier explorer).

ros2 launch x30_navigation navigation.launch.py profile:=sim|real|replay [odom_source:=gt|leg|lidar|fused]
    [slam:=occupancy|slam_toolbox|none] [nav2:=true] [explore:=false] [limits:=default|legged] [controller:=mppi|rpp]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC, PythonExpression
from launch_ros.actions import Node

from x30_navigation.launch_utils import config_dir, node_param_files


def _nodes(context):
    """direct_navigator (planner_backend:=direct) and waypoint_runner (always, when nav2:=true)."""
    get = lambda k: LC(k).perform(context)
    sim = {"use_sim_time": get("use_sim_time").lower() == "true"}
    out = [Node(package="x30_navigation", executable="waypoint_runner", name="waypoint_runner", output="screen",
                parameters=[sim, {"planner_backend": get("planner_backend")}])]
    if get("planner_backend") == "direct":
        out.append(Node(package="x30_navigation", executable="direct_navigator", name="direct_navigator", output="screen",
                        parameters=node_param_files(config_dir(get("config_dir")), get("profile")) + [sim]))
    return out if get("nav2") == "true" else []


def generate_launch_description():
    share = get_package_share_directory("x30_navigation")

    def include(name, args, condition=None):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share, "launch", name)),
                                        launch_arguments=args.items(), condition=condition)

    common = {"profile": LC("profile"), "use_sim_time": LC("use_sim_time"), "config_dir": LC("config_dir")}
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="sim", description="sim | real | replay"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("odom_tf", default_value="true", description="false: another node (the Gazebo sim) already publishes odom -> base_link"),
        DeclareLaunchArgument("config_dir", default_value=""),
        DeclareLaunchArgument("odom_source", default_value="", description="gt|leg|lidar|fused; empty = profile default"),
        DeclareLaunchArgument("slam", default_value="occupancy"),
        DeclareLaunchArgument("nav2", default_value="true"),
        DeclareLaunchArgument("planner_backend", default_value="nav2", description="nav2 | direct (Nav2-free tracker, same navigate_to_pose action)"),
        DeclareLaunchArgument("explore", default_value="false", description="start the autonomous frontier loop"),
        DeclareLaunchArgument("limits", default_value="default"),
        DeclareLaunchArgument("controller", default_value="mppi"),
        DeclareLaunchArgument("static_lidar_tf", default_value="false"),
        DeclareLaunchArgument("lidar_xyzrpy", default_value="0 0 0 0 0 0"),
        include("odometry.launch.py", {**common, "odom_source": LC("odom_source"), "odom_tf": LC("odom_tf"), "static_lidar_tf": LC("static_lidar_tf"),
                                       "lidar_xyzrpy": LC("lidar_xyzrpy")}),
        include("mapping.launch.py", {**common, "slam": LC("slam")}),
        include("nav2.launch.py", {**common, "limits": LC("limits"), "controller": LC("controller")}, IfCondition(PythonExpression(["'", LC("nav2"), "' == 'true' and '", LC("planner_backend"), "' == 'nav2'"]))),
        include("explore.launch.py", {**common, "autonomous": LC("explore")}, IfCondition(LC("nav2"))),
        OpaqueFunction(function=_nodes),
    ])
