"""lidar_odometry + odom_selector. For replay without a robot description, `static_lidar_tf:=true` publishes
base_link -> lidar_link from `lidar_xyzrpy` (identity by default: the bag has no calibration, see the report)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from x30_navigation.launch_utils import config_dir, node_param_files


def _setup(context):
    cfg = config_dir(LaunchConfiguration("config_dir").perform(context))
    profile = LaunchConfiguration("profile").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    odom_source = LaunchConfiguration("odom_source").perform(context)
    params = node_param_files(cfg, profile) + [{"use_sim_time": use_sim_time}]
    odom_tf = LaunchConfiguration("odom_tf").perform(context).lower() == "true"
    sel_params = params + ([{"odom_source": odom_source}] if odom_source else []) + [{"publish_tf": odom_tf}]
    nodes = [Node(package="x30_navigation", executable="lidar_odometry", name="lidar_odometry", output="screen", parameters=params),
             Node(package="x30_navigation", executable="odom_selector", name="odom_selector", output="screen", parameters=sel_params)]
    if LaunchConfiguration("static_lidar_tf").perform(context).lower() == "true":
        xyzrpy = [s for s in LaunchConfiguration("lidar_xyzrpy").perform(context).split()]
        x, y, z, r, p, yaw = xyzrpy
        nodes.append(Node(package="tf2_ros", executable="static_transform_publisher", name="lidar_static_tf",
                          arguments=["--x", x, "--y", y, "--z", z, "--roll", r, "--pitch", p, "--yaw", yaw,
                                     "--frame-id", "base_link", "--child-frame-id", "lidar_link"]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="sim", description="sim | real | replay"),
        DeclareLaunchArgument("odom_source", default_value="", description="gt|leg|lidar|fused; empty = profile default"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("odom_tf", default_value="true", description="false: another node (the Gazebo sim) already publishes odom -> base_link"),
        DeclareLaunchArgument("config_dir", default_value=""),
        DeclareLaunchArgument("static_lidar_tf", default_value="false"),
        DeclareLaunchArgument("lidar_xyzrpy", default_value="0 0 0 0 0 0", description="pose of lidar_link in base_link"),
        OpaqueFunction(function=_setup),
    ])
