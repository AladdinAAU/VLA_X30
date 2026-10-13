"""Nav2 (Humble) for the X30. Output on /cmd_vel_nav only (controller -> /cmd_vel_nav_raw -> velocity_smoother -> /cmd_vel_nav).

Arguments: profile (sim|real|replay), limits (default|legged; default: legged for sim locomotion:=legged else default),
controller (mppi|rpp), use_sim_time, config_dir.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from x30_navigation.launch_utils import config_dir

LIFECYCLE = ["controller_server", "smoother_server", "planner_server", "behavior_server", "bt_navigator",
             "waypoint_follower", "velocity_smoother"]


def _setup(context):
    cfg = config_dir(LaunchConfiguration("config_dir").perform(context))
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    limits = LaunchConfiguration("limits").perform(context)
    controller = LaunchConfiguration("controller").perform(context)
    bt_dir = os.path.join(get_package_share_directory("x30_navigation"), "behavior_trees")
    bt = {"default_nav_to_pose_bt_xml": os.path.join(bt_dir, "navigate_to_pose_legged.xml"),
          "default_nav_through_poses_bt_xml": os.path.join(bt_dir, "navigate_through_poses_legged.xml")}
    files = [os.path.join(cfg, "nav2_params.yaml"), os.path.join(cfg, f"limits_{limits}.yaml")]
    if controller == "rpp":
        files.append(os.path.join(cfg, "controller_rpp.yaml"))
    extra = {"use_sim_time": use_sim_time}
    params = files + [extra]
    raw = ("cmd_vel", "cmd_vel_nav_raw")

    def nav2(pkg, exe, name, remaps=(), extra_params=None):
        return Node(package=pkg, executable=exe, name=name, output="screen", parameters=params + ([extra_params] if extra_params else []),
                    remappings=list(remaps))

    return [
        nav2("nav2_controller", "controller_server", "controller_server", [raw]),
        nav2("nav2_smoother", "smoother_server", "smoother_server"),
        nav2("nav2_planner", "planner_server", "planner_server"),
        nav2("nav2_behaviors", "behavior_server", "behavior_server", [raw]),
        nav2("nav2_bt_navigator", "bt_navigator", "bt_navigator", extra_params=bt),
        nav2("nav2_waypoint_follower", "waypoint_follower", "waypoint_follower"),
        nav2("nav2_velocity_smoother", "velocity_smoother", "velocity_smoother",
             [("cmd_vel", "cmd_vel_nav_raw"), ("cmd_vel_smoothed", "cmd_vel_nav")]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_navigation", output="screen",
             parameters=[{"use_sim_time": use_sim_time, "autostart": True, "node_names": LIFECYCLE, "bond_timeout": 8.0}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="sim"),
        DeclareLaunchArgument("limits", default_value="default", description="default (0.6/0.3/0.8) | legged (0.4/0.3/0.5)"),
        DeclareLaunchArgument("controller", default_value="mppi", description="mppi | rpp"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("config_dir", default_value=""),
        OpaqueFunction(function=_setup),
    ])
