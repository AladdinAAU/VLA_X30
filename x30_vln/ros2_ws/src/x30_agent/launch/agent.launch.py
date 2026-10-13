from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [DeclareLaunchArgument("agent_kind", default_value="llm"), DeclareLaunchArgument("run_name", default_value=""),
            DeclareLaunchArgument("use_sim_time", default_value="false"), DeclareLaunchArgument("results_dir", default_value="/x30/results")]
    common = {"agent_kind": LaunchConfiguration("agent_kind"), "run_name": LaunchConfiguration("run_name"),
              "use_sim_time": LaunchConfiguration("use_sim_time"), "results_dir": LaunchConfiguration("results_dir")}
    return LaunchDescription(args + [
        Node(package="x30_agent", executable="reasoner_agent", name="reasoner_agent", output="screen", parameters=[common]),
        Node(package="x30_agent", executable="instruction_follower", name="instruction_follower", output="screen", parameters=[common]),
    ])
