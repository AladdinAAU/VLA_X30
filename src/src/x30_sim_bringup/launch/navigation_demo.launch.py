"""
Full navigation demo: spawns the X30 in the closed_arena world (sim.launch.py),
then brings up SLAM Toolbox + Nav2 (x30_navigation), and optionally the
paper's frontier exploration node so the robot autonomously maps and
explores the room. Point RViz's "2D Goal Pose" tool at the map to drive it
manually instead if you'd rather set goals by hand.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    explore_arg = DeclareLaunchArgument(
        'explore', default_value='false',
        description='Launch the paper\'s frontier exploration node for autonomous mapping')

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='Launch RViz2')

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run Gazebo without its GUI window (lower load)')

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('x30_sim_bringup'), 'launch', 'sim.launch.py')),
        launch_arguments={'rviz': LaunchConfiguration('rviz'),
                          'headless': LaunchConfiguration('headless')}.items(),
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('x30_navigation'), 'launch', 'slam.launch.py')),
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('x30_navigation'), 'launch', 'navigation.launch.py')),
    )

    frontier_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('x30_navigation'), 'launch', 'frontier_exploration.launch.py')),
        condition=IfCondition(LaunchConfiguration('explore')),
    )

    return LaunchDescription([
        explore_arg,
        rviz_arg,
        headless_arg,
        sim_launch,
        # Gazebo + the full ros2_control/controller stack are CPU-heavy to
        # bring up; starting SLAM Toolbox and the entire Nav2 stack (10+
        # more processes) at the exact same instant can starve Gazebo's
        # server thread long enough that its GUI watchdog gives up and
        # SIGKILLs it before the world ever loads. Give sim.launch.py a
        # head start so Gazebo, the robot spawn, and its controllers are
        # already up before piling on SLAM/Nav2.
        TimerAction(period=10.0, actions=[slam_launch]),
        TimerAction(period=10.0, actions=[navigation_launch]),
        TimerAction(period=10.0, actions=[frontier_launch]),
    ])
