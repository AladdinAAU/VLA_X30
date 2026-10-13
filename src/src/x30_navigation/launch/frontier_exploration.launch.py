"""
Runs the frontier exploration node from the quadruped_navigation_ros2
submodule's autonomous_navigation package (the paper's greedy
wavefront/Time-to-Closest-Viable-Frontier explorer). The node is written
generically against /map, /odometry, /cmd_vel and the navigate_to_pose
action, so it needs no robot-specific changes - only a topic remap since our
odometry is published on /odom rather than /odometry.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    frontier_explorer = Node(
        package='autonomous_navigation',
        executable='frontierExploration',
        name='explorer',
        output='screen',
        remappings=[('/odometry', '/odom')],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        frontier_explorer,
    ])
