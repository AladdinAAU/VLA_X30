"""
Spawns the DEEP Robotics X30 (upstream description from the
quadruped_ros2_control submodule, extended with a lidar + RGBD camera by
x30_sensors) inside the closed_arena world from x30_sim_worlds, brings up
ros2_control, and starts the cmd_vel -> control_input bridge so the robot
can be driven by Nav2 or teleop.

Mirrors the spawn/controller-spawner sequencing used upstream in
unitree_guide_controller/launch/gazebo.launch.py (legubiao/quadruped_ros2_control),
adapted to load our own world instead of empty.sdf and to add the sensor
bridge + walking bring-up.
"""
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription,
                            RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


NVIDIA_EGL_VENDOR = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'


def nvidia_render_env(mode):
    """Env vars that make Gazebo's ogre2 sensor rendering use the NVIDIA GPU.

    Without them Gazebo renders the lidar/RGBD camera on the integrated GPU / Mesa
    software path, which pins a CPU core and drops the sim to a fluctuating
    real-time factor (0.005-0.8). On the NVIDIA GPU it holds ~1.0, which the
    walking controller needs to stay balanced.
    """
    if mode == 'false' or (mode == 'auto' and not (
            os.path.exists(NVIDIA_EGL_VENDOR) and os.path.exists('/proc/driver/nvidia'))):
        return []
    return [
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES', NVIDIA_EGL_VENDOR),
    ]


def launch_setup(context, *args, **kwargs):
    init_height = context.launch_configurations['height']
    use_rviz = context.launch_configurations['rviz']
    headless = context.launch_configurations['headless'] == 'true'

    sensors_pkg = get_package_share_directory('x30_sensors')
    xacro_file = os.path.join(sensors_pkg, 'xacro', 'x30_sensors.xacro')
    robot_description = xacro.process_file(xacro_file, mappings={'GAZEBO': 'true', 'CLASSIC': 'false'}).toxml()

    world_path = os.path.join(
        get_package_share_directory('x30_sim_worlds'), 'worlds', 'closed_arena.sdf')

    bridge_config = os.path.join(
        get_package_share_directory('x30_sim_bringup'), 'config', 'ros_gz_bridge.yaml')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'publish_frequency': 50.0,
            'use_tf_static': True,
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name', 'x30',
                   '-allow_renaming', 'true', '-z', init_height],
    )

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{'config_file': bridge_config, 'use_sim_time': True}],
    )

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )

    imu_sensor_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['imu_sensor_broadcaster', '--controller-manager', '/controller_manager'],
    )

    leg_pd_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['leg_pd_controller', '--controller-manager', '/controller_manager'],
    )

    unitree_guide_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['unitree_guide_controller', '--controller-manager', '/controller_manager'],
    )

    cmd_vel_bridge = Node(
        package='x30_teleop_bridge',
        executable='cmd_vel_bridge',
        name='cmd_vel_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    rviz_config = os.path.join(
        get_package_share_directory('x30_sim_bringup'), 'rviz', 'x30_sim.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    actions = nvidia_render_env(context.launch_configurations['nvidia']) + [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])]),
            launch_arguments=[('gz_args', [' -r -s -v 3 ' if headless else ' -r -v 3 ', world_path])]),
        ros_gz_bridge,
        robot_state_publisher,
        gz_spawn_entity,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[imu_sensor_broadcaster, joint_state_broadcaster, cmd_vel_bridge],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster,
                on_exit=[leg_pd_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=leg_pd_controller,
                on_exit=[unitree_guide_controller],
            )
        ),
    ]

    if use_rviz == 'true':
        actions.append(rviz)

    return actions


def generate_launch_description():
    height_arg = DeclareLaunchArgument(
        'height', default_value='0.64',
        description='Initial spawn height (m) for the X30 above the ground plane')

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 alongside the simulation')

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run the Gazebo server only (no Gazebo GUI window): saves CPU/GPU for real-time speed')

    nvidia_arg = DeclareLaunchArgument(
        'nvidia', default_value='auto', choices=['auto', 'true', 'false'],
        description='Render Gazebo sensors on the NVIDIA GPU (auto: when its driver is present)')

    return LaunchDescription([
        height_arg,
        nvidia_arg,
        headless_arg,
        rviz_arg,
        OpaqueFunction(function=launch_setup),
    ])
