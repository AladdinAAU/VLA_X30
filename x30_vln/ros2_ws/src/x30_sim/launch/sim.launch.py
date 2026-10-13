"""X30 simulator.

  ros2 launch x30_sim sim.launch.py world:=default_arena locomotion:=kinematic gui:=false render:=sw spawn_x:=0 spawn_y:=0 spawn_yaw:=0

world        id in datasets/registry.yaml, or the built-in `default_arena`
locomotion   legged (unitree_guide_controller, needs real-time factor ~1) | kinematic (VelocityControl, fast, flat floors)
render       sw (Mesa llvmpipe, default, stable memory) | gpu (WSL2 D3D12; leaks memory with cameras, see docs/reports/sim.md)
thermal      true|false: optional thermal camera payload (not stock on the X30)
odom_noise   0 = ground-truth /odom, 1 = nominal leg-odometry error (configs/robot_sim.yaml odometry.nominal)
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from x30_sim import bridge_config, robot_description, worlds
from x30_sim import config as cfgmod

LEGGED_SPAWN_Z = 0.64        # prework value: the legs hang straight at spawn and the controller stands the robot up
MODEL = 'x30'


def _b(context, name):
    return context.launch_configurations[name].lower() in ('true', '1', 'yes')


def launch_setup(context, *args, **kwargs):
    lc = context.launch_configurations
    locomotion = lc['locomotion']
    thermal, gui = _b(context, 'thermal'), _b(context, 'gui')
    render = lc['render']
    if locomotion not in ('legged', 'kinematic'):
        raise RuntimeError(f'locomotion must be legged or kinematic, got {locomotion}')

    robot_config = cfgmod.find_robot_config(lc['robot_config'] or None)
    cfg = cfgmod.load_robot_config(robot_config)
    cfg['camera']['depth'] = cfg['camera']['depth'] and _b(context, 'depth')
    if not cfg['thermal']['enabled']:
        thermal = False

    share = Path(get_package_share_directory('x30_sim'))
    work = Path(lc['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    urdf, stand_h = robot_description.build(share / 'urdf' / 'x30_sim.urdf.xacro', robot_config, locomotion, thermal, cfg,
                                            mesh_cache=Path(lc['mesh_cache']))
    urdf_file = work / f'x30_{locomotion}.urdf'
    urdf_file.write_text(urdf)

    spec = worlds.resolve_world(lc['world'], lc['registry'] or None)
    dt = 0.001 if locomotion == 'legged' else 0.005
    world_file, world_name = worlds.prepare_world(spec, work, dt, float(lc['rtf']), cfg['thermal']['ambient_c'] + 273.15)

    sx = float(lc['spawn_x']) if lc['spawn_x'] != '' else spec.spawn['x']
    sy = float(lc['spawn_y']) if lc['spawn_y'] != '' else spec.spawn['y']
    syaw = float(lc['spawn_yaw']) if lc['spawn_yaw'] != '' else spec.spawn['yaw']
    clearance = cfg['robot']['spawn_clearance_m']
    z_stand = stand_h + (clearance if locomotion == 'kinematic' else 0.02)
    if lc['spawn_z'] != '':
        spawn_z = float(lc['spawn_z'])
    else:
        spawn_z = max(spec.spawn.get('z', 0.0), 0.0) + (z_stand if locomotion == 'kinematic' else LEGGED_SPAWN_Z)

    bridge_file = work / 'bridge.yaml'
    bridge_file.write_text(bridge_config.bridge_yaml(cfg, locomotion, thermal, MODEL))

    sim_time = {'use_sim_time': True}
    common = {'robot_config': str(robot_config), **sim_time}

    env = [SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', worlds.resource_path_for(spec))]
    if render == 'gpu':
        env += [UnsetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE'), SetEnvironmentVariable('GALLIUM_DRIVER', 'd3d12'),
                SetEnvironmentVariable('LD_LIBRARY_PATH', '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')),
                SetEnvironmentVariable('MESA_D3D12_DEFAULT_ADAPTER_NAME', 'NVIDIA')]
    else:
        env += [SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'), SetEnvironmentVariable('GALLIUM_DRIVER', 'llvmpipe')]

    # software rendering uses EGL surfaceless (--headless-rendering); the D3D12 path needs the WSLg X display (GLX), EGL surfaceless cannot init it
    server_cmd = ['gz', 'sim', '-s', '-r', '-v', '2'] + (['--headless-rendering'] if render != 'gpu' else []) + [str(world_file)]
    actions = env + [ExecuteProcess(cmd=server_cmd, name='gz_server', output='screen')]
    if gui:
        actions.append(ExecuteProcess(cmd=['gz', 'sim', '-g'], name='gz_gui', output='screen'))

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher', name='robot_state_publisher',
               parameters=[{'robot_description': urdf, 'publish_frequency': 50.0, 'use_tf_static': True, **sim_time}])
    base_tf = Node(package='tf2_ros', executable='static_transform_publisher', name='base_link_to_base',
                   arguments=['--frame-id', cfg['robot']['base_frame'], '--child-frame-id', 'base'], parameters=[sim_time])
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='ros_gz_bridge', output='screen',
                  parameters=[{'config_file': str(bridge_file), **sim_time}])
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-file', str(urdf_file), '-name', MODEL, '-x', str(sx), '-y', str(sy), '-z', str(spawn_z), '-Y', str(syaw)])

    nodes = [
        Node(package='x30_sim', executable='lidar_merge', name='lidar_merge', output='screen',
             parameters=[{**common, 'seed': int(lc['seed'])}]),
        Node(package='x30_sim', executable='sim_state', name='sim_state', output='screen',
             parameters=[{**common, 'locomotion': locomotion, 'odom_noise': float(lc['odom_noise']), 'seed': int(lc['seed']),
                          'thermal': thermal}]),
        Node(package='x30_sim', executable='ground_truth', name='ground_truth', output='screen',
             parameters=[{**common, 'world': lc['world'], 'registry': lc['registry']}]),
        Node(package='x30_sim', executable='sim_services', name='sim_services', output='screen',
             parameters=[{**common, 'world_name': world_name, 'locomotion': locomotion, 'model_name': MODEL, 'z_stand': z_stand,
                          'spawn_x': sx, 'spawn_y': sy, 'spawn_yaw': syaw}]),
    ]
    if thermal:
        nodes.append(Node(package='x30_sim', executable='thermal_processor', name='thermal_processor', output='screen',
                          parameters=[common]))

    if locomotion == 'kinematic':
        nodes.append(Node(package='x30_sim', executable='kinematic_driver', name='kinematic_driver', output='screen',
                          parameters=[{**common, 'latency_s': float(lc['latency'])}]))
        after_spawn = []
    else:
        def spawner(name):
            return Node(package='controller_manager', executable='spawner', arguments=[name, '--controller-manager', '/controller_manager'],
                        parameters=[sim_time])
        imu_b, jsb, pd, guide = (spawner(n) for n in ('imu_sensor_broadcaster', 'joint_state_broadcaster', 'leg_pd_controller',
                                                      'unitree_guide_controller'))
        nodes.append(Node(package='x30_sim', executable='legged_bridge', name='legged_bridge', output='screen', parameters=[common]))
        after_spawn = [imu_b, jsb,
                       RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[pd])),
                       RegisterEventHandler(OnProcessExit(target_action=pd, on_exit=[guide]))]

    actions += [rsp, base_tf, bridge, *nodes,
                TimerAction(period=float(lc['spawn_delay']), actions=[spawn])]
    if after_spawn:
        actions.append(RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=after_spawn)))
    return actions


def generate_launch_description():
    args = [
        ('world', 'default_arena', 'world id (datasets/registry.yaml) or the built-in default_arena'),
        ('locomotion', 'kinematic', 'legged | kinematic'),
        ('gui', 'false', 'show the Gazebo GUI (needs a display; slow with software rendering)'),
        ('render', os.environ.get('RENDER', 'sw'), 'sw (llvmpipe) | gpu'),
        ('thermal', 'true', 'simulate the optional thermal camera'),
        ('depth', 'true', 'simulate the sim-only depth image (rgbd camera); false is cheaper'),
        ('spawn_x', '', 'start x (default: registry spawn)'),
        ('spawn_y', '', 'start y'),
        ('spawn_z', '', 'start z (default: standing height / controller drop height)'),
        ('spawn_yaw', '', 'start yaw (rad)'),
        ('odom_noise', '0.0', '0 = ground-truth odometry, 1 = nominal leg-odometry error'),
        ('latency', '-1.0', 'kinematic backend command latency in s (negative: config value)'),
        ('rtf', '1.0', 'target real-time factor written into the world physics'),
        ('seed', '1', 'seed for sensor/odometry noise'),
        ('robot_config', '', 'path of robot_sim.yaml (default: $X30_CONFIGS/robot_sim.yaml)'),
        ('registry', '', 'path of datasets/registry.yaml'),
        ('work_dir', '/tmp/x30_sim', 'where the generated URDF, world and bridge files are written'),
        ('mesh_cache', '/tmp/x30_sim_meshes', 'cache of the decimated visual meshes'),
        ('spawn_delay', '6.0', 'seconds to wait for the Gazebo server before spawning'),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args]
                             + [OpaqueFunction(function=launch_setup)])
