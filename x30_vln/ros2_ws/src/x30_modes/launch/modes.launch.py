"""Launch the behaviour layer: cmd_vel_mux, safety_guard, mode_manager, skill_server.

  ros2 launch x30_modes modes.launch.py profile:=sim            # configs/modes/sim.yaml
  ros2 launch x30_modes modes.launch.py profile:=real use_sim_time:=false
  ros2 launch x30_modes modes.launch.py nodes:=guard,mux          # subset: mux, guard, manager, skills

The configs directory comes from $X30_CONFIG_DIR / $X30_CONFIGS or is found above the package source.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from x30_modes.paths import find_configs_dir

EXECUTABLES = {"mux": "cmd_vel_mux", "guard": "safety_guard", "manager": "mode_manager", "skills": "skill_server"}


def _launch(context, *args, **kwargs):
    profile = LaunchConfiguration("profile").perform(context)
    configs = LaunchConfiguration("configs_dir").perform(context)
    params_file = LaunchConfiguration("params_file").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() in ("1", "true", "yes")
    wanted = [n.strip() for n in LaunchConfiguration("nodes").perform(context).split(",") if n.strip()]
    if not params_file:
        base = configs or str(find_configs_dir() or "")
        if not base:
            raise RuntimeError("configs directory not found: set X30_CONFIG_DIR or pass configs_dir:=...")
        params_file = f"{base}/modes/{profile}.yaml"
    out = []
    for key in wanted:
        exe = EXECUTABLES[key]
        out.append(Node(package="x30_modes", executable=exe, name=exe, output="screen",
                        parameters=[params_file, {"use_sim_time": use_sim_time}]))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("profile", default_value="sim", description="sim | real"),
        DeclareLaunchArgument("configs_dir", default_value="", description="override the configs directory"),
        DeclareLaunchArgument("params_file", default_value="", description="override the whole parameter file"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("nodes", default_value="mux,guard,manager,skills"),
        OpaqueFunction(function=_launch),
    ])
