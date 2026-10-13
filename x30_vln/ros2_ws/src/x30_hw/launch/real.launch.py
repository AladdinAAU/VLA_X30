"""Real robot (or dry run against fake_x30) bring-up.

  ros2 launch x30_hw real.launch.py stage:=A              sensors only, nothing can move
  ros2 launch x30_hw real.launch.py stage:=B              teleop through /cmd_vel at <= 0.2 m/s (adapter starts DISARMED)
  ros2 launch x30_hw real.launch.py dry_run:=true         fake_x30 on loopback + nav_bridge + adapter; touches no real address

All numbers come from configs/robot_real.yaml (stage table, nav_bridge section). The ros1_bridge container is started
separately (docker compose --profile real).
"""

import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node

from x30_hw import config as cfgmod


def nav_bridge_params(cfg, stage, dry_run, motion_ip=None):
    nb = cfg["nav_bridge"]
    sl = cfgmod.stage_limits(cfg, stage)
    p = {
        "motion_host_ip": "127.0.0.1" if dry_run else (motion_ip or nb["motion_host_ip"]),
        "motion_host_port": 43893 if dry_run else nb["motion_host_port"],
        "local_recv_port": nb["local_recv_port"],
        "stand_target_gait": nb["stand_target_gait"],
        "cmd_vel_timeout_ms": nb["cmd_vel_timeout_ms"],
        "telemetry_stale_ms": nb["telemetry_stale_ms"],
        "heartbeat_interval_ms": nb["heartbeat_interval_ms"],
        "startup_acquire_control": bool(nb["startup_acquire_control"]),
        "enable_charge_interface": bool(nb["enable_charge_interface"]),
        "max_vx_forward": float(sl["vx_forward"]),
        "max_vx_backward": float(sl["vx_backward"]),
        "max_vy": float(sl["vy"]),
        "max_wz": float(sl["wz"]),
        "max_accel_x": nb["max_accel"], "max_accel_y": nb["max_accel"], "max_accel_yaw": nb["max_accel_yaw"],
        "max_decel_x": nb["max_accel"], "max_decel_y": nb["max_accel"], "max_decel_yaw": nb["max_accel_yaw"],
        "max_jerk_x": nb["max_jerk"], "max_jerk_y": nb["max_jerk"], "max_jerk_yaw": nb["max_jerk_yaw"],
        "odom_flip_vy": bool(nb["odom_flip_vy"]), "odom_flip_yaw": bool(nb["odom_flip_yaw"]),
        "battery_current_unit_a": nb["battery_current_unit_a"], "battery_current_sign": nb["battery_current_sign"],
    }
    for k in ("odom_pose_cov_xy", "odom_pose_cov_yaw", "odom_twist_cov_v", "odom_twist_cov_wz", "imu_orientation_cov",
              "imu_angvel_cov", "imu_linacc_cov"):
        p[k] = nb[k]
    return p


def launch_setup(context, *args, **kwargs):
    lc = lambda n: context.launch_configurations[n]  # noqa: E731
    cfg = cfgmod.load_robot_config(lc("config_file") or None)
    stage = lc("stage") or cfg["stage_default"]
    dry = lc("dry_run").lower() == "true"
    cfg_file = cfg["_path"]
    actions = []
    if dry:
        actions.append(ExecuteProcess(
            cmd=[sys.executable, "-m", "x30_hw.tools.fake_x30", "--time-scale", lc("fake_time_scale")], output="screen",
            name="fake_x30"))
    actions.append(Node(package="x30_nav_bridge", executable="nav_bridge_node", name="nav_bridge_node", output="screen",
                        parameters=[nav_bridge_params(cfg, stage, dry)], emulate_tty=True))
    actions.append(Node(package="x30_hw", executable="static_tf_publisher", name="static_tf_publisher", output="screen",
                        parameters=[{"config_file": cfg_file, "overlay_file": lc("tf_overlay")}]))
    actions.append(Node(package="x30_hw", executable="real_adapter", name="real_adapter", output="screen",
                        parameters=[{"config_file": cfg_file, "stage": stage,
                                     "camera_expected": lc("camera").lower() == "true",
                                     "imu_source": lc("imu_source") or ("udp" if dry else "")}]))
    if lc("camera").lower() == "true":
        actions.append(Node(package="x30_hw", executable="camera_ingest", name="camera_ingest", output="screen",
                            parameters=[{"config_file": cfg_file, "url": lc("camera_url")}]))
    if lc("thermal").lower() == "true":
        actions.append(Node(package="x30_hw", executable="thermal_ingest", name="thermal_ingest", output="screen",
                            parameters=[{"config_file": cfg_file}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("stage", default_value="", description="A|B|C|D (default: stage_default in robot_real.yaml)"),
        DeclareLaunchArgument("dry_run", default_value="false", description="true: fake_x30 on loopback, no real address"),
        DeclareLaunchArgument("config_file", default_value=""),
        DeclareLaunchArgument("camera", default_value="true"),
        DeclareLaunchArgument("camera_url", default_value="", description="override camera.url (e.g. synthetic://640x480@10)"),
        DeclareLaunchArgument("thermal", default_value="false"),
        DeclareLaunchArgument("imu_source", default_value="", description="ros1|udp (default: robot_real.yaml; udp in dry runs)"),
        DeclareLaunchArgument("tf_overlay", default_value="", description="calibration overlay YAML for static TF"),
        DeclareLaunchArgument("fake_time_scale", default_value="1.0"),
        OpaqueFunction(function=launch_setup),
    ])
