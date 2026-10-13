"""Replay profile: play the recorded real lidar bag, publish static TF, add clearly marked synthetic stubs.

  ros2 launch x30_hw replay.launch.py                          # clock:=true loop:=true, bag datasets/real/lab_2026-10-06
  ros2 launch x30_hw replay.launch.py clock:=false loop:=false rate:=0.5

Consumers must run with use_sim_time:=true when clock:=true. The stub node (/odom zero motion, /imu/data gravity only,
/x30/ready, /x30/battery, /x30/health) is SYNTHETIC; /x30/replay_stub is true while it runs. Note: with --loop and --clock
the time jumps back at every loop; TF buffers and odometry nodes should be reset or the replay run once (loop:=false).
"""

import os
import subprocess
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node

from x30_hw import config as cfgmod


def launch_setup(context, *args, **kwargs):
    lc = lambda n: context.launch_configurations[n]  # noqa: E731
    cfg = cfgmod.load_robot_config(lc("config_file") or None)
    bag = lc("bag") or os.path.join(os.environ.get("X30_ROOT", "/x30"), "datasets", "real", "lab_2026-10-06")
    compat = os.path.join(os.environ.get("X30_ROOT", "/x30"), "docker", "scripts", "bag_compat.py")
    if os.path.isfile(compat):  # the converted lab bag has metadata version 9, which Humble's rosbag2 rejects
        bag = subprocess.check_output([sys.executable, compat, bag], text=True).strip()
    clock = lc("clock").lower() == "true"
    cmd = ["ros2", "bag", "play", bag, "--rate", lc("rate")]
    if clock:
        cmd += ["--clock", lc("clock_hz")]  # default 40 Hz would throttle the 100 Hz sim-time stub timers
    if lc("loop").lower() == "true":
        cmd.append("--loop")
    sim = {"use_sim_time": clock}
    actions = [
        ExecuteProcess(cmd=cmd, output="screen", name="bag_play"),
        Node(package="x30_hw", executable="static_tf_publisher", name="static_tf_publisher", output="screen",
             parameters=[{"config_file": cfg["_path"], "overlay_file": lc("tf_overlay")}, sim]),
    ]
    if lc("stub").lower() == "true":
        actions.append(Node(package="x30_hw", executable="replay_stub", name="replay_stub", output="screen", parameters=[sim]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("bag", default_value="", description="rosbag2 directory (default datasets/real/lab_2026-10-06)"),
        DeclareLaunchArgument("clock", default_value="true", description="publish /clock from the bag stamps"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("clock_hz", default_value="200", description="rate of /clock"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("stub", default_value="true", description="synthetic /odom, /imu/data, /x30/* stand-ins"),
        DeclareLaunchArgument("config_file", default_value=""),
        DeclareLaunchArgument("tf_overlay", default_value=""),
        OpaqueFunction(function=launch_setup),
    ])
