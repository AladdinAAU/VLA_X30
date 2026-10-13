from glob import glob

from setuptools import find_packages, setup

package_name = "x30_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/behavior_trees", glob("behavior_trees/*.xml")),
    ],
    install_requires=["setuptools", "numpy", "scipy", "pyyaml"],
    zip_safe=True,
    maintainer="X30 VLN",
    maintainer_email="burgerbdd@gmail.com",
    description="Lidar odometry, odometry selection, 2D mapping, Nav2 and frontier exploration for the X30.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "lidar_odometry = x30_navigation.lidar_odometry:main",
            "odom_selector = x30_navigation.odom_selector:main",
            "cloud_leveler = x30_navigation.cloud_leveler:main",
            "cloud_to_scan = x30_navigation.cloud_to_scan:main",
            "obstacle_cloud = x30_navigation.obstacle_cloud:main",
            "occupancy_mapper = x30_navigation.occupancy_mapper:main",
            "frontier_explorer = x30_navigation.frontier_explorer:main",
            "direct_navigator = x30_navigation.direct_navigator:main",
            "waypoint_runner = x30_navigation.waypoint_runner:main",
            "nav_stub_robot = x30_navigation.tools.stub_robot:main",
            "replay_report = x30_navigation.tools.replay_report:main",
            "waypoint_bench = x30_navigation.tools.waypoint_bench:main",
            "goal_benchmark = x30_navigation.tools.goal_benchmark:main",
            "ate_eval = x30_navigation.tools.ate_eval:main",
            "map_eval = x30_navigation.tools.map_eval:main",
            "cmd_relay = x30_navigation.tools.cmd_relay:main",
            "known_map_nav = x30_navigation.known_map_nav:main",
            "next_viewpoint = x30_navigation.next_viewpoint:main",
            "known_map_bench = x30_navigation.tools.known_map_bench:main",
        ],
    },
)
