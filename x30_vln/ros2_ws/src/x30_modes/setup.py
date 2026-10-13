from glob import glob

from setuptools import find_packages, setup

package_name = "x30_modes"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools", "numpy", "pyyaml"],
    zip_safe=True,
    maintainer="X30 VLN",
    maintainer_email="burgerbdd@gmail.com",
    description="Mode manager, cmd_vel mux, safety guard and skill server for the X30 VLN framework.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cmd_vel_mux = x30_modes.cmd_vel_mux:main",
            "safety_guard = x30_modes.safety_guard:main",
            "mode_manager = x30_modes.mode_manager:main",
            "skill_server = x30_modes.skill_server:main",
        ],
    },
)
