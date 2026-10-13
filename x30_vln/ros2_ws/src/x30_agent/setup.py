from glob import glob

from setuptools import find_packages, setup

package_name = "x30_agent"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "x30_common"],
    zip_safe=True,
    maintainer="X30 VLN",
    maintainer_email="burgerbdd@gmail.com",
    description="Reasoner agent (tool calling) and instruction follower for the X30 VLN framework.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "reasoner_agent = x30_agent.ros.nodes:reasoner_main",
            "instruction_follower = x30_agent.ros.nodes:follower_main",
        ],
    },
)
