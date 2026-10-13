from glob import glob

from setuptools import find_packages, setup

package_name = "x30_follow"

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
    description="follow_person: person tracker and velocity controller for the X30 VLN framework.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["follow_person = x30_follow.follow_person_node:main"]},
)
