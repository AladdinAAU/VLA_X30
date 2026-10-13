from setuptools import find_packages, setup

package_name = "x30_common"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pyyaml"],
    zip_safe=True,
    maintainer="X30 VLN",
    maintainer_email="burgerbdd@gmail.com",
    description="Shared VLM client, image encoding, JSON repair and model config for the X30 VLN framework.",
    license="Apache-2.0",
)
