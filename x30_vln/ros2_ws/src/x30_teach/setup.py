from glob import glob

from setuptools import find_packages, setup

package_name = "x30_teach"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="X30 VLN",
    maintainer_email="burgerbdd@gmail.com",
    description="Learner phase: spoken/typed place names -> LLM -> named rooms and spots in the semantic map.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["teach_node = x30_teach.teach_node:main"]},
)
