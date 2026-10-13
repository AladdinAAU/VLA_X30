from setuptools import find_packages, setup

package_name = "x30_eval"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="x30",
    maintainer_email="burgerbdd@gmail.com",
    description="Evaluation for the X30 VLN framework",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "run_episodes = x30_eval.run_episodes:main",
            "x30_eval_report = x30_eval.report:main",
            "x30_eval_sweep = x30_eval.sweep.__main__:main",
            "x30_eval_export = x30_eval.export.__main__:main",
            "x30_eval_mock_stack = x30_eval.testing.mock_stack:main",
        ],
    },
)
