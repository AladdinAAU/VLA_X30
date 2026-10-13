from glob import glob

from setuptools import find_packages, setup

package_name = 'x30_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='x30',
    maintainer_email='burgerbdd@gmail.com',
    description='X30 Gazebo simulation behind the X30 VLN interface contract',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_merge = x30_sim.lidar_merge:main',
            'sim_state = x30_sim.sim_state:main',
            'kinematic_driver = x30_sim.kinematic_driver:main',
            'legged_bridge = x30_sim.legged_bridge:main',
            'thermal_processor = x30_sim.thermal_processor:main',
            'ground_truth = x30_sim.ground_truth:main',
            'sim_services = x30_sim.sim_services:main',
            'person_mover = x30_sim.person_mover:main',
        ],
    },
)
