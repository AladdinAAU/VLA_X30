from setuptools import find_packages, setup

package_name = 'x30_teleop_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='burgerbdd',
    maintainer_email='burgerbdd@gmail.com',
    description='Twist -> unitree_guide_controller control_input bridge for the X30',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cmd_vel_bridge = x30_teleop_bridge.cmd_vel_bridge:main',
            'joystick = x30_teleop_bridge.joystick_gui:main',
        ],
    },
)
