from glob import glob

from setuptools import find_packages, setup

package_name = 'x30_hw'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='X30 VLN',
    maintainer_email='burgerbdd@gmail.com',
    description='X30 real robot integration',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'real_adapter = x30_hw.real_adapter:main',
            'static_tf_publisher = x30_hw.static_tf_publisher:main',
            'camera_ingest = x30_hw.camera_ingest:main',
            'thermal_ingest = x30_hw.thermal_ingest:main',
            'replay_stub = x30_hw.replay_stub:main',
            'fake_x30 = x30_hw.tools.fake_x30:main',
            'x30ctl = x30_hw.x30ctl:main',
            'preflight = x30_hw.tools.preflight:main',
            'measure_step_response = x30_hw.tools.measure_step_response:main',
            'measure_odom_drift = x30_hw.tools.measure_odom_drift:main',
            'clock_offset = x30_hw.tools.clock_offset:main',
            'calibrate_camera = x30_hw.calib.calibrate_camera:main',
            'calibrate_lidar_camera = x30_hw.calib.lidar_camera_tool:main',
            'record_telemetry = x30_hw.tools.record_telemetry:main',
        ],
    },
)
