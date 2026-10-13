from glob import glob

from setuptools import find_packages, setup

package_name = 'x30_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),  # x30_perception and perception_server
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='x30',
    maintainer_email='burgerbdd@gmail.com',
    description='Perception client/server for the X30 VLN framework',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'perception_client = x30_perception.perception_client_node:main',
            'synthetic_stream = x30_perception.synthetic_stream:main',
        ],
    },
)
