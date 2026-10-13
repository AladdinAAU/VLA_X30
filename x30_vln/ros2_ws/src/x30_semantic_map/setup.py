from glob import glob

from setuptools import find_packages, setup

package_name = 'x30_semantic_map'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['prompts/*.txt']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='x30',
    maintainer_email='burgerbdd@gmail.com',
    description='Persistent 3D semantic object map, VLM annotator and annotated top-down map for the X30 VLN framework',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'semantic_mapper = x30_semantic_map.semantic_mapper_node:main',
        ],
    },
)
