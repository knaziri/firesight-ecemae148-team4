from setuptools import setup, find_packages
from glob import glob

package_name = 'robot_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    install_requires=['setuptools'],
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py')),
    ],
    entry_points={
        'console_scripts': [
            'test_node = robot_pkg.test_node:main',
            'test_vesc_node = robot_pkg.test_vesc_node:main',
            'nav_node       = robot_pkg.nav_node:main',
            'test_rpm_steering_node  = robot_pkg.test_rpm_steering_node:main',
        ],
    },
)
