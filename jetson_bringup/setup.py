import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'jetson_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch/ and config/ into the package's share dir so they can
        # be found at runtime via get_package_share_directory('jetson_bringup').
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hunter',
    maintainer_email='ankur.ag12@gmail.com',
    description='Launch files and configs that bring up Isaac ROS experiments '
                'on the Jetson Orin Nano (RealSense D456, AprilTag, Visual SLAM).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_monitor = jetson_bringup.sensor_monitor:main',
            'detection_probe = jetson_bringup.detection_probe:main',
            'tag_pose_stats = jetson_bringup.tag_pose_stats:main',
            'object_locator = jetson_bringup.object_locator:main',
            'wand_pointer = jetson_bringup.wand_pointer:main',
            'tap_detector = jetson_bringup.tap_detector:main',
            'tap_localizer = jetson_bringup.tap_localizer:main',
            'tap_game = jetson_bringup.tap_game:main',
        ],
    },
)
