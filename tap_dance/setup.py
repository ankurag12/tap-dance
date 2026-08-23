import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'tap_dance'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch/ into the package's share dir so it can be found at
        # runtime via get_package_share_directory('tap_dance').
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hunter',
    maintainer_email='ankur.ag12@gmail.com',
    description='Tap-target game on real desk objects: an AprilTag-and-IMU '
                'wand supplies the contact event, a RealSense D456 supplies '
                'where it happened.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_probe = tap_dance.debug.detection_probe:main',
            'tag_pose_stats = tap_dance.debug.tag_pose_stats:main',
            'tap_detector = tap_dance.debug.tap_detector:main',
            'tap_localizer = tap_dance.tap_localizer:main',
            'tap_game = tap_dance.tap_game:main',
            'game_overlay = tap_dance.game_overlay:main',
            'hover_probe = tap_dance.debug.hover_probe:main',
        ],
    },
)
