# SPDX-License-Identifier: Apache-2.0
#
# The whole game in one command:
#
#   realsense --> [rectify] --> AprilTag --> tap_localizer --> tap_game
#                     └------> [YOLOv8/TensorRT] ------┘
#
# Wraps realsense_apriltag.launch.py (the perception graph) and adds the two
# application nodes, so playing takes one terminal instead of four.
#
# NOT included, deliberately: the micro-ROS agent. It runs as a persistent Docker
# container on the host with its own lifecycle -- the wand connects to it whenever
# it powers on, independent of whether the game is running -- so folding it into a
# ROS launch would tie two things together that should not be:
#
#   docker run -d --restart unless-stopped --net=host \
#     --name uros_agent microros/micro-ros-agent:humble udp4 --port 8888
#
# The measurement tools stay separate too, since they are for answering questions
# rather than playing: tap_localizer's own status line (`wand tag found N%`),
# tag_pose_stats, tap_detector, detection_probe. See docs/runbook.md.
#
# Usage:
#   ros2 launch tap_dance tap_game.launch.py tag_size:=0.1225 \
#     target_names:='["cup","pen"]' target_u:='[600.0, 1000.0]'
#
#   ros2 launch tap_dance tap_game.launch.py tag_size:=0.1225 use_yolo:=true
#
#   # keep the camera's startup chatter for debugging a bring-up problem
#   ros2 launch tap_dance tap_game.launch.py tag_size:=0.1225 quiet:=false

import ast
import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Passed straight through to realsense_apriltag.launch.py.
PERCEPTION_ARGS = ('sensor', 'width', 'height', 'fps', 'tag_size', 'auto_exposure',
                   'exposure', 'gain', 'use_yolo', 'engine_file_path',
                   'confidence_threshold', 'nms_threshold')


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    def as_list(name):
        # Launch arguments arrive as strings, so '["cup","pen"]' has to be parsed
        # before it can be handed to a node as a genuine list parameter.
        value = ast.literal_eval(arg(name))
        if not isinstance(value, (list, tuple)):
            raise ValueError(f'{name} must be a list, got {value!r}')
        return list(value)

    quiet = arg('quiet').lower() in ('true', '1', 'yes')
    use_yolo = arg('use_yolo').lower() in ('true', '1', 'yes')

    # The camera and NITROS log heavily at INFO, which buries the game's prompts.
    # Quiet the perception graph by default; the localizer and game stay at INFO
    # because their output IS the interface.
    passthrough = {k: arg(k) for k in PERCEPTION_ARGS}
    passthrough['log_level'] = 'WARN' if quiet else 'INFO'

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('tap_dance'), 'launch',
            'realsense_apriltag.launch.py')]),
        launch_arguments=passthrough.items(),
    )

    localizer = Node(
        package='tap_dance', executable='tap_localizer', name='tap_localizer',
        output='screen',
        parameters=[{'wand_tag_id': int(arg('wand_tag_id'))}],
    )

    game_params = {
        'use_yolo': use_yolo,
        'rounds': int(arg('rounds')),
        'time_limit': float(arg('time_limit')),
    }
    if use_yolo:
        game_params['yolo_classes'] = as_list('yolo_classes')
        game_params['min_yolo_hits'] = int(arg('min_yolo_hits'))
        # tap_game has to undo the encoder's resize itself, because the YOLOv8
        # decoder emits bboxes in network pixels and is never told the image size.
        game_params['image_width'] = int(arg('width'))
        game_params['image_height'] = int(arg('height'))
        game_params['network_width'] = 640
        game_params['network_height'] = 640
    else:
        game_params['target_names'] = [str(n) for n in as_list('target_names')]
        game_params['target_u'] = [float(u) for u in as_list('target_u')]

    game = Node(
        package='tap_dance', executable='tap_game', name='tap_game',
        output='screen', parameters=[game_params],
    )

    return [perception, localizer, game]


def generate_launch_description():
    args = [
        # --- perception passthrough (documented in realsense_apriltag.launch.py)
        DeclareLaunchArgument('sensor', default_value='color'),
        DeclareLaunchArgument('width', default_value='1280'),
        DeclareLaunchArgument('height', default_value='720'),
        DeclareLaunchArgument('fps', default_value='30'),
        DeclareLaunchArgument('tag_size', default_value='0.1225'),
        DeclareLaunchArgument('auto_exposure', default_value='false'),
        DeclareLaunchArgument('exposure', default_value='2000'),
        DeclareLaunchArgument('gain', default_value='128'),
        DeclareLaunchArgument('use_yolo', default_value='false'),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='/workspaces/isaac_ros-dev/models/yolov8s.plan'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.35'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),

        # --- application
        DeclareLaunchArgument(
            'wand_tag_id', default_value='-1',
            description='-1 accepts whichever tag is in view (right for one tag).'),
        DeclareLaunchArgument(
            'target_names', default_value='["cup","pen"]',
            description='Target names when use_yolo:=false.'),
        DeclareLaunchArgument(
            'target_u', default_value='[600.0, 1000.0]',
            description='Horizontal pixel position of each target. Measure with '
                        '`tap_localizer` and re-measure after changing `sensor`, '
                        'since the imagers are physically offset.'),
        DeclareLaunchArgument(
            'yolo_classes',
            default_value='["cup","bottle","book","mouse","keyboard","cell phone"]',
            description='COCO classes that count as targets when use_yolo:=true.'),
        DeclareLaunchArgument('min_yolo_hits', default_value='15'),
        DeclareLaunchArgument('rounds', default_value='10'),
        DeclareLaunchArgument('time_limit', default_value='6.0'),
        DeclareLaunchArgument(
            'quiet', default_value='true',
            description='Suppress camera/NITROS INFO chatter so the game prompts '
                        'are readable. Set false when debugging bring-up.'),
    ]
    return launch.LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
