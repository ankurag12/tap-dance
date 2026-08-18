# SPDX-License-Identifier: Apache-2.0
#
# Live AprilTag detection on the Jetson Orin Nano with a RealSense D456:
#
#     realsense2_camera --> RectifyNode --> AprilTagNode --> /tag_detections (+ TF)
#
# All three run as composable nodes inside one container process, so Isaac ROS's
# NITROS transport can pass images between them on the GPU with zero copies.
#
# How this differs from isaac_ros_apriltag's stock realsense example:
#   * Resolution is a launch argument, defaulting to 720p. The stock example
#     hardcodes 1080p, whose rectify GPU pool OOMs the Orin Nano's 8 GB unified
#     memory; 720p fits comfortably.
#   * The AprilTag physical `size` is a launch argument (default 0.1225 m, our
#     printed tag) so the reported pose is correctly scaled.
#   * The graph is wired properly: apriltag consumes the *rectified* image
#     (matching isaac_ros_apriltag_core.launch.py). The stock realsense example
#     fed apriltag the raw image and left rectify orphaned.
#
# Usage:
#   ros2 launch tap_dance realsense_apriltag.launch.py
#   ros2 launch tap_dance realsense_apriltag.launch.py tag_size:=0.06
#   ros2 launch tap_dance realsense_apriltag.launch.py width:=848 height:=480

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # --- Launch arguments: resolved at launch time, overridable on the CLI ---
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    tag_size = LaunchConfiguration('tag_size')
    auto_exposure = LaunchConfiguration('auto_exposure')
    exposure = LaunchConfiguration('exposure')
    gain = LaunchConfiguration('gain')

    args = [
        DeclareLaunchArgument(
            'width', default_value='1280',
            description='Color image width (px). Lower it if the GPU pool OOMs.'),
        DeclareLaunchArgument(
            'height', default_value='720',
            description='Color image height (px).'),
        DeclareLaunchArgument(
            'tag_size', default_value='0.1225',
            description='AprilTag black-square edge length in meters. Must match '
                        'the tag in view or the reported pose distance is wrong.'),
        # Exposure controls MOTION BLUR, which is what limits tag detection on a
        # moving wand. Auto-exposure optimises for brightness and will happily
        # pick 20-33 ms indoors; at a hand speed of ~500 px/s that smears the tag
        # by 10-16 px, and a tag only ~40 px wide loses the corner sharpness
        # AprilTag needs to decode. Measured: 97-99% detection when stationary,
        # 38-53% in motion.
        #
        # Freezing motion means a SHORT exposure, and the cost is a darker image
        # -- hence more gain, which adds noise. That noise/blur trade is the real
        # tuning axis here, so both are exposed.
        DeclareLaunchArgument(
            'auto_exposure', default_value='true',
            description='Color auto-exposure. Set false to pin exposure/gain '
                        'and stop AE choosing a motion-blurring exposure.'),
        DeclareLaunchArgument(
            'exposure', default_value='80',
            description='Color exposure when auto_exposure:=false, in units of '
                        '100 us (80 = 8 ms). Lower freezes motion; compensate '
                        'with gain.'),
        DeclareLaunchArgument(
            'gain', default_value='128',
            description='Color sensor gain when auto_exposure:=false. Raise to '
                        'offset a short exposure, at the cost of noise.'),
    ]

    # LaunchConfiguration values arrive as strings; wrap them with an explicit
    # type so the nodes receive genuine int/float parameters.
    width_i = ParameterValue(width, value_type=int)
    height_i = ParameterValue(height, value_type=int)

    # --- 1) Camera: color only (AprilTag needs neither depth nor IR) ---
    realsense_node = ComposableNode(
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        name='realsense2_camera',
        namespace='',
        parameters=[{
            'color_width': width_i,
            'color_height': height_i,
            'enable_depth': False,
            'enable_infra1': False,
            'enable_infra2': False,
            'rgb_camera.enable_auto_exposure': ParameterValue(
                auto_exposure, value_type=bool),
            'rgb_camera.exposure': ParameterValue(exposure, value_type=int),
            'rgb_camera.gain': ParameterValue(gain, value_type=int),
        }],
        # Publish the color stream on the topics RectifyNode subscribes to.
        remappings=[
            ('color/image_raw', 'image_raw'),
            ('color/camera_info', 'camera_info'),
        ],
    )

    # --- 2) Rectify: undistort using K + distortion coefficients ---
    #     subscribes: image_raw, camera_info ; publishes: image_rect, camera_info_rect
    rectify_node = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        name='rectify',
        namespace='',
        parameters=[{
            'output_width': width_i,
            'output_height': height_i,
        }],
    )

    # --- 3) AprilTag: GPU detector, running on the rectified image ---
    apriltag_node = ComposableNode(
        package='isaac_ros_apriltag',
        plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
        name='apriltag',
        namespace='',
        parameters=[{
            'size': ParameterValue(tag_size, value_type=float),
            'max_tags': 64,
            'tile_size': 4,
            'tag_family': 'tag36h11',
            'backends': 'CUDA',
        }],
        # Consume the rectified stream (matches isaac_ros_apriltag_core.launch.py).
        remappings=[
            ('image', 'image_rect'),
            ('camera_info', 'camera_info_rect'),
        ],
    )

    container = ComposableNodeContainer(
        package='rclcpp_components',
        name='apriltag_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=[
            realsense_node,
            rectify_node,
            apriltag_node,
        ],
        output='screen',
    )

    return launch.LaunchDescription(args + [container])
