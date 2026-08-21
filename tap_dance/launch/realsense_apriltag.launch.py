# SPDX-License-Identifier: Apache-2.0
#
# Live AprilTag detection on the Jetson Orin Nano with a RealSense D456, on
# EITHER the colour or the infrared imager:
#
#   sensor:=color   realsense --> RectifyNode --> AprilTagNode --> /tag_detections
#   sensor:=infra1  realsense ------------------> AprilTagNode --> /tag_detections
#
# Nodes run as composable nodes in one container process, so Isaac ROS's NITROS
# transport passes images between them on the GPU with zero copies.
#
# WHY THE IR OPTION EXISTS
# The D456's colour imager is ROLLING SHUTTER; its left/right IR imagers are
# GLOBAL SHUTTER. A rolling shutter reads rows sequentially, so a moving tag is
# not merely blurred but SHEARED -- a square projects to a parallelogram, and
# AprilTag's quad detection rejects it or decodes it wrongly. A shorter exposure
# fixes blur but does nothing about shear, which matches what we measured: 100%
# detection stationary, ~50% in motion, and an exposure sweep that changed little.
#
# Three further advantages of the IR path:
#   * The D456's IR is FACTORY RECTIFIED, so RectifyNode is skipped entirely --
#     one less GPU node, less latency, less of the 8 GB unified memory.
#   * IR is monochrome, which is what AprilTag wants anyway; the node accepts
#     mono8 directly (mapped to VPI_IMAGE_FORMAT_U8).
#   * IR held a steady 30 FPS over USB where the colour stream measured ~20 and
#     jittery, and offers 848x480 at 60/90 FPS for less inter-frame motion.
#
# The IR trade-offs, both real:
#   * The projector MUST be off (emitter_enabled 0) or its dot pattern lands on
#     the tag. With it off the imagers rely on ambient IR -- adequate in this
#     room, but LED-only lighting emits little.
#   * Lower resolutions shrink the tag: a 12 cm tag spans ~40 px at 1280x720 but
#     only ~27 px at 848x480, near the decode floor. Start at 720p so the shutter
#     is the only variable changed.
#
# EXPOSURE UNITS DIFFER BY SENSOR -- a genuine librealsense trap:
#   colour  rgb_camera.exposure    in units of 100 us  (80 = 8 ms)
#   IR      depth_module.exposure  in microseconds     (8000 = 8 ms)
# This launch takes `exposure` in MICROSECONDS for both and converts.
#
# Also unchanged from the stock isaac_ros_apriltag realsense example: resolution
# is an argument (1080p OOMs the rectify GPU pool), the tag `size` is an argument
# so pose distance is scaled correctly, and apriltag consumes the RECTIFIED image
# (the stock example fed it the raw image and left rectify orphaned).
#
# Usage:
#   ros2 launch tap_dance realsense_apriltag.launch.py
#   ros2 launch tap_dance realsense_apriltag.launch.py sensor:=infra1
#   ros2 launch tap_dance realsense_apriltag.launch.py sensor:=infra1 width:=848 height:=480 fps:=60
#   ros2 launch tap_dance realsense_apriltag.launch.py sensor:=infra1 auto_exposure:=false exposure:=8000

import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    # OpaqueFunction runs at launch time, so LaunchConfigurations can be resolved
    # to plain Python values here. That is what lets the graph itself change --
    # ComposableNode has no `condition` argument, so RectifyNode can only be
    # omitted by building the node list conditionally.
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    sensor = arg('sensor')
    if sensor not in ('color', 'infra1'):
        raise ValueError(f"sensor must be 'color' or 'infra1', got {sensor!r}")

    width, height, fps = int(arg('width')), int(arg('height')), int(arg('fps'))
    tag_size = float(arg('tag_size'))
    auto_exposure = arg('auto_exposure').lower() in ('true', '1', 'yes')
    exposure_us, gain = int(arg('exposure')), int(arg('gain'))
    profile = f'{width}x{height}x{fps}'
    use_ir = sensor == 'infra1'

    # --- 1) Camera ---------------------------------------------------------
    if use_ir:
        camera_params = {
            'enable_color': False,
            'enable_depth': False,      # we want raw IR frames, not depth
            'enable_infra1': True,
            'enable_infra2': False,
            'depth_module.profile': profile,
            'depth_module.emitter_enabled': 0,
            'depth_module.enable_auto_exposure': auto_exposure,
        }
        if not auto_exposure:
            camera_params['depth_module.exposure'] = exposure_us
            camera_params['depth_module.gain'] = gain
        # Factory-rectified: feed AprilTag directly, no RectifyNode.
        camera_remaps = [
            ('infra1/image_rect_raw', 'image_rect'),
            ('infra1/camera_info', 'camera_info_rect'),
        ]
    else:
        camera_params = {
            'enable_depth': False,
            'enable_infra1': False,
            'enable_infra2': False,
            'rgb_camera.profile': profile,
            'rgb_camera.enable_auto_exposure': auto_exposure,
        }
        if not auto_exposure:
            camera_params['rgb_camera.exposure'] = exposure_us // 100
            camera_params['rgb_camera.gain'] = gain
        camera_remaps = [
            ('color/image_raw', 'image_raw'),
            ('color/camera_info', 'camera_info'),
        ]

    nodes = [ComposableNode(
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        name='realsense2_camera',
        namespace='',
        parameters=[camera_params],
        remappings=camera_remaps,
    )]

    # --- 2) Rectify: colour only; IR is already rectified ------------------
    if not use_ir:
        nodes.append(ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::RectifyNode',
            name='rectify',
            namespace='',
            parameters=[{'output_width': width, 'output_height': height}],
        ))

    # --- 3) AprilTag: GPU detector on the rectified image ------------------
    nodes.append(ComposableNode(
        package='isaac_ros_apriltag',
        plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
        name='apriltag',
        namespace='',
        parameters=[{
            'size': tag_size,
            'max_tags': 64,
            'tile_size': 4,
            'tag_family': 'tag36h11',
            'backends': 'CUDA',
        }],
        remappings=[
            ('image', 'image_rect'),
            ('camera_info', 'camera_info_rect'),
        ],
    ))

    return [ComposableNodeContainer(
        package='rclcpp_components',
        name='apriltag_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=nodes,
        output='screen',
    )]


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'sensor', default_value='color',
            description="'color' (rolling shutter, needs rectify) or 'infra1' "
                        '(global shutter, factory rectified, mono8).'),
        DeclareLaunchArgument(
            'width', default_value='1280',
            description='Image width (px). 1080p OOMs the rectify GPU pool.'),
        DeclareLaunchArgument(
            'height', default_value='720',
            description='Image height (px).'),
        DeclareLaunchArgument(
            'fps', default_value='30',
            description='Frame rate. IR offers 60/90 at 848x480, halving '
                        'inter-frame motion.'),
        DeclareLaunchArgument(
            'tag_size', default_value='0.1225',
            description='AprilTag black-square edge length in meters. Must match '
                        'the tag in view or the reported pose distance is wrong.'),
        DeclareLaunchArgument(
            'auto_exposure', default_value='true',
            description='Set false to pin exposure/gain and stop auto-exposure '
                        'choosing a motion-blurring exposure.'),
        DeclareLaunchArgument(
            'exposure', default_value='8000',
            description='Exposure in MICROSECONDS when auto_exposure:=false; '
                        'converted per sensor (see the note at the top).'),
        DeclareLaunchArgument(
            'gain', default_value='128',
            description='Sensor gain when auto_exposure:=false. Raise to offset a '
                        'short exposure, at the cost of noise.'),
    ]
    return launch.LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
