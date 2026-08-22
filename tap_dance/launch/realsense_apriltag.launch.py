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
# WHAT ACTUALLY LIMITS DETECTION: exposure, plus enough light to use it.
#
# Measured (found % of frames containing the tag, waving at game pace):
#   colour, auto exposure           38-70%   dim room, varies run to run
#   colour, 4 ms, dim room          50-70%
#   IR, auto exposure               50%      AE picks a long exposure: ambient
#                                            near-IR is scarce, emitter off
#   IR, 2-4 ms, daylight            100%
#   colour, 2-4 ms, daylight        ~100%
#
# The last two lines are the point. An earlier reading suggested the colour
# imager's ROLLING SHUTTER was to blame -- it reads rows sequentially, so a moving
# tag is sheared as well as blurred, and no exposure setting fixes shear. That
# theory does not survive the data: once the room was bright enough to expose a
# 2 ms frame, colour reached ~100% too. The apparent shutter advantage was
# confounded by scene brightness, because the colour sweep ran in a dimmer room
# than the IR test.
#
# So COLOUR IS THE DEFAULT, despite IR looking better mid-investigation:
#   * Visible light is abundant in any lit room; near-IR is not. LED lighting
#     emits almost none, and the projector cannot help because its dot pattern
#     lands on the tag. Colour is the more robust choice, not the weaker one.
#   * Tag and YOLO detections then share ONE coordinate frame. The imagers are
#     physically offset, so mixing them would need a registration step.
#
# The IR path is kept (sensor:=infra1) and remains useful:
#   * Global shutter, if shear ever does turn out to matter for faster motion.
#   * Factory rectified, so RectifyNode is skipped.
#   * Held a steady 30 FPS over USB where colour measured ~20 and jittery, and
#     offers 848x480 at 60/90 FPS for less inter-frame motion.
#
# IR trade-offs:
#   * The projector MUST be off (emitter_enabled 0) or its dot pattern lands on
#     the tag. With it off the imagers rely on ambient IR -- adequate in this
#     room, but LED-only lighting emits little.
#   * Lower resolutions shrink the tag: a 12 cm tag spans ~40 px at 1280x720 but
#     only ~27 px at 848x480, near the decode floor. Start at 720p so the shutter
#     is the only variable changed.
#   * cuAprilTags rejects mono8 outright ("only 'rgb8' or 'bgr8' image input"),
#     so the IR path needs an ImageFormatConverterNode to widen Y8 to rgb8. That
#     cancels the node saved by skipping rectify -- IR is still worth it for the
#     shutter, but not for pipeline length.
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

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
            # The D456's own IMU is unused -- inertial data comes from the wand's
            # M5Stick -- and the motion module adds traffic to a USB link that has
            # thrown control_transfer errors before.
            'enable_gyro': False,
            'enable_accel': False,
        }
        if not auto_exposure:
            camera_params['depth_module.exposure'] = exposure_us
            camera_params['depth_module.gain'] = gain
        # Factory-rectified, so no RectifyNode -- but the format converter
        # below still has to widen Y8 to rgb8, so publish into its input.
        camera_remaps = [
            ('infra1/image_rect_raw', 'ir_mono'),
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

    # --- 2a) IR only: mono8 -> rgb8, because cuAprilTags refuses mono8 -------
    if use_ir:
        nodes.append(ComposableNode(
            package='isaac_ros_image_proc',
            plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
            name='ir_to_rgb',
            namespace='',
            parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': width,
                'image_height': height,
            }],
            remappings=[
                ('image_raw', 'ir_mono'),
                ('image', 'image_rect'),
            ],
        ))

    # --- 2b) Rectify: colour only; IR is already rectified ------------------
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

    # --- 4) Optional: YOLOv8 via TensorRT, for naming the objects -----------
    # Feeds from image_rect, the SAME image AprilTag consumes. That matters:
    # rectification moves pixels (most at the edges), so a bbox centre and a tag
    # centre are only comparable if both were measured in the rectified frame.
    # This is why the game can match objects to taps with no depth, no
    # intrinsics and no cross-sensor registration.
    extra = []
    if arg('use_yolo').lower() in ('true', '1', 'yes'):
        if use_ir:
            raise ValueError(
                'use_yolo with sensor:=infra1 is not supported: the COCO-trained '
                'model expects colour, and grayscale-widened-to-rgb8 is out of '
                'distribution. Use sensor:=color.')

        nodes.append(ComposableNode(
            name='tensor_rt',
            package='isaac_ros_tensor_rt',
            plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
            namespace='',
            parameters=[{
                'engine_file_path': arg('engine_file_path'),
                'input_tensor_names': ['input_tensor'],
                # Binding names come from the Ultralytics ONNX export -- the
                # classic gotcha, since the defaults do not match.
                'input_binding_names': ['images'],
                'output_tensor_names': ['output_tensor'],
                'output_binding_names': ['output0'],
                'verbose': False,
                'force_engine_update': False,
            }],
        ))
        nodes.append(ComposableNode(
            name='yolov8_decoder_node',
            package='isaac_ros_yolov8',
            plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
            namespace='',
            parameters=[{
                'confidence_threshold': float(arg('confidence_threshold')),
                'nms_threshold': float(arg('nms_threshold')),
            }],
        ))

        # The encoder ships as its own launch file and can attach into an
        # existing container, so the whole graph stays in one process and NITROS
        # keeps the images on the GPU.
        encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
        extra.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]),
            launch_arguments={
                'input_image_width': str(width),
                'input_image_height': str(height),
                'network_image_width': '640',
                'network_image_height': '640',
                # The engine was exported without normalisation baked in.
                'image_mean': '[0.0, 0.0, 0.0]',
                'image_stddev': '[1.0, 1.0, 1.0]',
                'attach_to_shared_component_container': 'True',
                'component_container_name': 'apriltag_container',
                'dnn_image_encoder_namespace': 'yolov8_encoder',
                'image_input_topic': '/image_rect',
                'camera_info_input_topic': '/camera_info_rect',
                'tensor_output_topic': '/tensor_pub',
            }.items(),
        ))

    return [ComposableNodeContainer(
        package='rclcpp_components',
        name='apriltag_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=nodes,
        output='screen',
        # The camera and NITROS log heavily at INFO. When this graph runs under
        # tap_game.launch.py that chatter buries the game's prompts, so the level
        # is settable rather than fixed.
        arguments=['--ros-args', '--log-level', arg('log_level')],
    )] + extra


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'sensor', default_value='color',
            description="'color' (default: reaches ~100% at 2 ms in a lit room, "
                        "and shares a frame with YOLO) or 'infra1' (global "
                        'shutter, factory rectified, but needs a near-IR source).'),
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
            'auto_exposure', default_value='false',
            description='Defaults OFF. Auto-exposure optimises brightness and '
                        'picks a long, motion-blurring exposure; pinning it short '
                        'took detection in motion from ~50% to ~100%.'),
        DeclareLaunchArgument(
            'exposure', default_value='2000',
            description='Exposure in MICROSECONDS when auto_exposure:=false; '
                        'converted per sensor (see the note at the top). 2 ms and '
                        '4 ms measured alike in a bright room; 2 ms freezes faster '
                        'motion, so it is the default. Needs a well-lit scene -- '
                        'in a dim room a short exposure underexposes and detection '
                        'falls.'),
        DeclareLaunchArgument(
            'log_level', default_value='INFO',
            description='Log level for the perception container. WARN hides the '
                        'camera/NITROS startup chatter.'),
        DeclareLaunchArgument(
            'use_yolo', default_value='false',
            description='Also run YOLOv8/TensorRT on the rectified image, so the '
                        'game can name real objects instead of using '
                        'hand-measured target positions. Colour only.'),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='/workspaces/isaac_ros-dev/models/yolov8s.plan',
            description='TensorRT engine built from the Ultralytics ONNX export.'),
        DeclareLaunchArgument(
            'confidence_threshold', default_value='0.35',
            description='YOLOv8 detection confidence threshold.'),
        DeclareLaunchArgument(
            'nms_threshold', default_value='0.45',
            description='YOLOv8 non-maximum-suppression IoU threshold.'),
        DeclareLaunchArgument(
            'gain', default_value='128',
            description='Sensor gain when auto_exposure:=false. The stereo module '
                        'accepts more (try 248 with sensor:=infra1); raise it to '
                        'offset a short exposure, at the cost of noise.'),
    ]
    return launch.LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
