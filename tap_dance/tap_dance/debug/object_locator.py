# SPDX-License-Identifier: Apache-2.0
#
# Locate detected objects in 3D: YOLOv8 boxes + aligned depth -> metric points.
#
#   /detections_output (2D boxes)  ─┐
#   /aligned_depth (16UC1, mm)     ─┼─▶  deproject  ─▶  /objects_3d (Detection3DArray)
#   /camera_info (fx, fy, cx, cy)  ─┘
#
# Depth must be ALIGNED TO COLOR so a colour pixel and a depth pixel mean the
# same ray; run realsense with align_depth.enable:=true. The bbox coordinates,
# the aligned depth image and camera_info must all be at the same resolution --
# pass the colour resolution as YOLO's input_image_width/height or every
# deprojection is silently scaled wrong.
#
# Two robustness choices worth knowing:
#   * Depth is the MEDIAN over the middle of the box, not the centre pixel.
#     A centre pixel can land on a specular highlight or a hole and read 0;
#     the median over the inner region survives that. Zeros are dropped
#     (RealSense reports invalid depth as 0, which would otherwise pull the
#     median toward the camera).
#   * Positions are SMOOTHED and CACHED per class. Desk objects are stationary,
#     so an object stays targetable when the wand occludes it or YOLO blinks
#     for a frame -- which it will.
#
# Limitation (v1): one instance per class, keeping the highest-scoring box.
# Two cups on the desk collapse to one object. Clustering by 3D position would
# fix it; not needed for the first playable version.
#
# Usage:
#   ros2 run tap_dance object_locator
#   ros2 run tap_dance object_locator --ros-args -p classes:='["cup","book"]'

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import (BoundingBox3D, Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

from tap_dance.targets import class_name


class ObjectLocator(Node):

    def __init__(self):
        super().__init__('object_locator')

        depth_topic = self.declare_parameter(
            'depth_topic', '/camera/aligned_depth_to_color/image_raw').value
        info_topic = self.declare_parameter(
            'info_topic', '/camera/color/camera_info').value
        det_topic = self.declare_parameter('detections_topic', '/detections_output').value

        # Optional allow-list; empty = accept every COCO class.
        self._classes = set(self.declare_parameter('classes', []).value or [])
        self._min_score = self.declare_parameter('min_score', 0.35).value
        # Fraction of the box used for the depth median (0.5 = inner half).
        self._inner = self.declare_parameter('inner_fraction', 0.5).value
        # EMA weight for a new observation. Low = steady, slow to move.
        self._alpha = self.declare_parameter('smoothing', 0.3).value
        self._frame = self.declare_parameter(
            'frame_id', 'camera_color_optical_frame').value

        self._k = None          # (fx, fy, cx, cy)
        self._objects = {}      # name -> {'pos': np.array, 'score': float, 'size': (w,h)}

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos)

        # Detections and depth come from different pipelines with different
        # latencies, so pair them by timestamp rather than assuming lockstep.
        from vision_msgs.msg import Detection2DArray
        self._sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Detection2DArray, det_topic, qos_profile=qos),
             Subscriber(self, Image, depth_topic, qos_profile=qos)],
            queue_size=10, slop=0.08)
        self._sync.registerCallback(self._on_pair)

        self._pub = self.create_publisher(Detection3DArray, '/objects_3d', 10)
        self.create_timer(1.0, self._publish)

        self.get_logger().info(
            f'depth={depth_topic}  info={info_topic}  det={det_topic}  '
            + (f'classes={sorted(self._classes)}' if self._classes else 'all classes'))

    def _on_info(self, msg):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self._k = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_pair(self, dets, depth_msg):
        if self._k is None:
            return
        fx, fy, cx, cy = self._k

        if depth_msg.encoding not in ('16UC1', 'mono16'):
            self.get_logger().warn(
                f'unexpected depth encoding {depth_msg.encoding!r}; expected 16UC1 (mm)',
                throttle_duration_sec=10.0)
            return

        h, w = depth_msg.height, depth_msg.width
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(h, w)

        for det in dets.detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            name = class_name(hyp.class_id)
            if self._classes and name not in self._classes:
                continue
            if hyp.score < self._min_score:
                continue

            bb = det.bbox
            # Inner region only: box edges straddle the background, and their
            # depth would drag the median off the object.
            half_w = max(1.0, bb.size_x * self._inner / 2.0)
            half_h = max(1.0, bb.size_y * self._inner / 2.0)
            u0 = int(max(0, bb.center.position.x - half_w))
            u1 = int(min(w, bb.center.position.x + half_w))
            v0 = int(max(0, bb.center.position.y - half_h))
            v1 = int(min(h, bb.center.position.y + half_h))
            if u1 <= u0 or v1 <= v0:
                continue

            patch = depth[v0:v1, u0:u1]
            valid = patch[patch > 0]
            if valid.size < 10:
                continue                      # object has no usable depth
            z = float(np.median(valid)) / 1000.0   # mm -> m

            u, v = bb.center.position.x, bb.center.position.y
            pos = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])

            prev = self._objects.get(name)
            if prev is None:
                self._objects[name] = {'pos': pos, 'score': hyp.score,
                                       'size': (bb.size_x, bb.size_y)}
                self.get_logger().info(f'located {name} at '
                                       f'({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:.2f}) m')
            else:
                a = self._alpha
                prev['pos'] = (1.0 - a) * prev['pos'] + a * pos
                prev['score'] = max(prev['score'], hyp.score)
                prev['size'] = (bb.size_x, bb.size_y)

    def _publish(self):
        if not self._objects:
            return
        out = Detection3DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._frame

        for name, obj in self._objects.items():
            d = Detection3D()
            d.header = out.header
            d.id = name

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = float(obj['score'])
            hyp.pose.pose.position.x = float(obj['pos'][0])
            hyp.pose.pose.position.y = float(obj['pos'][1])
            hyp.pose.pose.position.z = float(obj['pos'][2])
            hyp.pose.pose.orientation.w = 1.0
            d.results.append(hyp)

            bb = BoundingBox3D()
            bb.center = hyp.pose.pose
            # Rough metric extent from the pixel box at the measured range;
            # used only for display, not for hit-testing.
            z = float(obj['pos'][2])
            fx, fy = self._k[0], self._k[1]
            bb.size.x = float(obj['size'][0]) * z / fx
            bb.size.y = float(obj['size'][1]) * z / fy
            bb.size.z = min(bb.size.x, bb.size.y)
            d.bbox = bb

            out.detections.append(d)

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectLocator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
