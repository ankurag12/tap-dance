# SPDX-License-Identifier: Apache-2.0
#
# hover_probe — one view of both halves of the association problem.
#
#   /tag_detections    (AprilTag, image pixels)   ─┐
#                                                  ├─▶  where is the tag, and
#   /detections_output (YOLOv8, network pixels)   ─┘     which object is it over?
#
# Answers the question the game answers, using the SAME code (tap_dance.targets),
# but without the round loop, timers or scoring in the way. Use it to check target
# positions, to confirm the tag reads where you think it does, and to see the
# tolerance each object actually gets.
#
# Two things it exists to make visible:
#   * The COORDINATE MISMATCH. YOLO bboxes arrive in the network's 640x640 space
#     while AprilTag reports full-image pixels, so the raw numbers differ by the
#     encoder's resize factor (x2 for 1280x720). Both the raw and scaled values
#     are printed, because a wrong scale is otherwise invisible: it just looks
#     like the tag being over the wrong object.
#   * The TOLERANCE each target gets, which is half the distance to its nearest
#     neighbour. Objects placed close together get narrow regions, and a tap
#     measured while the wand is moving then fails to resolve.
#
# Unlike detection_probe, this shows ONLY the classes in `yolo_classes` -- the
# ones that can actually become targets -- rather than everything COCO reports.
#
# Usage:
#   ros2 run tap_dance hover_probe
#   ros2 run tap_dance hover_probe --ros-args -p yolo_classes:='["cup","apple"]'
#   ros2 run tap_dance hover_probe --ros-args -p period:=0.5 -p wand_tag_id:=1

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection2DArray

from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from tap_dance.targets import TargetSet, class_name, network_to_image_scale


class HoverProbe(Node):

    def __init__(self):
        super().__init__('hover_probe')

        self._period = self.declare_parameter('period', 1.0).value
        self._wand_id = self.declare_parameter('wand_tag_id', -1).value
        self._stale = self.declare_parameter('stale_after', 0.5).value

        classes = list(self.declare_parameter(
            'yolo_classes',
            ['cup', 'bottle', 'book', 'mouse', 'keyboard', 'cell phone',
             'remote', 'scissors', 'banana', 'apple']).value)
        img_w = float(self.declare_parameter('image_width', 1280).value)
        img_h = float(self.declare_parameter('image_height', 720).value)
        net_w = float(self.declare_parameter('network_width', 640).value)
        net_h = float(self.declare_parameter('network_height', 640).value)
        self._scale = network_to_image_scale(img_w, img_h, net_w, net_h)

        self._targets = TargetSet(
            classes=classes,
            min_hits=self.declare_parameter('min_yolo_hits', 15).value,
            smoothing=self.declare_parameter('yolo_smoothing', 0.2).value,
            max_halfwidth=self.declare_parameter('max_halfwidth', 400.0).value,
            scale=self._scale)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            AprilTagDetectionArray, '/tag_detections', self._on_tags, qos)
        self.create_subscription(
            Detection2DArray, '/detections_output', self._on_detections, qos)

        self._tag = None            # (t, u, v)
        self._ids_seen = set()
        self._raw = {}              # name -> raw network-space u, for the report
        self.create_timer(self._period, self._report)
        self.get_logger().info(
            f'watching {sorted(classes)}  |  YOLO bbox scale x{self._scale:.3f} '
            f'({int(net_w)}x{int(net_h)} network -> {int(img_w)}x{int(img_h)} image)')

    def _on_tags(self, msg):
        for det in msg.detections:
            if self._wand_id >= 0 and det.id != self._wand_id:
                continue
            self._ids_seen.add(det.id)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._tag = (t, det.center.x, det.center.y)
            return

    def _on_detections(self, msg):
        for det in msg.detections:
            if det.results:
                name = class_name(det.results[0].hypothesis.class_id)
                self._raw[name] = det.bbox.center.position.x
        self._targets.update_from_detections(msg.detections)

    def _report(self):
        lines = []

        if self._targets.targets:
            lines.append('  objects (only classes that can become targets):')
            lines.append('      name              u(img)   u(net)  tol +/-   hits  score')
            for name, u in self._targets.targets:
                raw = self._raw.get(name)
                lines.append(
                    f'      {name:<16} {u:7.1f}  '
                    f'{("%7.1f" % raw) if raw is not None else "      -"}  '
                    f'{self._targets.tolerance[name]:7.0f}  '
                    f'{self._targets.hits(name):5d}  {self._targets.score(name):.2f}')
        else:
            waiting = sorted(self._raw)
            lines.append(f'  objects: none stable yet'
                         + (f' (seen but below min_yolo_hits: {waiting})'
                            if waiting else ' — is YOLO running?'))

        now = self.get_clock().now().nanoseconds * 1e-9
        if self._tag is None:
            lines.append('  tag: never seen')
        else:
            t, u, v = self._tag
            age = now - t
            if age > self._stale:
                lines.append(f'  tag: NOT seen for {age * 1000:.0f} ms')
            else:
                over = self._targets.match(u)
                lines.append(f'  tag: u={u:7.1f} v={v:7.1f}  ->  '
                             f'{("OVER " + over.upper()) if over else "not over any target"}')
                # Distance to every target, so a near-miss is distinguishable from
                # being nowhere near -- the difference between nudging an object
                # and re-measuring everything.
                if self._targets.targets:
                    gaps = '   '.join(
                        f'{n}:{u - cu:+.0f}'
                        f'{"" if abs(u - cu) <= self._targets.tolerance[n] else "(out)"}'
                        for n, cu in self._targets.targets)
                    lines.append(f'       offsets  {gaps}')

        if self._wand_id < 0 and len(self._ids_seen) > 1:
            lines.append(f'  WARNING: tag ids {sorted(self._ids_seen)} seen but '
                         'wand_tag_id is -1 (any) — a spurious tag may be tracked; '
                         'pin it with wand_tag_id:=<id>')

        self.get_logger().info('\n' + '\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = HoverProbe()
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
