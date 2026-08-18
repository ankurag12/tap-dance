# SPDX-License-Identifier: Apache-2.0
#
# Human-readable probe for vision_msgs/Detection2DArray.
#
# Raw `ros2 topic echo /detections_output` is unusable at 30 Hz: the messages
# scroll past and `class_id` is a bare numeric COCO index string ("2"), not a
# name. This node aggregates instead of streaming — every `period` seconds it
# prints one table of what is currently being detected, with class NAMES, plus
# per-class detection rate and score statistics.
#
# Two views, because they answer different questions:
#   * "latest"  -> what is in frame right now (one line per detection)
#   * "summary" -> stability over the window: how many of the last N frames
#                  contained this class, and its score spread. This is the one
#                  that tells you whether a detection is TRUSTWORTHY vs. flickering.
#
# Usage:
#   ros2 run jetson_bringup detection_probe
#   ros2 run jetson_bringup detection_probe --ros-args -p period:=2.0
#   ros2 run jetson_bringup detection_probe --ros-args -p classes:='["car","truck"]'
#   ros2 run jetson_bringup detection_probe --ros-args -p topic:=/detections_output

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection2DArray

# COCO 80-class order, as used by the Ultralytics YOLOv8 export. The decoder
# publishes the INDEX into this list as class_id (a string), so this is the
# lookup that turns "2" into "car".
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


def class_name(class_id):
    """Map the decoder's class_id string to a COCO name; pass through if odd."""
    try:
        return COCO_CLASSES[int(class_id)]
    except (ValueError, IndexError):
        return f'<id {class_id}>'


class DetectionProbe(Node):

    def __init__(self):
        super().__init__('detection_probe')

        topic = self.declare_parameter('topic', '/detections_output').value
        self._period = self.declare_parameter('period', 1.0).value
        # Optional allow-list of class names; empty = show everything.
        self._filter = set(self.declare_parameter('classes', []).value or [])

        # The Isaac ROS decoder publishes best-effort. A reliable subscription
        # would silently receive nothing (the classic QoS-mismatch trap).
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Detection2DArray, topic, self._on_detections, qos)

        self._frames = 0          # messages seen this window
        self._latest = []         # (name, score, cx, cy, w, h) from the last message
        self._per_class = {}      # name -> [count, min_score, max_score, score_sum]

        self.create_timer(self._period, self._report)
        self.get_logger().info(
            f'Probing {topic} — reporting every {self._period:.1f} s'
            + (f', filtered to {sorted(self._filter)}' if self._filter else ''))

    def _on_detections(self, msg):
        self._frames += 1
        latest = []
        for det in msg.detections:
            if not det.results:
                continue
            # Take the top hypothesis (the decoder already ran NMS).
            hyp = det.results[0].hypothesis
            name = class_name(hyp.class_id)
            if self._filter and name not in self._filter:
                continue
            score = hyp.score
            bbox = det.bbox
            latest.append((name, score,
                           bbox.center.position.x, bbox.center.position.y,
                           bbox.size_x, bbox.size_y))

            stats = self._per_class.setdefault(name, [0, 1.0, 0.0, 0.0])
            stats[0] += 1
            stats[1] = min(stats[1], score)
            stats[2] = max(stats[2], score)
            stats[3] += score
        self._latest = latest

    def _report(self):
        if self._frames == 0:
            self.get_logger().warn('no messages on the detections topic')
            return

        lines = [f'--- {self._frames} frames in {self._period:.1f} s '
                 f'({self._frames / self._period:.0f} Hz) ---']

        if self._latest:
            lines.append('  in frame now:')
            for name, score, cx, cy, w, h in sorted(self._latest,
                                                    key=lambda d: -d[1]):
                lines.append(f'    {name:<16} {score:.2f}  '
                             f'center=({cx:6.1f},{cy:6.1f})  {w:5.1f}x{h:5.1f} px')
        else:
            lines.append('  in frame now: (nothing)')

        if self._per_class:
            # hit% = fraction of frames in this window containing the class.
            # A trustworthy detection sits near 100%; a flickering one does not.
            lines.append('  over the window:   hit%   score min/mean/max')
            for name, (count, lo, hi, total) in sorted(
                    self._per_class.items(), key=lambda kv: -kv[1][0]):
                hit = 100.0 * count / self._frames
                lines.append(f'    {name:<16} {hit:5.0f}%   '
                             f'{lo:.2f} / {total / count:.2f} / {hi:.2f}')

        self.get_logger().info('\n'.join(lines))
        self._frames = 0
        self._per_class = {}


def main(args=None):
    rclpy.init(args=args)
    node = DetectionProbe()
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
