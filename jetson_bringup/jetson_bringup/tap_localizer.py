# SPDX-License-Identifier: Apache-2.0
#
# Tap localizer — joins the "WHEN" (IMU) to the "WHERE" (camera).
#
#   /tag_detections  (30 Hz, Jetson clock)  ─┐
#                                            ├─▶  where was the tag AT the tap?
#   /wand/tap        (events, M5Stick clock) ─┘         │
#                                                       ▼
#                                        /wand/tap_pixel (PointStamped, pixels)
#
# This is the cross-clock lookup the whole project rests on: the tap instant
# comes from the wand's clock (via micro-ROS session sync, so nominally the same
# epoch as the Jetson), and the pose history comes from the camera. Answering
# "which object was tapped" means indexing one timeline with the other's stamp.
#
# WHY A BUFFER AND NOT LIVE PROCESSING
# The tap is stamped on the M5Stick and crosses WiFi (measured: up to ~90 ms of
# arrival jitter); camera frames take a different path with different latency.
# So a tap routinely arrives BEFORE the camera frame that follows it in time.
# Both streams are therefore buffered, and a tap is held in a pending queue
# until the camera has produced a detection on each side of it -- or until
# max_wait expires, which is itself the answer "the tag was not visible".
#
# WHY INTERPOLATE
# Camera frames are ~33 ms apart; the tap stamp has ~2 ms resolution. Snapping
# to the nearest frame would concede up to 16 ms of wand travel. The report
# includes how far the tag moved between the bracketing frames, so the cost of
# snapping is visible rather than assumed.
#
# WHY HORIZONTAL POSITION IS WHAT MATTERS
# With a webcam-style view and taps coming DOWN onto the tops of objects, the tag
# sits ~10 cm above the contact point, so it appears well above the object in the
# image and is never inside its bounding box. A vertical tap preserves the tag's
# u (horizontal) coordinate while changing v, so association should key on u.
# This node reports both and leaves the matching to the next stage.
#
# Usage:
#   ros2 run jetson_bringup tap_localizer
#   ros2 run jetson_bringup tap_localizer --ros-args -p wand_tag_id:=0

from collections import deque

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class TapLocalizer(Node):

    def __init__(self):
        super().__init__('tap_localizer')

        self._wand_id = self.declare_parameter('wand_tag_id', -1).value
        buffer_seconds = self.declare_parameter('buffer_seconds', 3.0).value
        # How long to hold a tap waiting for the camera to produce a later
        # detection. Must exceed the worst combined arrival jitter (~90 ms
        # measured) plus one frame period (~33 ms).
        self._max_wait = self.declare_parameter('max_wait', 0.30).value
        # If the two bracketing detections are further apart than this, the tag
        # was effectively missing over the tap -- interpolating across it would
        # invent a position.
        self._max_gap = self.declare_parameter('max_gap', 0.12).value

        # 30 Hz * buffer_seconds, with headroom.
        self._poses = deque(maxlen=int(60 * buffer_seconds))
        self._pending = []          # [(tap_t, first_seen_wall_clock)]

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            AprilTagDetectionArray, '/tag_detections', self._on_tags, qos)
        # The firmware publishes taps RELIABLE (each one is a game event), so
        # subscribe reliably to match -- a best-effort sub would still work, but
        # matching QoS keeps the intent explicit.
        self.create_subscription(Header, '/wand/tap', self._on_tap, 10)

        self._pub = self.create_publisher(PointStamped, '/wand/tap_pixel', 10)

        # Retry pending taps even when no new detections arrive, so a tap with
        # no following frame still gets reported (as "tag not visible") instead
        # of sitting in the queue forever.
        self.create_timer(0.05, self._resolve_pending)

        self._n_taps = 0
        self._n_located = 0
        self._n_det_msgs = 0
        self._ids_seen = set()
        self.create_timer(
            self.declare_parameter('status_period', 5.0).value, self._status)
        self.get_logger().info(
            f'wand tag id {self._wand_id if self._wand_id >= 0 else "(any)"}; '
            f'buffering {buffer_seconds:.1f} s of detections, holding taps up to '
            f'{self._max_wait * 1000:.0f} ms')

    def _on_tags(self, msg):
        self._n_det_msgs += 1
        for det in msg.detections:
            # wand_tag_id < 0 means "whichever tag is in view", which is what you
            # want with a single tag: an ID mismatch would otherwise look exactly
            # like the tag never being visible.
            if self._wand_id >= 0 and det.id != self._wand_id:
                continue
            self._ids_seen.add(det.id)
            # center is already in PIXEL coordinates -- no intrinsics, no depth,
            # no deprojection needed for horizontal association.
            self._poses.append(
                (stamp_to_sec(msg.header.stamp), det.center.x, det.center.y))
            break
        self._resolve_pending()

    def _status(self):
        """
        Periodic input health. Without this, an empty pose buffer (wrong tag ID,
        topic not flowing, QoS mismatch) is indistinguishable from the tag simply
        not being visible at the tap -- they produce the same 'NO TAG' warning.
        """
        if self._poses:
            age = self.get_clock().now().nanoseconds * 1e-9 - self._poses[-1][0]
            self.get_logger().info(
                f'inputs: {len(self._poses)} poses buffered '
                f'(newest {age * 1000:.0f} ms old), tag ids seen '
                f'{sorted(self._ids_seen)}, {self._n_taps} taps, '
                f'{self._n_located} located')
        else:
            self.get_logger().warn(
                f'inputs: NO poses buffered after {self._n_det_msgs} detection '
                'messages — wrong wand_tag_id, or /tag_detections not flowing. '
                'Check `ros2 topic echo /tag_detections --once` for the real id.')

    def _on_tap(self, msg):
        self._n_taps += 1
        self._pending.append((stamp_to_sec(msg.stamp),
                              self.get_clock().now().nanoseconds * 1e-9))
        self._resolve_pending()

    def _bracket(self, t):
        """Return the (before, after) detections straddling time t, if any."""
        before = after = None
        for sample in self._poses:
            if sample[0] <= t:
                before = sample          # poses are appended in time order
            elif after is None:
                after = sample
                break
        return before, after

    def _resolve_pending(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        still_pending = []

        for tap_t, first_seen in self._pending:
            before, after = self._bracket(tap_t)

            if before is not None and after is not None:
                self._report(tap_t, before, after)
                continue

            # No later detection yet: the tap may simply have outrun the camera.
            if now - first_seen < self._max_wait:
                still_pending.append((tap_t, first_seen))
                continue

            # Waited long enough. Either the tag was not visible, or the tap
            # predates everything we still hold.
            if before is None and self._poses and tap_t < self._poses[0][0]:
                self.get_logger().warn(
                    f'TAP @ {tap_t:.3f} — older than the buffer; raise '
                    'buffer_seconds')
            else:
                gap = (f'{(tap_t - before[0]) * 1000:.0f} ms after the last '
                       'detection') if before else 'no detections at all'
                self.get_logger().warn(
                    f'TAP @ {tap_t:.3f} — NO TAG at the tap instant ({gap}). '
                    'Tag occluded, blurred, or out of frame during contact.')

        self._pending = still_pending

    def _report(self, tap_t, before, after):
        t0, u0, v0 = before
        t1, u1, v1 = after
        dt = t1 - t0

        if dt > self._max_gap:
            self.get_logger().warn(
                f'TAP @ {tap_t:.3f} — bracketing detections {dt * 1000:.0f} ms '
                f'apart (> max_gap {self._max_gap * 1000:.0f} ms); the tag was '
                'missing over the tap, not interpolating')
            return

        # Linear interpolation between the straddling detections.
        alpha = (tap_t - t0) / dt if dt > 0 else 0.0
        u = u0 + alpha * (u1 - u0)
        v = v0 + alpha * (v1 - v0)

        # How much the tag moved across the bracket, and therefore what snapping
        # to the nearest frame would have cost -- the empirical case for
        # interpolating at all.
        travel = ((u1 - u0) ** 2 + (v1 - v0) ** 2) ** 0.5
        nearest = before if alpha < 0.5 else after
        snap_err = ((u - nearest[1]) ** 2 + (v - nearest[2]) ** 2) ** 0.5

        self._n_located += 1
        self.get_logger().info(
            f'TAP @ {tap_t:.3f} -> u={u:7.1f}  v={v:7.1f}   '
            f'(bracket {dt * 1000:4.1f} ms, alpha {alpha:.2f}, '
            f'tag moved {travel:5.1f} px, snapping would cost {snap_err:4.1f} px)'
            f'   [{self._n_located}/{self._n_taps} located]')

        out = PointStamped()
        out.header.stamp.sec = int(tap_t)
        out.header.stamp.nanosec = int((tap_t % 1.0) * 1e9)
        out.header.frame_id = 'image'      # pixel coordinates, not a TF frame
        out.point.x = float(u)
        out.point.y = float(v)
        out.point.z = 0.0
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TapLocalizer()
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
