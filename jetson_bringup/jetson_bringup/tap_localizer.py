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
        # invent a position, so fall back to the last good pose instead.
        self._max_gap = self.declare_parameter('max_gap', 0.12).value
        # Fallback: how stale the last detection BEFORE the tap may be and still
        # be usable. Justified by geometry, not convenience -- a vertical tap
        # changes the tag's v but preserves its u, and association keys on u. So
        # a detection from shortly before contact carries the horizontal position
        # we need, and demanding a detection AFTER the tap discards good data.
        #
        # Deliberately generous (500 ms), because staleness is NOT the real risk
        # and is already guarded better elsewhere: what matters is how far the tag
        # drifted HORIZONTALLY over that interval, which is reported as the
        # uncertainty and gated downstream. A 400 ms-old pose from a hovering wand
        # is worth more than rejecting the tap outright; a 400 ms-old pose from a
        # wand still travelling produces a large drift estimate and gets rejected
        # on its own merits. A tight cutoff here just discards the good case too.
        self._max_stale = self.declare_parameter('max_stale', 0.50).value
        # If the pose immediately before the tap is at least this fresh, resolve
        # IMMEDIATELY instead of waiting max_wait for a bracket. Measured: hover
        # taps move ~0-1 px between frames, so interpolation adds under a pixel
        # of accuracy -- not worth 300 ms of feedback latency in a reaction-time
        # game. Waiting still happens when the last pose is older than this.
        self._fresh_enough = self.declare_parameter('fresh_enough', 0.06).value

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
        # A tap the firmware detected but the camera could not place. Published
        # so the application can TELL THE PLAYER their input was received but
        # unusable -- silently dropping it looks identical to the tap detector
        # having missed, and the player just waits, inflating their score.
        self._pub_unlocated = self.create_publisher(Header, '/wand/tap_unlocated', 10)

        # Retry pending taps even when no new detections arrive, so a tap with
        # no following frame still gets reported (as "tag not visible") instead
        # of sitting in the queue forever.
        self.create_timer(0.05, self._resolve_pending)

        self._n_taps = 0
        self._n_located = 0
        self._n_interp = 0
        self._n_last = 0
        self._n_det_msgs = 0
        self._n_wand_dets = 0
        self._ids_seen = set()
        self._status_period = self.declare_parameter('status_period', 5.0).value
        self.create_timer(self._status_period, self._status)
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
            self._n_wand_dets += 1
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
            msg_hz = self._n_det_msgs / self._status_period
            wand_hz = self._n_wand_dets / self._status_period
            # msg_hz is the camera pipeline's rate; wand_hz is how often the tag
            # is actually FOUND. A big gap between them means the tag is being
            # missed -- too small, too oblique, or motion-blurred -- which is the
            # real limit on how well any tap can be localized.
            self.get_logger().info(
                f'inputs: detections {msg_hz:.0f} Hz, wand tag found '
                f'{wand_hz:.0f} Hz ({100.0 * wand_hz / msg_hz if msg_hz else 0:.0f}% '
                f'of frames), newest {age * 1000:.0f} ms old, ids {sorted(self._ids_seen)}'
                f'  |  taps {self._n_taps}, located {self._n_located} '
                f'({self._n_interp} interp, {self._n_last} last-pose)')
        else:
            self.get_logger().warn(
                f'inputs: NO poses buffered after {self._n_det_msgs} detection '
                'messages — wrong wand_tag_id, or /tag_detections not flowing. '
                'Check `ros2 topic echo /tag_detections --once` for the real id.')
        self._n_det_msgs = 0
        self._n_wand_dets = 0

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

            # Best case: detections on both sides, close enough to interpolate.
            if before is not None and after is not None \
                    and (after[0] - before[0]) <= self._max_gap:
                self._report_interp(tap_t, before, after)
                continue

            # Already have a fresh pose from just before the tap: resolve NOW
            # rather than spending max_wait chasing a bracket worth under a pixel.
            if before is not None and (tap_t - before[0]) <= self._fresh_enough:
                self._report_last(tap_t, before)
                continue

            # Keep waiting -- a later detection may still arrive and allow
            # interpolation, which beats a stale fallback.
            if now - first_seen < self._max_wait:
                still_pending.append((tap_t, first_seen))
                continue

            # Fallback: last detection before the tap. Valid because a vertical
            # tap preserves u; see max_stale.
            if before is not None and (tap_t - before[0]) <= self._max_stale:
                self._report_last(tap_t, before)
                continue

            if before is None and self._poses and tap_t < self._poses[0][0]:
                self.get_logger().warn(
                    f'TAP @ {tap_t:.3f} — older than the buffer; raise '
                    'buffer_seconds')
            elif before is not None:
                self.get_logger().warn(
                    f'TAP @ {tap_t:.3f} — last detection was '
                    f'{(tap_t - before[0]) * 1000:.0f} ms before the tap '
                    f'(> max_stale {self._max_stale * 1000:.0f} ms); tag lost '
                    'too long before contact')
            else:
                self.get_logger().warn(
                    f'TAP @ {tap_t:.3f} — NO TAG at all near the tap instant.')
            self._announce_unlocated(tap_t)

        self._pending = still_pending

    def _announce_unlocated(self, tap_t):
        hdr = Header()
        hdr.stamp.sec = int(tap_t)
        hdr.stamp.nanosec = int((tap_t % 1.0) * 1e9)
        hdr.frame_id = 'unlocated'
        self._pub_unlocated.publish(hdr)

    def _h_speed(self, sample):
        """Horizontal pixel speed just before `sample`, to sanity-check the
        'u is unchanged' assumption the fallback relies on."""
        prev = None
        for s in self._poses:
            if s[0] >= sample[0]:
                break
            prev = s
        if prev is None or sample[0] == prev[0]:
            return None
        return abs(sample[1] - prev[1]) / (sample[0] - prev[0])   # px/s

    def _report_interp(self, tap_t, before, after):
        t0, u0, v0 = before
        t1, u1, v1 = after
        dt = t1 - t0
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
        self._n_interp += 1
        self.get_logger().info(
            f'TAP @ {tap_t:.3f} -> u={u:7.1f}  v={v:7.1f}  [interp]  '
            f'(bracket {dt * 1000:4.1f} ms, alpha {alpha:.2f}, '
            f'moved {travel:5.1f} px, snapping would cost {snap_err:4.1f} px)'
            f'   [{self._n_located}/{self._n_taps} located]')
        # Interpolating between close frames leaves at most a fraction of the
        # inter-frame travel as error; half of it is a fair bound.
        self._publish(tap_t, u, v, abs(u1 - u0) / 2.0)

    def _report_last(self, tap_t, before):
        t0, u0, v0 = before
        age_ms = (tap_t - t0) * 1000.0
        speed = self._h_speed(before)
        # Horizontal drift over the staleness window is the error this fallback
        # actually incurs -- if the wand was hovering, it is near zero.
        drift = f'{speed * (tap_t - t0):.1f} px' if speed is not None else 'n/a'

        self._n_located += 1
        self._n_last += 1
        self.get_logger().info(
            f'TAP @ {tap_t:.3f} -> u={u0:7.1f}  v={v0:7.1f}  [last]    '
            f'(pose {age_ms:5.1f} ms old, horiz speed '
            f'{"n/a" if speed is None else f"{speed:6.0f} px/s"}, '
            f'implied u drift {drift})   [{self._n_located}/{self._n_taps} located]')
        # The drift over the staleness window IS the uncertainty here.
        self._publish(tap_t, u0, v0,
                      0.0 if speed is None else speed * (tap_t - t0))

    def _publish(self, tap_t, u, v, u_uncertainty):
        out = PointStamped()
        out.header.stamp.sec = int(tap_t)
        out.header.stamp.nanosec = int((tap_t % 1.0) * 1e9)
        out.header.frame_id = 'image'      # pixel coordinates, not a TF frame
        out.point.x = float(u)
        out.point.y = float(v)
        # z carries the estimated HORIZONTAL uncertainty in pixels, not a
        # coordinate. Deliberate reuse of an otherwise-unused field so consumers
        # can reject untrustworthy taps without a custom message type: the
        # measured worst case was a 188 ms-old pose at 309 px/s = 58 px of drift,
        # which must not be treated as equal to a 0.4 px interpolation.
        out.point.z = float(u_uncertainty)
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
