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
#   ros2 run tap_dance tap_localizer
#   ros2 run tap_dance tap_localizer --ros-args -p wand_tag_id:=0

from collections import deque


def _solve3(A, b):
    """Gaussian elimination on a 3x3 system; None if singular."""
    M = [row[:] + [rhs] for row, rhs in zip(A, b)]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda r: abs(M[r][i]))
        if abs(M[pivot][i]) < 1e-12:
            return None
        M[i], M[pivot] = M[pivot], M[i]
        for r in range(3):
            if r == i:
                continue
            f = M[r][i] / M[i][i]
            for c in range(i, 4):
                M[r][c] -= f * M[i][c]
    return [M[i][3] / M[i][i] for i in range(3)]

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
        # How many recent poses the extrapolation fits. 4 is enough to see
        # deceleration without reaching back into a different phase of the motion.
        self._fit_poses = self.declare_parameter('fit_poses', 4).value
        # Fraction of the extrapolated distance charged as uncertainty.
        self._reach_penalty = self.declare_parameter('reach_penalty', 0.4).value

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
        # Beyond this the status line reports "not seen" instead of a position.
        self._live_max_age = self.declare_parameter('live_max_age', 0.5).value
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
            # Only quote a live position if the sighting is actually current.
            # Showing the last known u regardless would read as a valid reading
            # while the tag is hidden -- misleading exactly when this line is
            # being used to measure target positions.
            live = (f'tag now at u={self._poses[-1][1]:.0f} '
                    f'v={self._poses[-1][2]:.0f}' if age <= self._live_max_age
                    else f'tag NOT seen for {age * 1000:.0f} ms')
            # msg_hz is the camera pipeline's rate; wand_hz is how often the tag
            # is actually FOUND. A big gap between them means the tag is being
            # missed -- too small, too oblique, or motion-blurred -- which is the
            # real limit on how well any tap can be localized.
            # Report the tag's CURRENT pixel position too: target_u values for
            # tap_game are measured by hovering the wand over each object, and
            # having to tap repeatedly just to read a column is needless. They
            # also have to be re-measured whenever the imager changes -- the IR
            # and colour lenses are physically offset, so the same object sits at
            # a different column in each.
            self.get_logger().info(
                f'inputs: detections {msg_hz:.0f} Hz, wand tag found '
                f'{wand_hz:.0f} Hz ({100.0 * wand_hz / msg_hz if msg_hz else 0:.0f}% '
                f'of frames), newest {age * 1000:.0f} ms old, ids {sorted(self._ids_seen)}'
                f'  |  {live}'
                f'  |  taps {self._n_taps}, located {self._n_located} '
                f'({self._n_interp} interp, {self._n_last} last-pose)')
            # A spurious detection (id 211 was seen once in a cluttered scene at
            # short exposure) is indistinguishable from the wand when any tag is
            # accepted, and the localizer would silently track it instead.
            if self._wand_id < 0 and len(self._ids_seen) > 1:
                self.get_logger().warn(
                    f'{len(self._ids_seen)} tag ids seen {sorted(self._ids_seen)} '
                    'but wand_tag_id is -1 (any), so the wrong tag may be tracked '
                    '— pin it with wand_tag_id:=<id>')
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

    def _extrapolate(self, sample, tap_t):
        """
        Estimate u at the tap, and how much to trust it, from the last few poses.

        Returns (u_at_tap, uncertainty_px).

        A constant-velocity extrapolation of the last inter-frame speed is
        systematically PESSIMISTIC here, because the wand decelerates into a tap:
        with a 136 ms-old pose at 542 px/s it predicted 74 px of movement when the
        wand had nearly stopped, and the game then rejected a perfectly good tap as
        positionally ambiguous.

        So fit a quadratic to the last few poses, which can represent that
        deceleration, and take the uncertainty as the DISAGREEMENT between that fit
        and the naive constant-velocity guess. When both models agree the position
        is trustworthy however long the gap; when they diverge it genuinely is not.
        """
        recent = [x for x in self._poses if x[0] <= sample[0]][-self._fit_poses:]
        t0 = sample[0]
        if len(recent) < 2:
            return sample[1], 0.0

        ts = [x[0] - t0 for x in recent]
        us = [x[1] for x in recent]
        dt = tap_t - t0

        # constant velocity from the last pair -- the previous behaviour
        v_last = (us[-1] - us[-2]) / (ts[-1] - ts[-2]) if ts[-1] != ts[-2] else 0.0
        u_linear = us[-1] + v_last * dt

        if len(recent) < 3:
            return us[-1], abs(u_linear - us[-1])

        # least-squares quadratic u(t) = a t^2 + b t + c, solved via normal
        # equations to avoid a numpy dependency in this node
        n = len(ts)
        S = [sum(t ** k for t in ts) for k in range(5)]
        T = [sum(u * t ** k for u, t in zip(us, ts)) for k in range(3)]
        A = [[S[4], S[3], S[2]],
             [S[3], S[2], S[1]],
             [S[2], S[1], float(n)]]
        b = [T[2], T[1], T[0]]
        coef = _solve3(A, b)
        if coef is None:
            return us[-1], abs(u_linear - us[-1])
        a, bb, c = coef
        u_fit = a * dt * dt + bb * dt + c

        # Uncertainty is the worst of three things:
        #   * disagreement between the quadratic and constant-velocity models
        #   * the fit's own residual, so a noisy fit is not reported as certain
        #   * a fraction of how far we extrapolated -- without this, perfectly
        #     linear motion reports zero uncertainty even after extrapolating
        #     100+ px, which is over-confident: the wand is about to hit something
        #     and stop, and exactly when is not observable from past frames.
        rms = (sum((a * t * t + bb * t + c - u) ** 2
                   for t, u in zip(ts, us)) / n) ** 0.5
        reach = abs(u_fit - us[-1])
        return u_fit, max(abs(u_fit - u_linear), rms, self._reach_penalty * reach)

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
        u_est, unc = self._extrapolate(before, tap_t)

        self._n_located += 1
        self._n_last += 1
        self.get_logger().info(
            f'TAP @ {tap_t:.3f} -> u={u_est:7.1f}  v={v0:7.1f}  [extrap]  '
            f'(pose {age_ms:5.1f} ms old, last seen at u={u0:.0f}, '
            f'+/-{unc:.0f} px)   [{self._n_located}/{self._n_taps} located]')
        self._publish(tap_t, u_est, v0, unc)

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
