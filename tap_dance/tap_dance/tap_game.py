# SPDX-License-Identifier: Apache-2.0
#
# Whack-a-mole tap game — the application layer.
#
#   /wand/tap_pixel (PointStamped: u, v, u_uncertainty)  ─▶  match to a target
#                                                             region, score it
#
# Round loop: announce a target -> player brings the wand over it and taps ->
# match the tap's horizontal pixel position to a region -> HIT / WRONG / MISS.
#
# WHY HORIZONTAL REGIONS AND NOT BOUNDING BOXES
# With a webcam-style view, taps come DOWN onto the tops of objects, so the tag
# sits ~10 cm above the contact point and never falls inside an object's box. A
# vertical tap changes the tag's v but preserves its u, so a target is a range of
# u. Measured tap positions were well separated (u = 131, 585, 700, 846), so this
# is sufficient. Targets are hand-configured here; swapping in YOLO later means
# replacing where target_u comes from, not how matching works.
#
# WHY REACTION TIME USES THE TAP'S STAMP, NOT ITS ARRIVAL
# WiFi arrival jitter was measured up to ~90 ms; scoring on arrival would add
# that noise to every score. The prompt time comes from the JETSON's clock and
# the tap stamp from the M5STICK's, so each score is a cross-clock measurement --
# if micro-ROS session sync were off by 100 ms, every reaction time would be
# wrong by 100 ms. The time synchronisation is load-bearing inside the feature,
# not just in a side experiment.
#
# WHY STALE TAPS ARE REJECTED
# A tap stamped BEFORE the prompt cannot be a response to it -- it is a leftover
# in flight, or the ring-down of the previous round's tap. Without this check it
# would register as an impossibly fast reaction.
#
# WHY UNCERTAIN TAPS ARE REJECTED
# tap_localizer reports estimated horizontal uncertainty (point.z). When the tag
# was last seen 188 ms before contact while moving at 309 px/s, that is 58 px of
# drift -- comparable to the region size, so the answer is not trustworthy.
# Rejecting is honest; guessing would silently corrupt the score.
#
# Usage (measure your target u values first, e.g. with tap_localizer):
#   ros2 run tap_dance tap_game --ros-args \
#     -p target_names:='["cup","book","mug","pen"]' \
#     -p target_u:='[131.0, 450.0, 700.0, 866.0]'

import random

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header, String

from vision_msgs.msg import Detection2DArray

from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from tap_dance.detection_probe import class_name


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class TapGame(Node):

    def __init__(self):
        super().__init__('tap_game')

        names = list(self.declare_parameter(
            'target_names', ['left', 'mid_left', 'mid_right', 'right']).value)
        us = list(self.declare_parameter(
            'target_u', [150.0, 450.0, 750.0, 1050.0]).value)
        if len(names) != len(us):
            raise ValueError(
                f'target_names ({len(names)}) and target_u ({len(us)}) must match')
        self._targets = list(zip(names, [float(u) for u in us]))

        # Per-target tolerance = half the distance to that target's NEAREST
        # NEIGHBOUR, capped by max_halfwidth. A single global halfwidth taken
        # from the tightest pair would make every region as strict as the worst
        # one -- an isolated target 450 px from anything else would still reject
        # a tap 60 px off. Per-target keeps close pairs unambiguous while leaving
        # isolated targets generous.
        # Cap exists only so a single isolated target cannot claim the whole
        # image; with two or more targets the half-distance to the nearest
        # neighbour is the natural boundary and regions should MEET. At 200 the
        # cap bound instead, leaving a 335 px dead zone between targets 735 px
        # apart, so taps 16 px outside a region scored as "between targets" even
        # though the intended target was unambiguous.
        max_hw = self.declare_parameter('max_halfwidth', 400.0).value
        self._tolerance = {}
        for i, (name, cu) in enumerate(self._targets):
            others = [abs(cu - ou) for j, (_, ou) in enumerate(self._targets) if j != i]
            self._tolerance[name] = min(max_hw, (min(others) / 2.0) if others else max_hw)

        self._rounds = self.declare_parameter('rounds', 10).value
        self._time_limit = self.declare_parameter('time_limit', 6.0).value
        self._start_delay = self.declare_parameter('start_delay', 3.0).value
        # Reject a tap whose horizontal uncertainty is a large fraction of the
        # tightest region -- see the header note.
        # Absolute backstop only. The real test is whether the uncertainty
        # changes WHICH target the tap resolves to (see _on_tap_pixel): a fixed
        # threshold rejected a tap uncertain to 101 px against targets 735 px
        # apart, where 101 px cannot possibly confuse one for the other.
        self._max_uncertainty = self.declare_parameter(
            'max_uncertainty', 250.0).value
        # Game-level refractory, measured from the last SCORED tap. The device
        # refractory (100 ms) collapses the ring-down of one contact, but a
        # follow-through or second bounce ~500 ms later is a genuine new event --
        # and since a scored tap immediately advances the round, it was landing
        # as the next round's answer.
        self._tap_lockout = self.declare_parameter('tap_lockout', 0.40).value
        self._last_scored_t = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            AprilTagDetectionArray, '/tag_detections', self._on_tags, qos)
        self.create_subscription(
            PointStamped, '/wand/tap_pixel', self._on_tap_pixel, 10)
        self.create_subscription(
            Header, '/wand/tap_unlocated', self._on_tap_unlocated, 10)
        if self._use_yolo:
            self.create_subscription(
                Detection2DArray, '/detections_output', self._on_detections, qos)
        self._pub = self.create_publisher(String, '/game/status', 10)

        self._round = 0
        self._target = None
        self._prompt_t = None
        self._results = []          # (target, outcome, reaction_s)
        self._hover = None
        self._last_seen_t = None
        self._last_seen_u = None
        self._on_target_t = None
        self._unlocated = 0
        # How stale a sighting may be before hover reports "tag not seen".
        self._hover_timeout = self.declare_parameter('hover_timeout', 0.25).value

        self.create_timer(0.1, self._tick)
        self.create_timer(0.1, self._report_hover)
        self._state = 'countdown'
        self._countdown_until = self._now() + self._start_delay
        if self._use_yolo:
            self._say(f'waiting for YOLO to find at least 2 of '
                      f'{sorted(self._yolo_classes)} '
                      f'({self._min_yolo_hits} detections each)  |  rejecting taps '
                      f'uncertain beyond {self._max_uncertainty:.0f} px')
        else:
            self._say('targets: ' + ',  '.join(
                f'{n}@{int(u)}+/-{self._tolerance[n]:.0f}' for n, u in self._targets)
                + f'  |  rejecting taps uncertain beyond {self._max_uncertainty:.0f} px'
                + f'  |  starting in {self._start_delay:.0f} s')

    def _set_targets(self, targets):
        """Install a target list and recompute per-target tolerances.

        Tolerance is half the distance to that target's NEAREST NEIGHBOUR, capped
        by max_halfwidth. A single global halfwidth taken from the tightest pair
        would make every region as strict as the worst one -- an isolated target
        450 px from anything else would still reject a tap 60 px off. Recomputed
        on every change because with YOLO the target set is discovered at runtime.
        """
        self._targets = sorted(targets, key=lambda t: t[1])
        self._tolerance = {}
        for i, (name, cu) in enumerate(self._targets):
            others = [abs(cu - ou) for j, (_, ou) in enumerate(self._targets) if j != i]
            self._tolerance[name] = min(
                self._max_hw, (min(others) / 2.0) if others else self._max_hw)

    def _on_detections(self, msg):
        """Build the target list from YOLO: class name + bbox centre column."""
        if not self._use_yolo:
            return
        changed = False
        for det in msg.detections:
            if not det.results:
                continue
            name = class_name(det.results[0].hypothesis.class_id)
            if name not in self._yolo_classes:
                continue
            u = det.bbox.center.position.x
            entry = self._seen.get(name)
            if entry is None:
                self._seen[name] = [1, u]
            else:
                entry[0] += 1
                a = self._yolo_smoothing
                entry[1] = (1.0 - a) * entry[1] + a * u
                if entry[0] == self._min_yolo_hits:
                    changed = True

        stable = [(n, v[1]) for n, v in self._seen.items()
                  if v[0] >= self._min_yolo_hits]
        # Only reinstall when the SET changes; the smoothed positions keep moving
        # and rebuilding tolerances every frame would be pointless churn.
        if changed or {n for n, _ in self._targets} != {n for n, _ in stable}:
            self._set_targets(stable)
            self._say('targets from YOLO: ' + ',  '.join(
                f'{n}@{int(u)}+/-{self._tolerance[n]:.0f}' for n, u in self._targets))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _say(self, text):
        self.get_logger().info(text)
        self._pub.publish(String(data=text))

    def _say_throttled(self, text, period):
        now = self._now()
        if now - getattr(self, '_last_throttled_t', 0.0) >= period:
            self._last_throttled_t = now
            self._say(text)

    def _match(self, u):
        """Nearest target whose own tolerance contains u, or None."""
        best, best_d = None, float('inf')
        for name, cu in self._targets:
            d = abs(u - cu)
            if d <= self._tolerance[name] and d < best_d:
                best, best_d = name, d
        return best

    def _on_tags(self, msg):
        """Record the latest tag sighting. Reporting happens on a timer.

        Doing the reporting here instead would only ever fire when the tag is
        FOUND -- and detection drops to 60-77% during motion, so a message with
        an empty detection list never reaches the loop body. The player would
        then see no feedback at all while moving, and 'wand over: pen' only once
        they had stopped and the tag was re-acquired. Waiting for that cue
        serialises move-then-tap and inflates reaction times.
        """
        for det in msg.detections:
            self._last_seen_t = self._now()
            self._last_seen_u = det.center.x
            return

    def _report_hover(self):
        """Continuous hover state, including when the tag is NOT visible."""
        if self._state != 'waiting':
            return
        fresh = (self._last_seen_t is not None
                 and self._now() - self._last_seen_t <= self._hover_timeout)
        if not fresh:
            region = '(tag not seen)'
        else:
            region = self._match(self._last_seen_u) or '(between targets)'
            if region == self._target:
                region += '  <-- ON TARGET, tap now'
        if region != self._hover:
            self._hover = region
            self._say(f'   wand over: {region}')
            # First arrival on target: split the round into move time and dwell
            # time, so a slow score can be attributed to the player moving or to
            # the player hesitating/waiting for feedback.
            if self._on_target_t is None and region.startswith(self._target):
                self._on_target_t = self._now()

    def _on_tap_pixel(self, msg):
        if self._state != 'waiting':
            return

        tap_t = stamp_to_sec(msg.header.stamp)
        u, uncertainty = msg.point.x, msg.point.z

        # A tap stamped before the prompt cannot be a response to it.
        if tap_t < self._prompt_t:
            self._say(f'   (ignoring a tap stamped {(self._prompt_t - tap_t) * 1000:.0f} '
                      'ms before the prompt)')
            return

        if self._last_scored_t is not None \
                and tap_t - self._last_scored_t < self._tap_lockout:
            self._say(f'   (ignoring a tap {(tap_t - self._last_scored_t) * 1000:.0f} '
                      'ms after the last scored one — bounce or follow-through)')
            return

        if uncertainty > self._max_uncertainty:
            self._say(f'   (ignoring a tap: position uncertain to '
                      f'{uncertainty:.0f} px — beyond any usable bound)')
            return

        # Does the uncertainty actually change the answer? Resolve the tap at its
        # nominal position and at both extremes of the error bar; if all three
        # agree, the uncertainty is irrelevant however large it looks.
        hit_region = self._match(u)
        if not (self._match(u - uncertainty) == hit_region == self._match(u + uncertainty)):
            self._say(f'   (ignoring a tap: u={u:.0f} +/-{uncertainty:.0f} px spans '
                      'more than one target — hold the wand steadier)')
            return

        reaction = tap_t - self._prompt_t

        if hit_region == self._target:
            outcome = 'HIT'
            # Split move vs dwell: a slow round is either a slow traverse or
            # hesitation once already on target. Only the second is something
            # feedback latency can be blamed for.
            if self._on_target_t is not None:
                move = self._on_target_t - self._prompt_t
                dwell = tap_t - self._on_target_t
                split = (f', move {move * 1000:.0f} ms + dwell '
                         f'{dwell * 1000:.0f} ms')
            else:
                split = ', never seen on target before the tap'
            self._say(f'   HIT  {self._target}  in {reaction * 1000:.0f} ms'
                      f'{split}  (u={u:.0f}, +/-{uncertainty:.0f} px)')
        elif hit_region is None:
            outcome = 'WRONG'
            self._say(f'   MISS — tapped between targets (u={u:.0f}), '
                      f'wanted {self._target}')
        else:
            outcome = 'WRONG'
            self._say(f'   WRONG — tapped {hit_region} (u={u:.0f}), '
                      f'wanted {self._target}')

        self._results.append((self._target, outcome, reaction))
        self._last_scored_t = tap_t
        self._next_round()

    def _on_tap_unlocated(self, msg):
        """The firmware saw a tap the camera could not place.

        Reported rather than ignored: the player DID tap, and telling them the
        camera lost the tag is actionable ('hold the wand so the tag faces the
        camera') where silence is not. Silently dropping it also inflates the
        score, because the round keeps running and the eventual reaction time
        includes this wasted attempt.
        """
        if self._state != 'waiting':
            return
        tap_t = stamp_to_sec(msg.stamp)
        if tap_t < self._prompt_t:
            return
        self._unlocated += 1
        self._say('   TAP SEEN but the camera could not see the tag then — '
                  'angle the wand so the tag faces the camera, and try again')

    def _tick(self):
        now = self._now()

        if self._state == 'countdown':
            # With YOLO the target set is discovered at runtime, so hold the
            # countdown until at least two targets exist -- a one-target game has
            # nothing to choose between, and an empty one has nothing at all.
            if len(self._targets) < 2:
                self._countdown_until = now + self._start_delay
                if self._use_yolo:
                    self._say_throttled(
                        f'still waiting: {len(self._targets)} target(s) found',
                        period=3.0)
                return
            if now >= self._countdown_until:
                self._next_round()

        elif self._state == 'waiting':
            if now - self._prompt_t > self._time_limit:
                self._say(f'   MISS — out of time ({self._time_limit:.0f} s)')
                self._results.append((self._target, 'TIMEOUT', None))
                self._next_round()

    def _next_round(self):
        if self._round >= self._rounds:
            self._summary()
            self._state = 'done'
            return

        if len(self._targets) < 2:
            # Targets vanished mid-game (YOLO lost them). Fall back to waiting
            # rather than crashing on an empty choice.
            self._state = 'countdown'
            self._countdown_until = self._now() + self._start_delay
            self._say('lost targets — waiting for at least 2 again')
            return

        self._round += 1
        # Never repeat the previous target: a repeat can be answered without
        # moving, which is not what the game is measuring.
        choices = [n for n, _ in self._targets if n != self._target] \
            or [n for n, _ in self._targets]
        self._target = random.choice(choices)
        self._prompt_t = self._now()
        self._on_target_t = None
        self._hover = None
        self._state = 'waiting'
        self._say(f'[{self._round}/{self._rounds}]  TAP THE  >>> '
                  f'{self._target.upper()} <<<')

    def _summary(self):
        hits = [r for r in self._results if r[1] == 'HIT']
        wrong = [r for r in self._results if r[1] == 'WRONG']
        timeouts = [r for r in self._results if r[1] == 'TIMEOUT']
        lines = ['', '=' * 52, f'  {len(hits)}/{len(self._results)} hits   '
                 f'{len(wrong)} wrong   {len(timeouts)} timed out']
        if self._unlocated:
            # Not a player error: taps the camera could not place. A high count
            # means the tag is not visible at the moment of contact, which is a
            # MECHANICAL problem (tag orientation/occlusion), not a tuning one.
            lines.append(f'  {self._unlocated} taps detected but not localizable '
                         '— tag not visible at contact')
        if hits:
            times = sorted(r[2] for r in hits)
            lines.append(f'  reaction: best {times[0] * 1000:.0f} ms   '
                         f'median {times[len(times) // 2] * 1000:.0f} ms   '
                         f'worst {times[-1] * 1000:.0f} ms')
        lines += ['=' * 52, '']
        self._say('\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = TapGame()
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
