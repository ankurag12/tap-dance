# SPDX-License-Identifier: Apache-2.0
#
# Tap detector — the "WHEN" half of the tap game.
#
# The camera says WHERE the wand is; this says WHEN it hit something. A camera
# at 30 Hz cannot tell a tap from a hover (contact is millimetres over
# milliseconds, and it falls between frames), so the discrete contact event has
# to come from the IMU.
#
# SIGNAL: |‖a‖ - g|. The accelerometer always reads gravity (~9.81 m/s^2 at
# rest) and its direction changes as the wand rotates, so per-axis thresholds
# would depend on how you hold it. The magnitude deviation is
# orientation-independent: ~0 at rest, large for any real acceleration whatever
# its direction. It cannot tell you which way the tap came from -- we do not
# care.
#
# TIMESTAMP: taken from header.stamp, which the M5Stick firmware sets from
# rmw_uros_epoch_nanos() -- i.e. the JETSON's clock, via micro-ROS session sync.
# Using reception time instead would bake in WiFi jitter (~8 ms measured) and
# destroy the cross-clock alignment that lets us ask "where was the wand at the
# tap instant".
#
# WHY EXCURSIONS, NOT JUST PEAKS: peak amplitude alone does not separate a tap
# from abruptly stopping a hard swing -- both reach ~8 g. The physics difference
# is DURATION. A tap is a collision: momentum reversed against a rigid object in
# 5-20 ms. A swing is your arm applying force over a 100-300 ms stroke. So this
# reports, for every threshold crossing, the peak AND the width together. Expect
# taps high-and-narrow, swings lower-and-wide.
#
# WHY PER-AXIS MIN/MAX: the accelerometer clips PER AXIS, not on the magnitude,
# so a saturated tap spread over two axes can still produce a magnitude well
# below any obvious ceiling. Only the per-axis extremes reveal clipping -- and
# clipping would flatten every hard tap to the same value, destroying exactly
# the separation we are trying to measure.
#
# TWO MODES:
#   characterize (default) -- no detection, just report what the signal does.
#     Tap things, move as the game would make you move, and compare.
#   detect -- emit /wand/tap (std_msgs/Header, stamp = tap instant) when the
#     deviation crosses `threshold`, with a refractory lockout.
#
# ONSET vs PEAK: the event is stamped at the FIRST crossing, not the peak. Onset
# is physically closer to the contact instant; the peak lags by a few ms.
#
# Usage:
#   ros2 run jetson_bringup tap_detector
#   ros2 run jetson_bringup tap_detector --ros-args -p excursion_floor:=20.0
#   ros2 run jetson_bringup tap_detector --ros-args -p mode:=detect -p threshold:=100.0

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Header

G = 9.80665

# Full-scale limits (m/s^2) for the MPU6886's selectable ranges. A per-axis
# reading parked on one of these means the sensor is saturating.
CLIP_LEVELS = {'+/-2g': 2 * G, '+/-4g': 4 * G, '+/-8g': 8 * G, '+/-16g': 16 * G}


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class TapDetector(Node):

    def __init__(self):
        super().__init__('tap_detector')

        topic = self.declare_parameter('imu_topic', '/m5stick/imu').value
        self._mode = self.declare_parameter('mode', 'characterize').value
        self._threshold = self.declare_parameter('threshold', 100.0).value
        self._refractory = self.declare_parameter('refractory_ms', 100.0).value / 1000.0
        self._period = self.declare_parameter('report_period', 2.0).value
        # Low bar for *reporting* an excursion in characterize mode. Well below
        # any tap: the point is to catch swings too, so their width shows up.
        self._floor = self.declare_parameter('excursion_floor', 25.0).value
        self._max_show = self.declare_parameter('max_excursions_shown', 6).value

        if self._mode not in ('characterize', 'detect'):
            raise ValueError("mode must be 'characterize' or 'detect'")

        # The firmware publishes best-effort; a reliable subscription would
        # silently receive nothing.
        qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, topic, self._on_imu, qos)
        self._pub = self.create_publisher(Header, '/wand/tap', 10)

        self._reset()
        # Excursion state persists across windows so one straddling the boundary
        # is not split into two.
        self._active = False
        self._exc_peak = 0.0
        self._exc_n = 0
        self._exc_t0 = 0.0

        self._armed = True
        self._last_tap = 0.0

        self.create_timer(self._period, self._report)
        self.get_logger().info(
            f'{topic} — mode={self._mode}'
            + (f', threshold={self._threshold:.1f} m/s^2, '
               f'refractory={self._refractory * 1000:.0f} ms'
               if self._mode == 'detect' else
               f', excursion floor {self._floor:.0f} m/s^2 — tap things, and move '
               'the way the GAME would make you move, then compare peak vs width'))

    def _reset(self):
        self._dev = []
        self._axis_min = [float('inf')] * 3
        self._axis_max = [float('-inf')] * 3
        self._excursions = []     # (peak, n_samples, duration_ms)
        self._taps = []

    def _on_imu(self, msg):
        a = msg.linear_acceleration
        axes = (a.x, a.y, a.z)
        dev = abs(math.sqrt(sum(v * v for v in axes)) - G)
        t = stamp_to_sec(msg.header.stamp)

        self._dev.append(dev)
        for i, v in enumerate(axes):
            self._axis_min[i] = min(self._axis_min[i], v)
            self._axis_max[i] = max(self._axis_max[i], v)

        # --- excursion tracking: how long does the signal stay elevated? ---
        if dev >= self._floor:
            if not self._active:
                self._active = True
                self._exc_peak, self._exc_n, self._exc_t0 = dev, 1, t
            else:
                self._exc_peak = max(self._exc_peak, dev)
                self._exc_n += 1
        elif self._active:
            self._active = False
            self._excursions.append(
                (self._exc_peak, self._exc_n, (t - self._exc_t0) * 1000.0))

        if self._mode != 'detect':
            return

        if dev >= self._threshold and self._armed:
            self._armed = False
            self._last_tap = t
            hdr = Header()
            hdr.stamp = msg.header.stamp      # onset, in Jetson epoch
            hdr.frame_id = 'wand_tap'
            self._pub.publish(hdr)
            self._taps.append((t, dev))
        elif not self._armed and (t - self._last_tap) >= self._refractory:
            self._armed = True

    def _report(self):
        n = len(self._dev)
        if n == 0:
            self.get_logger().warn(
                'no IMU messages — is the micro-ROS agent up and the M5Stick connected?')
            return

        rate = n / self._period
        ordered = sorted(self._dev)
        lines = [f'--- {n} samples in {self._period:.1f} s ({rate:.0f} Hz) ---',
                 f'  |‖a‖-g|   median {ordered[n // 2]:5.1f}   '
                 f'p95 {ordered[int(0.95 * (n - 1))]:5.1f}   PEAK {ordered[-1]:6.1f}']

        # Per-axis extremes: the only view that reveals clipping.
        ax = '  '.join(f'{self._axis_min[i]:+7.1f}/{self._axis_max[i]:+7.1f}'
                       for i in range(3))
        lines.append(f'  per-axis min/max (x y z):  {ax}')

        clipped = [
            name for name, lim in CLIP_LEVELS.items()
            if any(abs(abs(v) - lim) < 0.02 * lim
                   for v in self._axis_min + self._axis_max if math.isfinite(v))]
        if clipped:
            lines.append(f'  *** SATURATING at {clipped[0]} — raise the accel range, '
                         'or every hard tap reads the same value ***')

        if self._excursions:
            lines.append(f'  excursions above {self._floor:.0f}:'
                         '        peak    width')
            for peak, ns, ms in sorted(self._excursions,
                                       key=lambda e: -e[0])[:self._max_show]:
                # A collision is 5-20 ms; a swing stroke is 100-300 ms.
                kind = 'tap-like' if ms <= 30.0 else 'swing-like'
                lines.append(f'      {peak:7.1f}   {ns:3d} samples '
                             f'({ms:5.1f} ms)   {kind}')
            if len(self._excursions) > self._max_show:
                lines.append(f'      ... and {len(self._excursions) - self._max_show} more')

        # A tap transient is ~5-20 ms. At 100 Hz that is 1-2 samples, so width
        # is barely a measurement and the peak may be missed entirely.
        if rate < 150:
            lines.append(f'  NOTE: {rate:.0f} Hz = {1000 / rate:.0f} ms/sample; a tap '
                         'lasts ~5-20 ms, so width is quantised to 1-2 samples. '
                         'Raise the firmware rate to 400 Hz to make width usable.')

        if self._mode == 'detect':
            lines.append(f'  taps emitted: {len(self._taps)}')

        self.get_logger().info('\n'.join(lines))
        self._reset()


def main(args=None):
    rclpy.init(args=args)
    node = TapDetector()
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
