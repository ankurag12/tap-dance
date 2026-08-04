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
# TWO MODES:
#   characterize (default) -- no threshold decisions, just report what the
#     signal does. Tap things, swing the wand, and compare the peaks. Use this
#     to CHOOSE a threshold rather than guessing one.
#   detect -- emit /wand/tap (std_msgs/Header, stamp = tap instant) whenever the
#     deviation crosses `threshold`, with a refractory lockout.
#
# REFRACTORY: one physical tap rings, producing several threshold crossings.
# The lockout collapses them into one event. It also sets the fastest multi-tap
# you can resolve, so it cannot be too long -- 100 ms allows ~10 taps/s, well
# above deliberate tapping.
#
# ONSET vs PEAK: the event is stamped at the FIRST crossing, not the peak.
# Onset is physically closer to the contact instant; the peak lags by a few ms.
#
# Usage:
#   ros2 run jetson_bringup tap_detector
#   ros2 run jetson_bringup tap_detector --ros-args -p mode:=detect -p threshold:=40.0

import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Header

G = 9.80665


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class TapDetector(Node):

    def __init__(self):
        super().__init__('tap_detector')

        topic = self.declare_parameter('imu_topic', '/m5stick/imu').value
        self._mode = self.declare_parameter('mode', 'characterize').value
        self._threshold = self.declare_parameter('threshold', 40.0).value
        self._refractory = self.declare_parameter('refractory_ms', 100.0).value / 1000.0
        self._period = self.declare_parameter('report_period', 2.0).value

        if self._mode not in ('characterize', 'detect'):
            raise ValueError("mode must be 'characterize' or 'detect'")

        # The firmware publishes best-effort; a reliable subscription would
        # silently receive nothing.
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, topic, self._on_imu, qos)

        self._pub = self.create_publisher(Header, '/wand/tap', 10)

        self._dev = []              # |‖a‖-g| this window
        self._stamps = deque(maxlen=4)
        self._last_tap = 0.0
        self._taps = []             # (stamp, peak) this window
        self._armed = True          # False while inside the refractory window
        self._peak_hold = 0.0
        self._peak_stamp = 0.0

        self.create_timer(self._period, self._report)
        self.get_logger().info(
            f'{topic} — mode={self._mode}'
            + (f', threshold={self._threshold:.1f} m/s^2, '
               f'refractory={self._refractory * 1000:.0f} ms'
               if self._mode == 'detect' else
               ' — tap things and swing the wand, then compare the peaks'))

    def _on_imu(self, msg):
        a = msg.linear_acceleration
        dev = abs(math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z) - G)
        t = stamp_to_sec(msg.header.stamp)

        self._dev.append(dev)
        self._stamps.append(t)

        if self._mode != 'detect':
            return

        if dev >= self._threshold:
            if self._armed:
                # Stamp the ONSET: first crossing is closest to contact.
                self._armed = False
                self._last_tap = t
                self._peak_hold = dev
                self._peak_stamp = t

                hdr = Header()
                hdr.stamp = msg.header.stamp
                hdr.frame_id = 'wand_tap'
                self._pub.publish(hdr)
            else:
                self._peak_hold = max(self._peak_hold, dev)

        # Re-arm once quiet for the refractory period, and log the completed tap.
        if not self._armed and (t - self._last_tap) >= self._refractory:
            self._armed = True
            self._taps.append((self._last_tap, self._peak_hold))

    def _report(self):
        n = len(self._dev)
        if n == 0:
            self.get_logger().warn(
                'no IMU messages — is the micro-ROS agent up and the M5Stick connected?')
            return

        rate = n / self._period
        ordered = sorted(self._dev)
        peak = ordered[-1]
        p95 = ordered[int(0.95 * (n - 1))]
        median = ordered[n // 2]

        lines = [f'--- {n} samples in {self._period:.1f} s ({rate:.0f} Hz) ---',
                 f'  |‖a‖-g|   median {median:5.1f}   p95 {p95:5.1f}   '
                 f'PEAK {peak:6.1f}  m/s^2']

        # A tap transient is ~5-20 ms. At 100 Hz that is 1-2 samples, so the
        # peak may be missed entirely; flag it rather than let it mislead.
        if rate < 150:
            lines.append(f'  NOTE: {rate:.0f} Hz gives ~{1000 / rate:.0f} ms between '
                         'samples; a tap lasts ~5-20 ms. Consider raising the '
                         'firmware rate to 200-500 Hz.')

        if self._mode == 'detect':
            lines.append(f'  taps: {len(self._taps)}'
                         + (''.join(f'\n    t={t:.3f}  peak {p:.1f}'
                                    for t, p in self._taps) if self._taps else ''))

        self.get_logger().info('\n'.join(lines))
        self._dev = []
        self._taps = []


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
