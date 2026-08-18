#!/usr/bin/env python3
# Jetson-side sensor health monitor.
#
# Uses diagnostic_updater to watch our real sensor topics and publish
# diagnostic_msgs/DiagnosticArray on /diagnostics:
#   * FrequencyStatus  -> is the topic hitting its expected Hz? (dead sensor / USB drops)
#   * TimeStampStatus  -> is (now - header.stamp) within tolerance? (stale / desynced data)
#
# View in Foxglove's "Diagnostics" panel (reads /diagnostics directly). Unplug the
# D456 or drop the M5Stick's WiFi and watch the status flip OK -> STALE/ERROR.

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from diagnostic_updater import (
    Updater, FrequencyStatusParam, TimeStampStatusParam, TopicDiagnostic)
from sensor_msgs.msg import Image, Imu


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class SensorMonitor(Node):
    def __init__(self):
        super().__init__('sensor_monitor')

        # The Updater publishes /diagnostics periodically (~1 Hz) and runs each
        # registered check. setHardwareID tags every status with the platform.
        self.updater = Updater(self)
        self.updater.setHardwareID('jetson-orin: d456 + m5stick')

        # Sensor streams are best-effort; a best-effort subscriber is compatible
        # with best-effort AND reliable publishers.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # --- D456 IR camera: expect ~15-30 Hz (varies with exposure/light) ---
        # FrequencyStatusParam({min,max}, tolerance, window): OK if measured Hz is
        # within [min*(1-tol), max*(1+tol)] over the last `window` updates.
        # TimeStampStatusParam(min_accept, max_accept) in seconds of (now - stamp):
        # here warn if stamps are >0.5 s stale (or in the future).
        self.cam_diag = TopicDiagnostic(
            '/camera/infra1/image_rect_raw', self.updater,
            FrequencyStatusParam({'min': 10.0, 'max': 40.0}, 0.1, 5),
            TimeStampStatusParam(-1.0, 0.5))
        self.create_subscription(
            Image, '/camera/infra1/image_rect_raw', self.cam_cb, qos)

        # --- M5Stick IMU: expect ~100 Hz; tighter staleness bound ---
        self.imu_diag = TopicDiagnostic(
            '/m5stick/imu', self.updater,
            FrequencyStatusParam({'min': 80.0, 'max': 120.0}, 0.1, 5),
            TimeStampStatusParam(-1.0, 0.2))
        self.create_subscription(Imu, '/m5stick/imu', self.imu_cb, qos)

    # On each message, "tick" the topic's diagnostic with its header stamp.
    # No messages -> FrequencyStatus measures 0 Hz -> status goes ERROR/STALE,
    # which is exactly the health signal we want when a sensor dies.
    def cam_cb(self, msg):
        self.cam_diag.tick(stamp_to_sec(msg.header.stamp))

    def imu_cb(self, msg):
        self.imu_diag.tick(stamp_to_sec(msg.header.stamp))


def main():
    rclpy.init()
    node = SensorMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
