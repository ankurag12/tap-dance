# SPDX-License-Identifier: Apache-2.0
#
# Quantify AprilTag pose jitter — the M0 baseline for the pointing capstone.
#
# Pointing error at range IS orientation error, so "the pose looks wobbly in
# Foxglove" has to become a number before any fusion can be said to improve it.
# Hold the tag STILL and this node reports, per window:
#
#   * detection rate + apparent tag size in PIXELS (measured from the reported
#     corners, not estimated from a FOV formula)
#   * range and OBLIQUITY (0 deg = tag square to the camera, 90 deg = edge-on)
#   * position noise, and angular noise SPLIT INTO OUT-OF-PLANE vs IN-PLANE
#   * the implied pointing error at a chosen range, in cm
#
# Why the out-of-plane / in-plane split is the interesting part: for a planar
# target viewed face-on, projected width goes as cos(theta), whose derivative
# at theta=0 is ZERO. Out-of-plane rotation is therefore near-unobservable when
# square to the camera (and this is also where planar-PnP flip ambiguity lives),
# while in-plane rotation about the tag normal stays well observed. Expect the
# out-of-plane numbers to dominate, and to IMPROVE as you tilt the tag away
# from face-on. Sweep the tilt and read the curve off `obliq`.
#
# Rotation noise is measured as deviation from the window's MEAN orientation,
# expressed in the TAG's own frame, so the x/y components are out-of-plane tilt
# and the z component is in-plane roll.
#
# Usage:
#   ros2 run jetson_bringup tag_pose_stats
#   ros2 run jetson_bringup tag_pose_stats --ros-args -p period:=5.0
#   ros2 run jetson_bringup tag_pose_stats --ros-args -p tag_id:=0 -p point_range:=1.5

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray


def quat_to_matrix(q):
    """q = (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(a, b):
    """Hamilton product of (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def mean_quat(quats):
    """
    Average orientation. Quaternions are sign-ambiguous (q and -q are the same
    rotation), so align every sample to the first before summing; for the small
    spreads we are measuring, the normalized sum is an accurate mean.
    """
    ref = quats[0]
    aligned = np.array([q if np.dot(q, ref) >= 0.0 else -q for q in quats])
    m = aligned.mean(axis=0)
    return m / np.linalg.norm(m)


class TagPoseStats(Node):

    def __init__(self):
        super().__init__('tag_pose_stats')

        topic = self.declare_parameter('topic', '/tag_detections').value
        self._period = self.declare_parameter('period', 3.0).value
        # -1 = whichever tag is in view; set an id when several are visible.
        self._tag_id = self.declare_parameter('tag_id', -1).value
        # Range at which to express the angular noise as a lateral distance.
        self._point_range = self.declare_parameter('point_range', 1.5).value

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(AprilTagDetectionArray, topic, self._on_tags, qos)

        self._reset()
        self.create_timer(self._period, self._report)
        self.get_logger().info(
            f'Measuring {topic} every {self._period:.1f} s — HOLD THE TAG STILL. '
            f'Angular noise reported as pointing error at {self._point_range:.2f} m.')

    def _reset(self):
        self._pos = []      # (x, y, z) in camera frame
        self._quat = []     # (x, y, z, w)
        self._px = []       # apparent tag edge length, pixels
        self._obliq = []    # angle between tag normal and line of sight, deg
        self._windows = 0

    def _on_tags(self, msg):
        for det in msg.detections:
            if self._tag_id >= 0 and det.id != self._tag_id:
                continue

            p = det.pose.pose.pose.position
            o = det.pose.pose.pose.orientation
            pos = np.array([p.x, p.y, p.z])
            quat = np.array([o.x, o.y, o.z, o.w])

            # Apparent size: mean of the four edge lengths, straight from the
            # detector's corner pixels. This is the real driver of pose noise.
            c = [(pt.x, pt.y) for pt in det.corners]
            edges = [math.dist(c[i], c[(i + 1) % 4]) for i in range(4)]

            # Obliquity: angle between the tag's normal (its local +Z) and the
            # line of sight to it. abs() on the dot product makes this immune to
            # whether the normal points toward or away from the camera.
            normal = quat_to_matrix(quat) @ np.array([0.0, 0.0, 1.0])
            los = pos / np.linalg.norm(pos)
            obliq = math.degrees(math.acos(min(1.0, abs(float(np.dot(normal, los))))))

            self._pos.append(pos)
            self._quat.append(quat)
            self._px.append(float(np.mean(edges)))
            self._obliq.append(obliq)

    def _report(self):
        n = len(self._pos)
        if n < 5:
            self.get_logger().warn(
                f'only {n} detections this window — tag not seen (check tag_size, '
                'lighting, distance, or that the tag is in frame)')
            self._reset()
            return

        pos = np.array(self._pos)
        quats = np.array(self._quat)

        rng = float(np.linalg.norm(pos.mean(axis=0)))
        px = float(np.mean(self._px))
        obliq = float(np.mean(self._obliq))

        # --- position noise (mm) ---
        pos_sd = pos.std(axis=0) * 1000.0
        # Peak-to-peak exposes actual hand movement, which would otherwise be
        # misread as sensor noise. If span >> sd, the tag was not held still.
        pos_span = (pos.max(axis=0) - pos.min(axis=0)) * 1000.0

        # --- angular noise, in the TAG's own frame ---
        qm = mean_quat(quats)
        qm_inv = np.array([-qm[0], -qm[1], -qm[2], qm[3]])
        rotvec = []
        for q in quats:
            qs = q if np.dot(q, qm) >= 0.0 else -q
            qe = quat_mul(qm_inv, qs)
            # Small-angle: rotation vector ~ 2 * vector part (rad).
            rotvec.append(2.0 * qe[:3])
        rotvec = np.degrees(np.array(rotvec))

        sd_x, sd_y, sd_z = rotvec.std(axis=0)
        out_of_plane = math.hypot(sd_x, sd_y)   # tilt about in-plane axes
        in_plane = sd_z                          # roll about the tag normal
        total = math.sqrt(out_of_plane ** 2 + in_plane ** 2)

        # Angular noise -> lateral pointing error at the chosen range.
        point_err_cm = math.tan(math.radians(out_of_plane)) * self._point_range * 100.0

        self.get_logger().info(
            f'\n--- {n} samples in {self._period:.1f} s ({n / self._period:.0f} Hz) ---\n'
            f'  geometry   range {rng:.2f} m   tag {px:.0f} px   '
            f'obliq {obliq:.0f} deg (0=face-on)\n'
            f'  position   sd [{pos_sd[0]:.1f} {pos_sd[1]:.1f} {pos_sd[2]:.1f}] mm'
            f'   span [{pos_span[0]:.0f} {pos_span[1]:.0f} {pos_span[2]:.0f}] mm\n'
            f'  rotation   OUT-of-plane {out_of_plane:.2f} deg'
            f'   in-plane {in_plane:.2f} deg   total {total:.2f} deg\n'
            f'  => pointing error at {self._point_range:.2f} m: '
            f'{point_err_cm:.1f} cm  (from out-of-plane noise)')

        self._reset()


def main(args=None):
    rclpy.init(args=args)
    node = TagPoseStats()
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
