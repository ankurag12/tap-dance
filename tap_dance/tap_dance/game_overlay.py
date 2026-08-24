# SPDX-License-Identifier: Apache-2.0
#
# game_overlay — the on-screen display, rendered into an image topic.
#
#   /image_rect  ─┐
#                 ├─▶  brighten + draw HUD  ─▶  /game/overlay  (view in Foxglove)
#   /game/hud    ─┘
#
# WHY AN IMAGE TOPIC AND NOT A WINDOW
# The Jetson runs headless -- everything is watched through Foxglove from another
# machine -- so a pygame or OpenCV window would open on a display nobody is looking
# at, and there is no ROS on the viewing machine to run one there. Publishing the
# rendered frame costs nothing extra: Foxglove already has an Image panel open.
#
# WHY BRIGHTEN HERE
# The camera exposure is pinned to ~2 ms because that is what stops the tag
# blurring during the reach, and a 2 ms frame looks dark. Rather than compromise
# the detector for the sake of the picture, expose for the DETECTOR and grade for
# the VIEWER: this node processes its own copy and the detector never sees it.
#
# Three stages, because gamma alone looks washed out -- it lifts midtones by
# flattening the tone curve, which costs contrast and desaturates colour:
#   1. gamma      global lift, so the frame is not simply dark
#   2. CLAHE      adaptive local contrast on L, which is what restores "punch"
#                 without crushing the highlights a global stretch would
#   3. chroma     scale a/b away from neutral, since a short exposure lands in the
#                 low-saturation part of the sensor's response
# Stages 2 and 3 share one BGR->LAB->BGR round trip: CLAHE acts on L while
# saturation is just a scale on a and b, so there is no separate HSV conversion.
#
# WHY THE HUD IS TIME-ALIGNED TO THE FRAME
# The text and the picture reach the viewer by very different routes. A frame goes
# capture -> publish -> JPEG -> here -> bridge -> WiFi -> Foxglove, a few hundred
# milliseconds; the HUD arrives from the game in a few tens. Drawing the newest text
# onto the frame in hand therefore shows the score BEFORE the recorded picture shows
# the tap -- which looks like the game jumping the gun, when only the compositing is
# wrong.
#
# So HUD states are kept with the time they arrived, and each frame is drawn with the
# state that was current at ITS capture time. Text and picture then agree no matter
# how much delay sits downstream.
#
# WHAT IS DELIBERATELY NOT DRAWN
# No detection boxes, no tracked-tag marker. Showing them would tell the player
# exactly what the system thinks each object is and where it believes the wand is
# -- which is the game. The HUD carries only what a player should see: the prompt,
# the outcome, and the score. Use hover_probe when you want to see the internals.
#
# SUBSCRIBES TO THE COMPRESSED STREAM, and that is not cosmetic. A Python node
# receiving raw 720p receives ~2.7 MB per message; at 30 Hz that is ~83 MB/s being
# deserialised into Python objects, which measurably starved the perception
# pipeline -- camera throughput fell from ~30 Hz to 13-18 Hz. Longer frame gaps
# meant staler tag poses at the tap instant, which the game then had to reject as
# positionally ambiguous, so a real tap did not score and had to be repeated.
#
# JPEG is roughly 10x smaller on the wire even on a noisy frame (more on a real
# scene) and decodes in a few ms, and the display does not need the rectified image
# at all -- only the detector does. Rendering is still throttled to 10 Hz at 960 px.
#
# Usage:
#   ros2 run tap_dance game_overlay
#   ros2 run tap_dance game_overlay --ros-args -p gamma:=2.2 -p rate:=15.0

from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


class GameOverlay(Node):

    def __init__(self):
        super().__init__('game_overlay')

        # Default is the camera's COMPRESSED colour stream: cheap to receive, and
        # the display has no use for rectification.
        image_topic = self.declare_parameter(
            'image_topic', '/image_raw/compressed').value
        self._compressed = self.declare_parameter('compressed', True).value
        self._rate = self.declare_parameter('rate', 10.0).value
        self._width = self.declare_parameter('output_width', 960).value
        # >1 brightens midtones without clipping highlights, which is what a
        # short-exposure frame needs; a plain gain multiply would blow out the
        # bright areas first. Lower than it once was because CLAHE now carries part
        # of the lift, and stacking both over-brightened.
        self._gamma = self.declare_parameter('gamma', 1.8).value
        # CLAHE clip limit. 0 disables it. Higher is punchier and noisier -- the
        # limit is what stops it amplifying sensor noise in flat regions, which is
        # exactly the failure mode of plain histogram equalisation on a dark frame.
        self._clahe_clip = self.declare_parameter('contrast', 2.0).value
        # Chroma scale about neutral. 1.0 leaves colour alone.
        self._saturation = self.declare_parameter('saturation', 1.4).value

        self._clahe = (cv2.createCLAHE(clipLimit=self._clahe_clip,
                                       tileGridSize=(8, 8))
                       if self._clahe_clip > 0 else None)
        # Draw the HUD that was current when the frame was CAPTURED, not the
        # newest one -- see the note at the top. false gives snappier on-screen
        # feedback at the cost of text leading the picture.
        self._sync_hud = self.declare_parameter('sync_hud_to_frame', True).value

        self._bridge = CvBridge()
        self._topic = image_topic
        self._frame = None
        self._hud = ''
        self._hud_log = deque(maxlen=64)   # (arrival_time, text)
        self._age = None
        self._age_warned = False
        self._lut = self._build_lut(self._gamma)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        if self._compressed:
            self.create_subscription(CompressedImage, image_topic,
                                     self._on_image, qos)
        else:
            self.create_subscription(Image, image_topic, self._on_image, qos)
        self.create_subscription(String, '/game/hud', self._on_hud, 10)
        self._pub = self.create_publisher(Image, '/game/overlay', 1)

        self.create_timer(1.0 / self._rate, self._render)
        # A wrong topic name is otherwise silent: the node simply renders nothing.
        self.create_timer(5.0, self._check_input)
        self.get_logger().info(
            f'{image_topic} -> /game/overlay at {self._rate:.0f} Hz, {self._width} '
            f'px wide  |  display grading: gamma {self._gamma:.1f}, contrast '
            f'{self._clahe_clip:.1f}, saturation {self._saturation:.1f}')

    @staticmethod
    def _build_lut(gamma):
        # Precomputed 256-entry curve: applying pow() per pixel per frame in
        # Python would dominate the frame budget.
        inv = 1.0 / max(gamma, 0.01)
        return np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                        dtype=np.uint8)

    def _grade(self, img):
        """Display-only grading: gamma lift, local contrast, chroma boost."""
        img = cv2.LUT(img, self._lut)
        if self._clahe is None and self._saturation == 1.0:
            return img

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        if self._clahe is not None:
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        if self._saturation != 1.0:
            # a and b are signed chroma stored offset by 128, so scaling about 128
            # pushes colours away from neutral grey. int16 first: uint8 would wrap
            # instead of clipping on the way out.
            for ch in (1, 2):
                v = lab[:, :, ch].astype(np.int16)
                lab[:, :, ch] = np.clip((v - 128) * self._saturation + 128,
                                        0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _check_input(self):
        if self._frame is not None and self._age is not None:
            # How far behind reality the displayed picture is, before the bridge and
            # the network add their own delay. Worth knowing when recording.
            self.get_logger().info(
                f'frame age at render: {self._age * 1000:.0f} ms'
                + ('' if self._sync_hud else '  (HUD not time-aligned)'))
        if self._frame is None:
            self.get_logger().warn(
                f'no frames on {self._topic} — check the topic exists '
                "(`ros2 topic list | grep image`). For a raw stream pass "
                'compressed:=false and the uncompressed topic name.')

    def _on_image(self, msg):
        # Keep only the newest frame; the render timer decides when to use one, so
        # queuing would just add latency.
        self._frame = msg

    def _on_hud(self, msg):
        self._hud = msg.data
        self._hud_log.append((self._now(), msg.data))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _hud_for_frame(self, header):
        """The HUD text that was current when this frame was captured."""
        if not self._sync_hud or not self._hud_log:
            return self._hud

        stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        age = self._now() - stamp
        # If the camera's stamps are not on this host's clock the age is nonsense,
        # so fall back rather than showing something arbitrary.
        if not 0.0 <= age <= 2.0:
            if not self._age_warned:
                self._age_warned = True
                self.get_logger().warn(
                    f'frame age {age:.2f} s is implausible — camera stamps may not '
                    'be on the host clock; showing the newest HUD instead')
            return self._hud
        self._age = age

        cutoff = self._now() - age
        text = self._hud_log[0][1]
        for t, value in self._hud_log:
            if t <= cutoff:
                text = value
            else:
                break
        return text

    def _render(self):
        if self._frame is None:
            return
        try:
            if self._compressed:
                img = self._bridge.compressed_imgmsg_to_cv2(
                    self._frame, desired_encoding='bgr8')
            else:
                img = self._bridge.imgmsg_to_cv2(self._frame,
                                                 desired_encoding='bgr8')
        except Exception as exc:                                # noqa: BLE001
            self.get_logger().warn(f'cannot convert image: {exc}',
                                   throttle_duration_sec=5.0)
            return

        h, w = img.shape[:2]
        if w != self._width:
            img = cv2.resize(img, (self._width, int(h * self._width / w)),
                             interpolation=cv2.INTER_AREA)
        img = self._grade(img)

        hud = self._hud_for_frame(self._frame.header)
        if hud:
            self._draw_hud(img, hud.split('\n'))

        out = self._bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out.header = self._frame.header
        self._pub.publish(out)

    def _draw_hud(self, img, lines):
        w = img.shape[1]
        scales = [1.4, 0.9, 0.7][:len(lines)]
        pad, gap = 14, 10
        heights = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)[0][1]
                   for t, sc in zip(lines, scales)]
        band = pad * 2 + sum(heights) + gap * (len(lines) - 1)

        # Darken behind the text rather than filling flat, so the frame stays
        # visible underneath and the band does not read as a separate window.
        strip = img[0:band, 0:w]
        cv2.addWeighted(strip, 0.25, np.zeros_like(strip), 0.75, 0, strip)

        y = pad
        for text, scale, th in zip(lines, scales, heights):
            y += th
            cv2.putText(img, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), 2, cv2.LINE_AA)
            y += gap


def main(args=None):
    rclpy.init(args=args)
    node = GameOverlay()
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
