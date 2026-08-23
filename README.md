# tap-dance

Whack-a-mole on real desk objects. The game names one — *"tap the CUP"* — and you
bring a tagged wand over it and **tap**. It scores your reaction time.

The IMU says *when*. The camera says *where*. Neither can answer alone:

| | Camera @30 Hz | IMU @500 Hz |
|---|---|---|
| Is the wand over the cup? | **yes** | no |
| Did it touch, or hover 2 cm above? | no — sub-cm, sub-frame | **yes** |
| Exactly when? | ±16 ms | **~1 ms** |
| Three taps in 600 ms? | no | **trivially** |

Contact is millimetres over ~10 ms, invisible at 30 Hz. Double-integrated
accelerometer position diverges in under a second, so the IMU cannot say *which*
object. Discrete event from the IMU, spatial context from the camera, joined
across two clock domains.

## Hardware

| | |
|---|---|
| Compute | Jetson Orin Nano — JetPack 6.2, ROS 2 Humble, Isaac ROS 3.2 |
| Camera | Intel RealSense D456, webcam-style on the desk |
| Wand | M5StickC Plus2 (ESP32 + MPU6886) + AprilTag, micro-ROS over WiFi |

## Pipeline

```
  M5StickC Plus2 (wand)                     JETSON
   ┌────────────────────┐
   │ AprilTag           │◀── sees ──  D456 ─▶ rectify ─▶ AprilTag ─▶ tag centre (px)
   │ MPU6886 @500 Hz    │                                              │
   │ detects taps       │                                     ┌────────┴────────┐
   └─────────┬──────────┘                                     │  tap_localizer  │
             │ micro-ROS / WiFi                               │  where was the  │
             │ (stamped in the Jetson's epoch)                │  tag AT the tap │
             ├── /m5stick/imu   100 Hz, diagnostics ────────▶ │  stamp?         │
             └── /wand/tap      one msg per contact ────────▶ └────────┬────────┘
                                                                       │
                                                            /wand/tap_pixel
                                                                       │
                                                                  tap_game
```

## Nodes

Runtime — `tap_dance/`:

| Node | Role |
|---|---|
| `tap_localizer` | Cross-clock lookup: interpolates the tag's pixel position to the tap instant |
| `tap_game` | Round loop, target matching, reaction-time scoring |
| `targets` | Shared library: YOLO→image scaling, target discovery, region bounds, matching |
| `sensor_monitor` | `diagnostic_updater` health checks on camera + IMU topics |

Debug tools — `tap_dance/debug/`:

| Tool | Answers |
|---|---|
| `hover_probe` | Where is the tag, which object is it over, what tolerance does each get? |
| `tag_pose_stats` | How much does the tag's pose jitter? |
| `tap_detector` | Do taps separate from swings, and at what threshold? |
| `detection_probe` | What is YOLO seeing, across all 80 COCO classes? |
| `object_locator` | YOLO boxes + depth → 3D positions (unused by the game) |

`targets.py` sits in the runtime package even though the debug tools use it:
dependencies point from `debug/` toward the runtime, never the reverse, so
`hover_probe` cannot disagree with the game about which object a position matches.

## Run it

```bash
# micro-ROS agent — persistent, on the host. Separate from the ROS launch on
# purpose: the wand connects to it whenever it powers on, game or no game.
docker run -d --restart unless-stopped --net=host \
  --name uros_agent microros/micro-ros-agent:humble udp4 --port 8888

# firmware (needs include/secrets.h)
cd firmware/m5stick_imu && pio run -t upload
```

Then, in the Isaac ROS container, one command:

```bash
ros2 launch tap_dance tap_game.launch.py tag_size:=0.1225 \
  target_names:='["cup","pen"]' target_u:='[600.0, 1000.0]'
```

Or let YOLOv8 find and name the objects instead of measuring them:

```bash
ros2 launch tap_dance tap_game.launch.py tag_size:=0.1225 use_yolo:=true
```

`target_u` is each target's horizontal pixel position. Measure it once by hovering
the wand over each object and reading `tag now at u=` from `tap_localizer`, and
re-measure after changing `sensor:` — the imagers are physically offset, so the
same object sits at a different column in each.

Measurement and debugging tools (`tag_pose_stats`, `tap_detector`,
`detection_probe`) stay separate from the game launch; see
[`docs/runbook.md`](docs/runbook.md).

## Design decisions

**Detection runs on the ESP32.** Streaming fast enough to catch a 5–20 ms transient
exhausted the ESP32's UDP buffers (`ENOMEM`) and dropped ~40% of samples — and a
dropped sample can *be* the tap peak. Detecting at the source costs no bandwidth,
loses nothing, and beats any streaming rate the link allows.

**Association keys on horizontal position only.** Taps land on the tops of objects,
so the tag sits ~10 cm above the contact point and never falls inside an object's
bounding box — but a vertical tap preserves the tag's `u`. This removes depth
alignment, intrinsics, deprojection and the tag→tip extrinsic entirely.

**Reaction time uses the tap's stamp, not its arrival.** WiFi arrival jitter reaches
~90 ms. The prompt time is on the Jetson's clock and the tap stamp on the
M5Stick's, so every score is a cross-clock measurement — 100 ms of sync drift would
make every score wrong by 100 ms.

**QoS is asymmetric.** Best-effort for the IMU stream, where a retransmit stalls
newer samples and one lost sample is invisible. Reliable for `/wand/tap`, where
each message is a game event and a loss means the hit does not register.

## Measured

| | |
|---|---|
| On-device sample rate | 485 Hz of 500 requested (rclc timer granularity) |
| Packet loss after decimation | 0 — 485/5 sampled = 97 Hz received |
| Tap vs. hard-swing separation | peak 100–120 vs 40–75 m/s² → threshold 90 |
| AprilTag out-of-plane jitter | 1.9–5.7° (12 cm tag), 5.7–11.3° (6 cm) |
| Tag detection, wand stationary | 97–100% of frames |
| Tag detection in motion — auto exposure | 38–70% |
| Tag detection in motion — 4 ms, dim room | 50–70% |
| **Tag detection in motion — 2 ms, lit room** | **~100%** (colour or IR) |
| Tap → score latency | ~75 ms, from ~330 ms |
| **Game, hand-configured targets** | **10/10 hits, median reaction 868 ms** |
| **Game, YOLO-discovered targets** | **8/10 hits, median reaction 1203 ms** |

## Status

Working end to end, both ways of choosing targets: 10/10 hits at a median 868 ms
with hand-measured positions, 8/10 at 1203 ms with three objects discovered by
YOLOv8.

Four measured fixes got it there. Pinning the camera exposure to 2 ms took tag
detection during the reach from ~50% to ~100% — auto-exposure optimises brightness
and picks a long exposure that blurs the tag exactly when it matters. Cutting a
needless 300 ms wait in the tap lookup dropped tap-to-score latency from ~330 ms to
~50 ms. Loosening the staleness bound stopped good taps being discarded, once it was
clear only horizontal position mattered. And a game-level lockout stopped a tap's
follow-through scoring the next round.

The one operating condition to know: a 2 ms exposure needs a well-lit room. In poor
light the image underexposes and detection falls — the fixes being more light, a
longer exposure, or the global-shutter IR imager (`sensor:=infra1`), which needs a
near-IR source of its own.

YOLOv8 target detection (`use_yolo:=true`) runs its TensorRT engine in the same
container on the same rectified image AprilTag uses, so bbox centres and tag centres
are directly comparable — no depth, no intrinsics, no cross-sensor registration. A
class must be detected repeatedly before it becomes a target, so one false positive
cannot inject one mid-game.

Why it is built this way, and what the measurements said: [`docs/design.md`](docs/design.md).
Every command for running and debugging: [`docs/runbook.md`](docs/runbook.md).
