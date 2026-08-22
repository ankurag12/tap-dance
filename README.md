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

| Node | Role |
|---|---|
| `tap_localizer` | Cross-clock lookup: interpolates the tag's pixel position to the tap instant |
| `tap_game` | Round loop, region matching, reaction-time scoring |
| `tag_pose_stats` | Measures AprilTag pose jitter — the accuracy baseline |
| `tap_detector` | Characterizes the tap signal; detection itself runs on-device |
| `detection_probe` | Readable view of YOLOv8 detections |
| `object_locator` | YOLO boxes + depth → 3D object positions |
| `sensor_monitor` | `diagnostic_updater` health checks on camera + IMU topics |

## Run it

```bash
# micro-ROS agent (host, persistent — different lifecycle from the ROS launch)
docker run -d --restart unless-stopped --net=host \
  --name uros_agent microros/micro-ros-agent:humble udp4 --port 8888

# firmware (needs include/secrets.h)
cd firmware/m5stick_imu && pio run -t upload

# in the Isaac ROS container
ros2 launch tap_dance realsense_apriltag.launch.py tag_size:=0.1225 width:=1280 height:=720
ros2 run tap_dance tap_localizer
ros2 run tap_dance tap_game --ros-args \
  -p target_names:='["cup","pen"]' -p target_u:='[131.0, 866.0]'
```

`target_u` is each target's horizontal pixel position — measure once by tapping
each object with `tap_localizer` running.

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
| **Game result** | **10/10 hits, 0 unlocalizable, median reaction 868 ms** |

## Status

Working end to end: 10/10 hits with no unlocalizable taps and a median reaction
time of 868 ms. Getting there took four measured fixes -- cutting a needless
300 ms wait in the tap lookup, loosening the staleness bound once it was clear
only horizontal position mattered, adding a game-level lockout so a tap's
follow-through stopped scoring the next round, and moving AprilTag to the IR
imager with a pinned exposure.

The tag-visibility problem is solved: detection in motion went from
~50% to 100% by moving AprilTag onto the D456's global-shutter IR imager with a
pinned 4 ms exposure. Two effects were masking each other — a rolling shutter
shears a moving tag (which exposure cannot fix), while IR auto-exposure picks a
long, blurring exposure because ambient near-IR is dim with the projector off.

That last point is also the configuration's main constraint: the projector has to
stay off, so the IR imagers run passive and need an actual near-IR source.
Daylight through a window is enough; LED room lighting is not, and an IR
illuminator or a halogen lamp would be the fix for evening use.

Next: replace hand-configured target positions with **TensorRT YOLOv8** detections
so the game names real objects with no setup.

Scope and milestones: [`docs/capstone-tap-game.md`](docs/capstone-tap-game.md).
Every command for running and debugging: [`docs/runbook.md`](docs/runbook.md).
