# Runbook

Every command needed to run or debug this project. Kept in the repo because the
Isaac ROS dev container runs with `docker run --rm`, so shell history does not
survive a restart.

`cat docs/runbook.md` from inside the container.

Prompts below distinguish the two shells:

- **host** — `hunter@orin-nano:~$`
- **container** — `admin@orin-nano:/workspaces/isaac_ros-dev$`

---

## 0. Bring-up

**host** — start the container:

```bash
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh
```

**host** — micro-ROS agent (persistent, outside the ROS launch: different lifecycle):

```bash
docker run -d --restart unless-stopped --net=host \
  --name uros_agent microros/micro-ros-agent:humble udp4 --port 8888
```

Already created once? `docker start uros_agent`. Watch it: `docker logs -f uros_agent`.

**container** — build (`--packages-up-to` because `isaac_ros_apriltag` is built
from source in this workspace and `tap_dance` depends on it):

```bash
cd /workspaces/isaac_ros-dev && colcon build --packages-up-to tap_dance && source install/setup.bash
```

`source install/setup.bash` is needed in every new shell.

---

## 1. Play the game

Three **container** shells.

```bash
# 1 — camera + AprilTag (defaults are the measured-best config:
#     IR global shutter, 4 ms pinned exposure -> 100% detection in motion)
ros2 launch tap_dance realsense_apriltag.launch.py tag_size:=0.1225

# 2 — cross-clock lookup (tap instant -> tag pixel position)
ros2 run tap_dance tap_localizer

# 3 — the game
ros2 run tap_dance tap_game --ros-args \
  -p target_names:='["cup","pen"]' -p target_u:='[131.0, 866.0]'
```

### Measuring `target_u`

`target_u` is each target's horizontal pixel position. Run `tap_localizer`, hover
the wand over each object in turn, and read `tag now at u=...` from its status
line — no tapping needed.

**Re-measure whenever you change `sensor:`.** The IR and colour lenses are
physically offset and have slightly different fields of view, so the same object
sits at a different column in each.

Two placement notes:

- With N targets the boundary between two of them is the MIDPOINT of their `u`
  values, so regions are only equal if the objects are spread evenly. A target
  near the frame edge wastes half its region outside the image.
- Keep objects well inside the frame. Near the edges the tag risks leaving view
  entirely during the reach.

Useful `tap_game` parameters:

```bash
  -p rounds:=10              # rounds per game
  -p time_limit:=6.0         # seconds allowed per round
  -p tap_lockout:=0.40       # ignore taps this soon after a scored one (bounce)
  -p max_halfwidth:=400.0    # cap on per-target region half-width, px
  -p max_uncertainty:=250.0  # absolute backstop on tap position uncertainty, px
```

---

## 2. Measure

**Tag detection rate and pose latency** — the `wand tag found N%` line is the key
health number. 100% in motion with the default IR + 4 ms config; drop to colour or
to auto-exposure and it falls to 38–70%.

```bash
ros2 run tap_dance tap_localizer
```

**AprilTag pose jitter** — the pointing-accuracy baseline. Hold the tag still:

```bash
ros2 run tap_dance tag_pose_stats --ros-args -p period:=5.0 -p point_range:=0.4
```

**Tap signal characterization** — peak and pulse width of taps vs swings. Detection
itself runs on the M5Stick; this reads the 100 Hz diagnostic stream:

```bash
ros2 run tap_dance tap_detector --ros-args -p report_period:=3.0
```

**Raw topic checks:**

```bash
ros2 topic hz /m5stick/imu          # expect ~97 Hz
ros2 topic hz /tag_detections       # expect ~30 Hz
ros2 topic echo /wand/tap           # one message per contact
ros2 topic echo /wand/tap_pixel     # x=u, y=v, z=u uncertainty (px)
ros2 topic list | grep -E "tap|m5stick|infra|color"
```

---

## 3. Exposure / motion blur — SOLVED, kept for reference

Detection in motion, waving at game pace:

| Sensor | Exposure | Found % |
|---|---|---|
| colour | auto | 38–70% |
| colour | ~4 ms | 50–70% |
| IR | auto | 50% (dim ambient IR -> AE picks a long exposure) |
| **IR** | **2–4 ms** | **100%** |

**IR needs an IR light source.** 100% was measured with daylight through an open
window. The projector must stay off (its dots cover the tag), so the imagers run
passive, and ambient near-IR indoors is scarce — LED lighting emits almost none.
At night, expect the IR image to darken and detection to fall. Options, best
first: a cheap IR illuminator/floodlight aimed at the play area; an incandescent
or halogen lamp (rich in near-IR, unlike LED); a longer exposure (trades blur
back in); or fall back to `sensor:=color`.

Both effects are real and each masked the other: exposure causes blur, and the
colour imager's rolling shutter shears a moving tag regardless of exposure. The
launch defaults are now IR at 4 ms.

```bash
# baseline: auto-exposure
ros2 launch tap_dance realsense_apriltag.launch.py tag_size:=0.1225

# pinned exposure; units are 100 us, so 80 = 8 ms
ros2 launch tap_dance realsense_apriltag.launch.py tag_size:=0.1225 \
  auto_exposure:=false exposure:=40 gain:=200
ros2 launch tap_dance realsense_apriltag.launch.py tag_size:=0.1225 \
  auto_exposure:=false exposure:=80 gain:=160
ros2 launch tap_dance realsense_apriltag.launch.py tag_size:=0.1225 \
  auto_exposure:=false exposure:=150 gain:=120
```

Read `wand tag found N%` from `tap_localizer` for each while waving the wand at
game pace.

---

## 4. IR / global shutter (the motion fix)

The colour imager is rolling shutter, so motion SHEARS the tag and no exposure
setting can fix it. The IR imagers are global shutter. Compare:

```bash
# colour baseline (rolling shutter)
ros2 launch tap_dance realsense_apriltag.launch.py sensor:=color tag_size:=0.1225

# IR, same resolution and rate: only the shutter changes
ros2 launch tap_dance realsense_apriltag.launch.py sensor:=infra1 tag_size:=0.1225

# IR at 60 FPS: halves inter-frame motion, but the tag shrinks to ~27 px
ros2 launch tap_dance realsense_apriltag.launch.py sensor:=infra1 tag_size:=0.1225 \
  width:=848 height:=480 fps:=60
```

Read `wand tag found N%` from `tap_localizer` for each while waving at game pace.
The IR path drops RectifyNode (IR is factory rectified) but adds an
ImageFormatConverterNode, because cuAprilTags rejects mono8 — "only 'rgb8' or
'bgr8' image input". Same node count; IR is worth it for the shutter, not for
pipeline length.

`exposure` is in MICROSECONDS for both sensors and converted internally --
librealsense wants 100 us units for colour and microseconds for IR.

### First-time IR sanity check

The D456's colour imager is **rolling shutter** (motion shears the tag, which
exposure cannot fix); the IR imagers are **global shutter**. Before switching the
pipeline to IR, confirm the tag is even visible there — some inks are near-IR
transparent.

**container** — camera on IR only, projector off (its dot pattern would cover the tag):

```bash
ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -p enable_color:=false \
  -p enable_depth:=false \
  -p enable_infra1:=true \
  -p depth_module.profile:=1280x720x30 \
  -p depth_module.emitter_enabled:=0 \
  -p depth_module.enable_auto_exposure:=true
```

**container** — find the topic name, then view it:

```bash
ros2 topic list | grep infra
ros2 run foxglove_bridge foxglove_bridge
```

Then on the Mac: Foxglove → `ws://orin-nano.local:8765` → Image panel on the
infra1 topic. The tag must read as crisp black-on-white. If it looks blank or
washed out, the ink is IR-transparent and the IR path is not viable.

---

## 5. Firmware

**host / Mac** — needs `include/secrets.h` (WIFI_SSID, WIFI_PASS, AGENT_IP =
Jetson's LAN address, AGENT_PORT 8888):

```bash
cd firmware/m5stick_imu
pio run -t upload
pio device monitor -b 115200
```

Serial prints `samples/s:` (expect ~485) and one line per tap. The LCD shows the
tap count and `sync` / `NOSYNC` — `NOSYNC` means timestamps are not in the
Jetson's epoch and the whole cross-clock premise is broken.

Tuning lives in `src/main.cpp`: `TAP_THRESHOLD` (90 m/s²),
`TAP_REFRACTORY_MS` (100), `SAMPLE_PERIOD_NS` (2 ms = 500 Hz).

---

## 6. Viewing

**container:**

```bash
ros2 run foxglove_bridge foxglove_bridge
```

Mac → Foxglove Studio → `ws://orin-nano.local:8765`.

Raw 720p over WiFi saturates the link and shows up as latency; use the compressed
topic, or throttle:

```bash
ros2 run topic_tools throttle messages /image_rect 5.0 /image_slow
```

---

## 7. Gotchas

| Symptom | Cause |
|---|---|
| `No executable found` | New shell without `source install/setup.bash`, or setup.py changed without a rebuild |
| `Failed to find ... package.sh` after deleting `install/` | `isaac_ros_apriltag` is a source package here — use `colcon build --packages-up-to tap_dance` |
| colcon fails on duplicate package name | Two repos in `src/` both providing the same package; only `tap-dance` should be there |
| `control_transfer ... Resource temporarily unavailable` | USB hiccup — Ctrl-C, `pkill -f realsense`, replug the D456, verify with `rs-enumerate-devices -s` |
| Camera node starts but no topics | Same USB issue; `RealSense Node Is Up!` prints before streaming is confirmed |
| `endPacket(): could not send data: 12` on the M5Stick | ENOMEM: publishing faster than WiFi drains. Why the IMU stream is decimated to 100 Hz |
| Taps never localize, `no detections at all` | Tag not in view, or the pose buffer is empty — check `tag ids seen` in the status line |
| `GXF_OUT_OF_MEMORY` at rectify | 1080p OOMs the 8 GB unified memory; stay at 720p |
| Git fails inside the container | Do git on the **host**; the workspace is bind-mounted so both see the same files |
