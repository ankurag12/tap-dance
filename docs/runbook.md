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

ONE terminal:

```bash
ros2 launch tap_dance tap_game.launch.py tag_size:=0.1225 \
  target_names:='["cup","pen"]' target_u:='[600.0, 1000.0]'
```

With YOLOv8 naming the objects instead of hand-measured positions:

```bash
ros2 launch tap_dance tap_game.launch.py tag_size:=0.1225 use_yolo:=true \
  yolo_classes:='["cup","bottle","book"]'
```

This starts the camera, AprilTag, optionally YOLOv8/TensorRT, `tap_localizer` and
`tap_game`. The camera and NITROS are logged at WARN so the prompts stay readable;
add `quiet:=false` when debugging bring-up.

The micro-ROS agent is NOT part of this launch — it is a persistent Docker
container with its own lifecycle (see §0), since the wand connects to it whenever
it powers on, game or no game.

Common overrides:

```bash
  rounds:=10             time_limit:=6.0
  sensor:=infra1         # global shutter; needs a near-IR source
  exposure:=4000         # microseconds; raise in a dim room
  gain:=248              # only the stereo module accepts this much
  min_yolo_hits:=15      # detections before a class becomes a target
  wand_tag_id:=-1        # -1 = whichever tag is in view
```

Deeper per-node parameters (`tap_lockout`, `outer_margin`, `max_uncertainty`,
`max_stale`, `fresh_enough`) are not exposed by the launch; run the node directly
with `ros2 run` to change them.

## 2. Debug tools

Source lives in `tap_dance/tap_dance/debug/`; each is still a normal
`ros2 run tap_dance <name>` executable. They answer questions rather than playing.
Run them against a running `tap_game.launch.py`, or start the perception graph
alone with:

```bash
ros2 launch tap_dance realsense_apriltag.launch.py tag_size:=0.1225
```


**Where is the tag, and which object is it over?** The one to reach for when the
game says the wrong thing. Uses the same matching code as the game, so it cannot
disagree with it.

```bash
ros2 run tap_dance hover_probe --ros-args \
  -p yolo_classes:='["cup","banana","apple"]' -p wand_tag_id:=1 -p period:=1.0
```

```
  objects (only classes that can become targets):
      name              u(img)   u(net)      owns cols   hits  score
      apple              286.0    143.0    -114.. 359     120  0.71
      banana             432.0    216.0     359.. 643     118  0.66
      cup                854.0    427.0     643..1254     140  0.83
  tag: u=  290.4 v=  241.0  ->  OVER APPLE
       offsets  apple:+4   banana:-142(out)   cup:-564(out)
```

`u(img)` vs `u(net)` are both shown because a wrong scale is otherwise invisible —
it just looks like the tag being over the wrong object. `owns cols` is the column
interval that target claims; adjacent intervals MEET at the midpoint between two
targets, so there is no dead band, and only the outermost edges are bounded (by
`outer_margin`). `offsets` distinguishes a near-miss from being nowhere near.

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

| Sensor | Exposure | Room | Found % |
|---|---|---|---|
| colour | auto | dim | 38–70% |
| colour | 4 ms | dim | 50–70% |
| IR | auto | daylight | 50% (scarce near-IR -> AE picks a long exposure) |
| IR | 2–4 ms | daylight | 100% |
| **colour** | **2 ms** | **daylight** | **~100%** ← the default |

**It is exposure plus light, not shutter type.** A short exposure freezes the tag;
a bright scene is what makes a short exposure usable. An intermediate reading
suggested the colour imager's rolling shutter was shearing the tag, but colour
reaches ~100% too once the room is bright enough — the earlier colour sweep simply
ran in a dimmer room.

**A dim room breaks it.** At 2 ms in poor light the image underexposes and
detection falls. Fixes: more room light, a longer exposure (`exposure:=4000`),
or more gain (`gain:=248` works on `sensor:=infra1`, whose range is wider).

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
| OOM, or two cameras fighting | `tap_game.launch.py` already includes the perception graph — running `realsense_apriltag.launch.py` as well starts a second camera, rectify and TensorRT engine. Recover with `pkill -f component_container_mt`, `pkill -f realsense`, and on the host `sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'`; replug the D456 if it will not reopen |
| Git fails inside the container | Do git on the **host**; the workspace is bind-mounted so both see the same files |
