# Runbook

Kept in the repo because the dev container runs with `docker run --rm`, so shell
history does not survive a restart. `cat docs/runbook.md` from inside the container.

**host** = `hunter@orin-nano` · **container** = `admin@orin-nano:/workspaces/isaac_ros-dev`

---

## Bring-up

**host** — start the container, and the micro-ROS agent (persistent, its own
lifecycle: the wand connects whenever it powers on):

```bash
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh
```
```bash
docker run -d --restart unless-stopped --net=host \
  --name uros_agent microros/micro-ros-agent:humble udp4 --port 8888
```

Already created? `docker start uros_agent` · watch it with `docker logs -f uros_agent`.

**container** — build. `--packages-up-to` because `isaac_ros_apriltag` is a source
package here and `tap_dance` depends on it:

```bash
cd /workspaces/isaac_ros-dev && colcon build --packages-up-to tap_dance && source install/setup.bash
```

`source install/setup.bash` is needed in every new shell.

---

## Play

```bash
ros2 launch tap_dance tap_game.launch.py tag_size:=0.1225 use_yolo:=true \
  yolo_classes:='["cup","banana","apple"]' wand_tag_id:=1
```

Or with hand-measured positions instead of YOLO:

```bash
ros2 launch tap_dance tap_game.launch.py tag_size:=0.1225 \
  target_names:='["cup","pen"]' target_u:='[600.0, 1000.0]'
```

Measure `target_u` by hovering the wand over each object and reading `tag now at u=`
from `tap_localizer` (below). Re-measure if you change `sensor:` — the imagers are
physically offset.

Common overrides:

```bash
  rounds:=6              time_limit:=6.0
  exposure:=4000         # microseconds; raise in a dim room, at the cost of blur
  min_yolo_hits:=15      # detections before a class becomes a target
  verbose:=true          # log hover changes and rejected taps
```

### Viewing, and recording a video

Start the bridge, then connect Foxglove Studio on the Mac to
`ws://orin-nano.local:8765`:

```bash
ros2 run foxglove_bridge foxglove_bridge
```

Image panel on **`/game/overlay`** — already 960 px wide, so use it raw.
Start recording once `targets from YOLO:` has appeared; `rounds:=6` gives a ~40 s clip.

The overlay grades the picture for viewing only — the detector still sees the original
short-exposure frame:

```bash
  gamma:=2.0 contrast:=3.0 saturation:=1.6   # punchier
  contrast:=0.0 saturation:=1.0              # flat, if it looks overcooked
  use_overlay:=false                         # skip it if CPU is tight
```

Push `contrast` too high and it amplifies sensor noise. More room light is the better
trade — it buys a short exposure *and* a clean picture.

The overlay reads the camera's **compressed** stream by default — cheap to receive,
and the display never needed the rectified image. Confirm the topic name if it warns
that no frames are arriving:

```bash
ros2 topic list | grep compressed
```
```bash
ros2 launch tap_dance tap_game.launch.py ... overlay_image_topic:=<name>
```

The colour stream on this D456 delivers ~20 Hz rather than 30, jittery, over USB.
That is pre-existing and not caused by the overlay. It means tag poses can be
100–400 ms old at the tap instant, which is why the localizer extrapolates rather
than using the last sighting directly.

**The HUD is time-aligned to the frame.** A picture reaches the viewer a few hundred
ms after capture, while the HUD arrives in tens, so drawing the newest text onto the
frame in hand showed the score *before* the recorded picture showed the tap. Each
frame is instead drawn with the HUD state that was current at its capture time. The
overlay logs `frame age at render` so you can see how far behind the picture is;
`sync_hud_to_frame:=false` reverts to snappier but misaligned feedback.

---

## Debug tools

Run against a live game, or start perception alone with:

```bash
ros2 launch tap_dance realsense_apriltag.launch.py tag_size:=0.1225 use_yolo:=true
```

| Command | Shows |
|---|---|
| `ros2 run tap_dance hover_probe --ros-args -p wand_tag_id:=1 -p period:=1.0` | objects and the columns they own, where the tag is, which one it is over |
| `ros2 run tap_dance tap_localizer --ros-args -p status_period:=1.0` | `wand tag found N%`, live tag `u`, taps located |
| `ros2 run tap_dance tag_pose_stats --ros-args -p period:=5.0` | pose jitter with the tag held still |
| `ros2 run tap_dance tap_detector --ros-args -p report_period:=3.0` | tap vs. swing peaks on the 100 Hz stream |
| `ros2 run tap_dance detection_probe --ros-args -p period:=2.0` | every COCO class YOLO sees, by name |

`hover_probe` uses the same matching code as the game, so it cannot disagree with it.
It prints `u(img)` beside `u(net)`: YOLO bboxes arrive in the network's 640×640 space
and are scaled ×2 into image pixels.

Raw checks:

```bash
ros2 topic hz /m5stick/imu       # ~97 Hz
ros2 topic hz /tag_detections    # ~30 Hz
ros2 topic echo /wand/tap        # one message per contact
```

---

## Firmware

**Mac** — `pio` lives at `~/.platformio/penv/bin/pio` if it is not on your `PATH`.
Needs `include/secrets.h` with `WIFI_SSID`, `WIFI_PASS`, `AGENT_IP` (the Jetson's LAN
address) and `AGENT_PORT`:

```bash
cd firmware/m5stick_imu
pio run -t upload
pio device monitor -b 115200
```

Serial prints `samples/s:` (~485) and every tap's onset and peak. The LCD shows the tap
count and `sync` / `NOSYNC` — **`NOSYNC` means timestamps are not in the Jetson's epoch
and attribution is invalid.**

Tuning is in `src/main.cpp`: `TAP_THRESHOLD` (70 m/s²), `TAP_REFRACTORY_MS` (100),
`SAMPLE_PERIOD_NS` (2 ms). Taps feeling heavy or twitchy? Watch the serial peaks, adjust
the threshold, reflash. Measured: firm taps 100–120, hard swing *stops* 40–75.

---

## Gotchas

| Symptom | Cause |
|---|---|
| `No executable found` | New shell without `source install/setup.bash`, or `setup.py` changed without a rebuild |
| `Failed to find ... package.sh` after deleting `install/` | `isaac_ros_apriltag` is a source package — use `colcon build --packages-up-to tap_dance` |
| OOM, or two cameras contending | `tap_game.launch.py` already includes the perception graph; running `realsense_apriltag.launch.py` too starts a second camera and TensorRT engine. `pkill -f component_container_mt`, `pkill -f realsense`, then on the host `sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'` |
| `control_transfer ... Resource temporarily unavailable` | USB hiccup — Ctrl-C, `pkill -f realsense`, replug the D456, check `rs-enumerate-devices -s` |
| `GXF_OUT_OF_MEMORY` at rectify | 1080p exhausts the 8 GB unified memory; stay at 720p |
| Taps detected but never localized | Tag not visible at contact. Check `wand tag found N%`; usually too dark, or too long an exposure |
| Wrong object reported | Stale `target_u` after changing `sensor:`, or objects too close together |
| Git fails inside the container | Do git on the **host**; the workspace is bind-mounted so both see the same files |
