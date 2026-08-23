# Design notes

## The one sentence

> **The IMU says WHEN. The camera says WHERE. Neither can answer alone.**

| Question | Camera @30 Hz | IMU @500 Hz |
|---|---|---|
| Is the wand over the cup? | **yes** | no |
| Did it *touch*, or hover 2 cm above? | no — sub-cm, sub-frame | **yes** |
| Exactly *when*? | ±16 ms at best | **~1 ms** |
| Three taps in 600 ms? | no | **trivially** |

Contact is millimetres of displacement over ~10 ms, so a 30 Hz camera cannot tell a
tap from a hover. Double-integrated accelerometer position diverges in under a
second, so the IMU cannot say *which* object.

## Association is horizontal only

Taps land on the *tops* of objects, so the tag sits ~10 cm above the contact point
and never falls inside an object's bounding box — but a vertical tap preserves the
tag's `u`. A target is therefore an interval of image columns, which removes any need
for depth, TF or intrinsics.

Intervals extend halfway to each neighbour independently, so adjacent ones meet at
the midpoint and no column belongs to nobody; only the outermost edges are bounded,
so a tap far off to one side is still rejected.

## Two clock domains

The tap timestamp comes from the M5Stick's clock, the pose history from the Jetson's.
Indexing one timeline with the other's timestamp only works because micro-ROS session
sync maps the ESP32's clock into the Jetson's epoch.

So the reaction-time score is itself a cross-clock measurement — and it uses the
tap's **stamp**, never its arrival, because WiFi arrival jitter reaches ~90 ms.

Camera poses arrive every ~33 ms while the tap stamp has ~1 ms resolution, so the tap
rarely lands on a frame. Where two detections bracket it the position is interpolated;
otherwise the most recent pose before the tap is used, valid because a vertical tap
preserves `u`. That fallback's uncertainty is the horizontal drift over the gap, and
the game rejects a tap whose uncertainty could place it in more than one target.

## Detection runs on the ESP32

Streaming samples fast enough to catch a 5–20 ms contact transient did not fit the
link: at 400 Hz the ESP32 sustained ~300 callbacks/s, the Jetson received ~175/s, and
the UDP path threw `ENOMEM` — ~40% of samples lost, and a lost sample can *be* the tap
peak. Detecting at the source costs no bandwidth and loses nothing: 485 samples/s
on-device, 97 Hz received, zero loss. The cost is that the threshold lives in firmware.

The signal is `|‖a‖ − g|`, which is orientation-independent — a per-axis threshold
would depend on how you hold the wand. Events are stamped at the first crossing rather
than the peak, since onset is closer to contact, and a refractory lockout collapses one
contact's ring-down into a single event.

QoS is deliberately asymmetric: best-effort for the IMU stream, where a retransmit
would stall newer samples and one lost sample is invisible; reliable for `/wand/tap`,
where each message is a game event and losing one means the hit does not register.

## Nodes

Runtime — `tap_dance/`:

| Node | Role |
|---|---|
| `tap_localizer` | Cross-clock lookup: tag pixel position at the tap instant |
| `tap_game` | Round loop, target matching, reaction-time scoring |
| `game_overlay` | Renders prompt + score onto a graded frame → `/game/overlay` |
| `targets` | Shared library: YOLO→image scaling, target discovery, bounds, matching |

Debug tools — `tap_dance/debug/`:

| Tool | Answers |
|---|---|
| `hover_probe` | Where is the tag, which object is it over, what does each own? |
| `tag_pose_stats` | How much does the tag's pose jitter? |
| `tap_detector` | Do taps separate from swings, and at what threshold? |
| `detection_probe` | What is YOLO seeing, across all 80 COCO classes? |

`targets.py` sits in the runtime package even though the debug tools use it:
dependencies point from `debug/` toward the runtime, never the reverse, so
`hover_probe` cannot disagree with the game about which object a position matches.

## Results

| | |
|---|---|
| On-device sample rate | 485 Hz of 500 requested |
| Packet loss after decimation | 0 — 485/5 sampled = 97 Hz received |
| Tap vs. hard-swing separation | peak 100–120 vs 40–75 m/s² |
| AprilTag out-of-plane jitter | 1.9–5.7° (12 cm tag), 5.7–11.3° (6 cm) |
| Tag detection in motion | ~100% at 2 ms exposure in a lit room, ~50% on auto |
| Tap → score latency | ~50 ms |
| Tap localization | every tap resolved to an object across a 10-round session |
