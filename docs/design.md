# Design notes

Why the system is built the way it is, and what the measurements said. The
[README](../README.md) has the summary; this has the reasoning.

---

## 1. The one sentence

> **The IMU says WHEN. The camera says WHERE. Neither can answer alone.**

| Question | Camera @30 Hz | IMU @500 Hz |
|---|---|---|
| Is the wand over the cup? | **yes** | no |
| Did it *touch*, or hover 2 cm above? | no — sub-cm, sub-frame | **yes**, sharp spike |
| Exactly *when*? | ±16 ms at best | **~1 ms** |
| Three taps in 600 ms? | no — 18 frames, contact invisible | **trivially** |

Contact is millimetres of displacement over ~10 ms, so a 30 Hz camera cannot tell
a tap from a hover. Double-integrated accelerometer position diverges in under a
second, so the IMU cannot say *which* object. Discrete event from the IMU, spatial
context from the camera.

---

## 2. Tapping rather than pointing

An earlier version of this project had you *point* the wand at objects. Tapping is
better conditioned, measurably:

| | Pointing | **Tapping** |
|---|---|---|
| Uses the tag's… | **orientation** — its worst axis, 1.9–5.7° measured | **position** — its best, ~mm |
| Error amplifier | distance to target, ~0.5 m | lever arm tag→contact, ~0.1 m |
| 5° of tag noise becomes | 4.4 cm | **0.9 cm** |
| Needs depth, intrinsics, deprojection? | yes | **no** |
| Needs a tag→stick extrinsic? | yes | **no** |
| Planar-PnP face-on degeneracy | a live problem | mostly irrelevant |

Five times less sensitive to the noise actually measured, and it deletes depth
alignment, intrinsics, deprojection, ray casting and the pointing-axis
calibration.

---

## 3. Association is horizontal only

Taps land on the *tops* of objects, so the tag sits ~10 cm above the contact point
and appears well above the object in the image — never inside its bounding box. But
a vertical tap changes the tag's `v` while preserving its `u`.

So a target is an interval of **columns**, and everything else falls away: no
depth, no TF, no intrinsics. Both the tag centre and the YOLO boxes are already in
image pixels.

Each target's interval extends halfway to its neighbour **on each side
independently**, so adjacent intervals meet exactly at the midpoint and no column
belongs to nobody. An earlier version used one symmetric tolerance per target —
half the distance to its *nearest* neighbour — which left gaps under uneven
spacing: with targets at 286, 432 and 854, the middle one took ±73 from its close
left neighbour and stopped at 505 while the right one began at 643, leaving a
138 px dead band. Only the outer edges of the outermost targets are bounded, so a
tap far off to one side is still rejected rather than silently attributed.

---

## 4. Two clock domains

The tap timestamp comes from the **M5Stick's clock**; the wand pose history from
the **Jetson's**. Answering "where was the wand at the tap instant" means indexing
one timeline with the other's timestamp, which only works because micro-ROS session
sync maps the ESP32's clock into the Jetson's epoch.

That makes the reaction-time score itself a cross-clock measurement: the prompt is
stamped on the Jetson, the tap on the M5Stick. Sync drift of 100 ms would make
every score wrong by 100 ms. At a ~1.5 m/s approach it would also misplace the wand
by ~15 cm, enough to attribute a tap to the wrong object.

Scores therefore use the tap's **stamp**, never its arrival: WiFi arrival jitter was
measured up to ~90 ms, and scoring on arrival would fold that into every number.

Two related details:

- **Interpolation.** Camera poses arrive every ~33 ms while the tap stamp has ~1 ms
  resolution, so the tap rarely lands on a frame. Where a close pair of detections
  brackets the tap, the position is interpolated between them; snapping to the
  nearest frame would concede up to 16 ms of wand travel. In practice hovering taps
  move ~0–1 px between frames, so this is insurance rather than load-bearing.
- **Last-pose fallback.** When no bracketing pair exists, the most recent pose
  before the tap is used, valid because a vertical tap preserves `u`. Its
  uncertainty is the horizontal drift over that interval, which the game uses to
  reject taps whose position could belong to more than one target.

---

## 5. Detection runs on the ESP32

Streaming samples fast enough to catch a 5–20 ms contact transient does not fit the
link. At a 400 Hz publish rate the ESP32 sustained ~300 callbacks/s, the Jetson
received ~175/s, and the UDP path threw `ENOMEM` — about 40% of samples lost, and a
lost sample can *be* the tap peak.

A tap is a rare, tiny event; the raw stream is high-rate bulky data. Detecting at
the source costs no bandwidth, loses nothing, and gives better time resolution than
any streaming rate the link allows. Measured after the change: 485 samples/s
on-device, 97 Hz received on the Jetson — exactly 485/5, so zero loss.

The cost is that the threshold lives in firmware, so retuning means a reflash. A
100 Hz stream is kept alongside for diagnostics.

Signal is `|‖a‖ − g|`: the accelerometer always reads gravity and its direction
changes as the wand rotates, so a per-axis threshold would depend on how you hold
it. Events are stamped at the first crossing rather than the peak, since onset is
physically closer to contact, and a refractory lockout collapses one contact's
ring-down into one event.

---

## 6. What actually limited tag detection

Detection in motion, waving at game pace:

| Sensor | Exposure | Room | Found % |
|---|---|---|---|
| colour | auto | dim | 38–70% |
| colour | 4 ms | dim | 50–70% |
| IR | auto | daylight | 50% |
| IR | 2–4 ms | daylight | 100% |
| **colour** | **2 ms** | **daylight** | **~100%** ← the default |

**Short exposure, plus enough light to use it.** Auto-exposure optimises brightness
and picks a long exposure that blurs the tag during the reach.

An intermediate reading pointed at the colour imager's rolling shutter: it reads
rows sequentially, so a moving tag is sheared as well as blurred, and no exposure
setting fixes shear. The IR imagers are global shutter and did reach 100% first.
That theory did not survive — once the room was bright enough to expose a 2 ms
frame, colour reached ~100% too. The apparent shutter advantage was confounded by
scene brightness, since the colour sweep had run in a dimmer room.

Colour is the default because visible light is abundant in any lit room while
near-IR is not: LED lighting emits almost none, and the projector cannot substitute
because its dot pattern lands on the tag. Colour also puts tag and YOLO detections
in one coordinate frame — the imagers are physically offset, so mixing them would
need a registration step. The IR path remains available for its global shutter.

---

## 7. Coordinate frames that look alike but are not

Two bugs had the same shape — two numbers that look like pixels but are measured in
different frames.

**Imager offset.** The colour and IR lenses are physically offset with slightly
different fields of view, so the same object sits at a different column in each.
Target positions must be re-measured after changing `sensor:`.

**Network versus image pixels.** `YoloV8DecoderNode` declares only `tensor_name`,
two thresholds and `num_classes` — it is never told the original image size — so it
emits bboxes in the network's 640×640 space while AprilTag reports full-image
pixels. The encoder scales the image uniformly by `min(net_w/img_w, net_h/img_h)`
into the top-left of the tensor and pads the rest, after which its crop computes a
zero offset and is a no-op, so the inverse is a pure scale: ×2 for 1280×720. Before
the fix an apple reported at u=143 sat under a tag reading u=290.

Both are now printed at startup, so a future mismatch announces itself instead of
looking like "the game is over the wrong object".

---

## 8. Results

| | |
|---|---|
| On-device sample rate | 485 Hz of 500 requested (rclc timer granularity) |
| Packet loss after decimation | 0 — 485/5 sampled = 97 Hz received |
| Tap vs. hard-swing separation | peak 100–120 vs 40–75 m/s² → threshold 90 |
| AprilTag out-of-plane jitter | 1.9–5.7° (12 cm tag), 5.7–11.3° (6 cm) |
| Tag detection in motion | ~100% at 2 ms exposure in a lit room |
| Tap → score latency | ~50 ms, from ~330 ms |
| Hand-configured targets | 10/10 hits, median reaction 868 ms |
| YOLO-discovered targets | 8/10 hits, median reaction 1203 ms |

The 10/10 and 8/10 runs differ mainly in target spacing: the hand-configured pair
sat 400 px apart, while YOLO found three objects with the closest pair 300 px
apart, so there was less room for a sloppy tap.

---

## 9. Approaches tried and dropped

| Dropped | Why |
|---|---|
| **Ballistic toy car** | Needed 2–3 m of hard floor; the room is carpeted and a pull-back car will not run on it |
| **Pointing at objects** | Uses the tag's noisiest axis and amplifies the error by target distance — see §2 |
| **Gesture "spell" wand** | An IMU-only gesture wand is a solved 2006 Wii remote; the camera would contribute nothing to a *how-is-it-moving* feature |
| **Camera-detected contact** | Hover and tap are indistinguishable at 30 Hz, and multi-tap is impossible. This is precisely what makes the IMU load-bearing |
| **Depth-based association** | Only needed for objects overlapping in image at different ranges, which laterally separated desk objects never do |
| **Symmetric per-target tolerance** | Left a dead band under uneven spacing — see §3 |
| **±16 g accelerometer range** | `setAccelFsr` is protected and M5Unified caches a scale factor from the range it believes is set, so writing the register directly would halve every reported value. Taps clip at 8 g harmlessly, since detection only needs them to separate from swings |
