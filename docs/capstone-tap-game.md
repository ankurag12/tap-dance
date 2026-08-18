# Capstone: Whack-a-Mole with a Tagged Wand

> 4–5 objects sit on the desk in the camera's view. The game calls one out —
> *"tap the CUP"* — and you have a few seconds. You bring the wand over it and **tap**.
> The **IMU** says *when* you tapped; the **camera** says *where the wand was* at that instant;
> the game scores your reaction time. Stretch: *"tap the cup **twice**."*

**Status:** Scope set 2026-08-02. Tap detector written; nothing measured yet.
**Supersedes:** the pointing-game scope, the ballistic-car scope, the tracked-wand scope.
See git history.

---

## 1. The one sentence

> **The IMU says WHEN. The camera says WHERE. Neither can answer alone.**

That is the whole project. Everything else follows from it.

| Question | Camera @30 Hz | IMU |
|---|---|---|
| Is the wand over the cup? | **yes** | no |
| Did it *touch*, or hover 2 cm above? | no — sub-cm, sub-frame | **yes**, sharp spike |
| Exactly *when*? | ±16 ms at best | **~1 ms** |
| Three taps in 600 ms? | no — 18 frames, contact invisible | **trivially** |

Contact is millimetres of displacement over ~10 ms. A 30 Hz camera cannot see it: hovering and
tapping look nearly identical. Conversely double-integrated accelerometer position diverges in
under a second, so the IMU cannot say *which* object. **Discrete event from the IMU, spatial
context from the camera.**

---

## 2. Why this design beats the earlier pointing version

| | Pointing | **Tapping** |
|---|---|---|
| Uses the tag's… | **orientation** (its worst axis: 1.9–5.7° measured) | **position** (its best axis: ~mm) |
| Error amplifier | distance to target (~0.5 m) | lever arm tag→contact (~0.1 m) |
| 5° of tag noise becomes | 4.4 cm | **0.9 cm** |
| Needs depth / deprojection? | yes | **no** — association is 2-D pixel-space |
| Needs tag→stick extrinsic? | yes | **no** |
| PnP face-on degeneracy | a live problem | mostly irrelevant |

**5× less sensitive to the noise we actually measured**, and it deletes depth alignment,
intrinsics, deprojection, ray casting, and the pointing-axis calibration. May well work with the
6 cm tag after all.

---

## 3. Why time-sync is load-bearing (not narrated)

The tap timestamp lives in the **M5Stick's clock**. The wand pose history lives in the **Jetson's
clock**. Answering "where was the wand at the tap instant" means indexing one timeline with the
other's timestamp — a genuine cross-clock lookup.

At a ~1.5 m/s approach speed:

| Clock skew δ | Position error | Outcome |
|---|---|---|
| 3 ms (our measured micro-ROS sync) | 0.5 cm | correct object |
| 30 ms | 4.5 cm | marginal |
| **100 ms** | **15 cm** | **wrong object reported** |

With objects 15–20 cm apart, injecting skew makes the game **visibly report the wrong answer** —
you tap the cup, it says you tapped the book. No explanation needed for a bystander.

Second-order effect worth measuring: the wand *decelerates* into contact, so sampling early
biases the position short of the object, along the approach direction — a systematic bias, not
random noise.

**Concept this forces:** the camera gives a pose every ~33 ms; the tap timestamp has ~1 ms
resolution, so the tap almost never lands on a frame. The wand position must be **interpolated
between the two straddling poses**. Snapping to the nearest frame instead would give away up to
16 ms (~2 cm) for free.

---

## 4. Architecture

```
  M5StickC Plus2 (handheld)                 JETSON (stationary)
   ┌──────────────────────┐
   │ AprilTag             │◀── sees ──  D456 ─▶ rectify ─▶ AprilTag ─▶ tag centre (px)
   │ MPU6886 accel        │                                              │
   └──────────┬───────────┘             D456 ─▶ YOLOv8 TensorRT ─▶ object boxes (px)
              │ micro-ROS/WiFi                                           │
              │ (stamps in Jetson epoch)                        ┌────────┴────────┐
              └── /m5stick/imu ──▶ TAP DETECTOR ──/wand/tap──▶  │  ASSOCIATION    │
                                   (|‖a‖-g| spike)              │  which box was  │
                                                                │  the tag in AT  │
                                                                │  the tap stamp? │
                                                                └────────┬────────┘
                                                                         │
                                                                  GAME (prompt,
                                                                  timer, score)
```

**No depth. No TF. No intrinsics.** Both the tag centre and the YOLO boxes are already in colour
image pixel coordinates, so association is 2-D containment/nearest-box. This is the single
biggest simplification over every earlier scope.

---

## 5. Milestones

| M | Deliverable | Reuses | New work |
|---|---|---|---|
| **M0** | **Tap detector characterized** | `firmware/m5stick_imu` | `tap_detector` (written). Measure tap vs. swing peaks; pick a threshold; **check whether 100 Hz is fast enough** |
| **M1** | Wand tracked well enough | `realsense_apriltag.launch.py`, `tag_pose_stats` | Move the wand around; confirm the tag centre is tracked over the whole play area, incl. at contact |
| **M2** | Objects located in 2-D | **YOLOv8s FP16 engine (done)**, `detection_probe` | Boxes in pixel coords; cache positions (objects are static) |
| **M3** | **Association: which object was tapped** | M0–M2 | Pose ring buffer; interpolate to the tap stamp; nearest/containing box |
| **M4** | Game | M3 | Prompt, countdown, scoring, Foxglove overlay |
| **M5** | *(stretch)* Multi-tap | M0 | Count taps in a window — near-free with the IMU, impossible at 30 Hz |
| **M6** | *(stretch)* Sync experiment | micro-ROS time-sync | Inject skew; show wrong-object reports; plot accuracy vs δ |

**One launch file** bringing up camera → rectify → AprilTag → YOLO is still needed; the current
`realsense_apriltag.launch.py` starts its own camera and collides with a standalone one.

---

## 6. Open questions (to be answered by measurement, not argument)

- [ ] **Is 100 Hz enough?** A tap transient is ~5–20 ms → 1–2 samples at 100 Hz. May need 200–500 Hz (MPU6886 supports 1 kHz)
- [ ] **Does `M5.Imu.getAccelData()` low-pass the signal?** Would smooth away the transient we need
- [ ] **Tap vs. swing separability** — amplitude alone, or is jerk needed?
- [ ] **Is the tag visible at contact?** Hand/object occlusion, motion blur on approach
- [ ] **Two-stage prompt?** ("move over the object", *then* "tap") — reduces the need for good tracking during fast motion. Decide after M1
- [ ] How coarse can association be? Possibly just 5 lateral sections rather than real boxes

---

## 7. Decisions / rejected

| Decision | Rationale |
|---|---|
| **Tapping, not pointing** | Uses the tag's good axis (position) instead of its bad one (orientation); 5× less error amplification; deletes depth + extrinsic calibration |
| **No depth** | Association is 2-D pixel-space. Depth would only matter for objects overlapping in image at different ranges — not a desk with laterally separated objects |
| **Not "camera detects contact"** | Hover and tap are indistinguishable at 30 Hz; multi-tap is impossible. This is what makes the IMU load-bearing |
| **Magnitude deviation `|‖a‖-g|`, not per-axis** | Orientation-independent, so it works however you hold the wand |
| **Stamp the tap ONSET, not the peak** | Onset is physically closer to the contact instant |
| **Refractory lockout** | One physical tap rings across several samples; also sets the fastest resolvable multi-tap |
| **Ballistic car, pointing game, gesture wand** | Dropped: carpet floor (no runway), orientation-noise-limited, and an IMU-only gesture wand is a solved 2006 Wii remote respectively |

---

## 8. Resume talking points

*(numbers to be filled from real runs)*

- Built an interactive tap-target game on a Jetson Orin Nano fusing a **wireless IMU** (micro-ROS
  over WiFi) with **fiducial tracking** and a **TensorRT object detector**: the IMU supplies the
  contact event to ~1 ms, the camera supplies spatial context, and correct attribution requires
  aligning **two independent clock domains**.
- Made **cross-device time synchronization** measurable rather than assumed: injected clock skew
  degrades object attribution predictably (100 ms → ~15 cm → wrong object), quantifying the
  timing budget the application actually needs.
- **Temporal interpolation** between 30 Hz camera poses to recover wand position at a
  sub-frame event timestamp.
- Full **sensor-platform integration**: NITROS composable pipeline, micro-ROS firmware, TensorRT
  engine deployment, reproducible containerized build, rosbag offline dev, health diagnostics.
