// M5StickC Plus micro-ROS client — IMU stream + ON-DEVICE tap detection.
//
// Publishes two topics, both stamped in the AGENT's (Jetson) epoch via
// micro-ROS session time-sync, so their stamps are directly comparable to
// anything the Jetson timestamps itself (camera, D456 IMU):
//
//   /m5stick/imu   sensor_msgs/Imu     100 Hz, diagnostics + monitoring
//   /wand/tap      std_msgs/Header     one message per detected tap; the STAMP
//                                      is the payload — the contact instant
//
// WHY DETECTION RUNS HERE, NOT ON THE JETSON
// Streaming raw samples fast enough to catch a 5-20 ms contact transient does
// not fit the link. Measured: at a 400 Hz publish rate the ESP32 only sustained
// ~300 callbacks/s, the Jetson received ~175/s, and the UDP path threw
// `endPacket(): could not send data: 12` (ENOMEM — LwIP send buffers exhausted).
// ~40% of samples were being dropped, and a dropped sample can BE the tap peak.
//
// A tap is a rare, tiny event; the raw stream is high-rate, bulky data. So
// detect at the source and send the event: sampling at 500 Hz never touches the
// network, no sample can be lost, the time resolution is better than any
// streaming rate we could achieve, and the link stops being a bottleneck.
// Cost: the threshold lives in firmware, so retuning means a reflash. The
// 100 Hz IMU stream is kept so `tap_detector` (characterize mode) and
// sensor_monitor still have data to look at.
//
// M5.begin() initializes the MPU6886 IMU AND takes over the LCD (clearing the
// stale image from the previous firmware).

#include <Arduino.h>
#include <M5Unified.h>
#include <WiFi.h>          // WiFi.setSleep() — modem-sleep control, see setup()
#include <math.h>
#include <string.h>

#include <micro_ros_platformio.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>   // rmw_uros_sync_session / rmw_uros_epoch_nanos
#include <sensor_msgs/msg/imu.h>
#include <std_msgs/msg/header.h>

#include "secrets.h"

rcl_publisher_t imu_pub;
rcl_publisher_t tap_pub;
sensor_msgs__msg__Imu imu_msg;
std_msgs__msg__Header tap_msg;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

static const float ACCEL_G = 9.80665f;               // g -> m/s^2
static const float DEG2RAD = 3.14159265f / 180.0f;   // deg/s -> rad/s
static char imu_frame_id[] = "m5stick_imu";
static char tap_frame_id[] = "wand_tap";

// --- Tap detection tuning ---------------------------------------------------
// Threshold on |‖a‖ - g|. The accelerometer always reads gravity and its
// direction changes as the wand rotates, so a per-axis threshold would depend
// on how you hold it; the magnitude deviation is orientation-independent.
//
// Measured with this wand: firm taps peak at 100-120, hard swing STOPS at 40-75.
// (Taps clip at the +-8G rail, so their true peaks are higher -- harmless, since
// only the separation matters.) 90 sat safely in that gap but demanded a firm tap.
//
// 70 deliberately dips into the top of the swing range so lighter taps register.
// That is acceptable because a stray trigger has to clear three game-side guards
// before it can score: taps only count during an active round, a 400 ms lockout
// follows every scored tap, and a tap whose position could belong to more than one
// target is rejected outright. Lower it further if taps still feel heavy -- the
// firmware prints every onset and peak over serial, so tune against real numbers
// rather than guessing, and re-measure with `ros2 run tap_dance tap_detector` if
// the wand or the objects change.
static const float TAP_THRESHOLD = 70.0f;            // m/s^2

// One physical tap rings across several samples. The lockout collapses those
// into one event; it also sets the fastest resolvable multi-tap, so it must
// stay well under the interval of deliberate tapping.
static const uint32_t TAP_REFRACTORY_MS = 100;

// Sampling period. 500 Hz (2 ms) puts ~5-6 samples inside a 10 ms contact and
// is purely local, so it costs no bandwidth. IMU messages go out every 5th
// sample => 100 Hz on the wire, which the link carries comfortably.
static const uint64_t SAMPLE_PERIOD_NS = 2000000ULL;  // 2 ms = 500 Hz
static const uint32_t IMU_PUBLISH_EVERY = 5;          // 500 / 5 = 100 Hz

// Housekeeping divisors, in samples at 500 Hz. Retune if SAMPLE_PERIOD_NS moves.
static const uint32_t RESYNC_EVERY = 2500;            // ~5 s
static const uint32_t LCD_EVERY = 100;                // ~5 Hz

#define RCCHECK(fn)     { if ((fn) != RCL_RET_OK) { errorLoop(); } }
#define RCSOFTCHECK(fn) { rcl_ret_t rc_ = (fn); (void) rc_; }

void errorLoop() { while (true) { delay(200); } }

// Synced (agent/Jetson) epoch time, split into a ROS stamp.
static void stampNow(builtin_interfaces__msg__Time* stamp) {
  int64_t t_ns = rmw_uros_epoch_nanos();
  stamp->sec     = (int32_t)(t_ns / 1000000000LL);
  stamp->nanosec = (uint32_t)(t_ns % 1000000000LL);
}

void timer_callback(rcl_timer_t* t, int64_t /*last_call_time*/) {
  if (t == NULL) return;

  float ax, ay, az, gx, gy, gz;
  M5.Imu.getAccelData(&ax, &ay, &az);   // in g
  M5.Imu.getGyroData(&gx, &gy, &gz);    // in deg/s

  const float axm = ax * ACCEL_G, aym = ay * ACCEL_G, azm = az * ACCEL_G;

  // --- tap detection, every sample (500 Hz) ---
  const float dev = fabsf(sqrtf(axm * axm + aym * aym + azm * azm) - ACCEL_G);
  const uint32_t now_ms = millis();

  static bool armed = true;
  static uint32_t last_tap_ms = 0;
  static float tap_peak = 0.0f;
  static uint32_t tap_count = 0;

  if (armed && dev >= TAP_THRESHOLD) {
    // Publish on the FIRST crossing, not the peak: onset is physically closer
    // to the contact instant, and publishing immediately keeps latency low —
    // which matters, because the game scores reaction time.
    armed = false;
    last_tap_ms = now_ms;
    tap_peak = dev;
    tap_count++;

    stampNow(&tap_msg.stamp);
    RCSOFTCHECK(rcl_publish(&tap_pub, &tap_msg, NULL));
    Serial.printf("TAP #%lu  onset %.1f m/s^2\n",
                  (unsigned long) tap_count, dev);
  } else if (!armed) {
    if (dev > tap_peak) tap_peak = dev;
    if (now_ms - last_tap_ms >= TAP_REFRACTORY_MS) {
      armed = true;
      Serial.printf("   tap #%lu peak %.1f\n",
                    (unsigned long) tap_count, tap_peak);
    }
  }

  static uint32_t n = 0;
  n++;

  // --- IMU stream, decimated to 100 Hz ---
  if (n % IMU_PUBLISH_EVERY == 0) {
    imu_msg.linear_acceleration.x = axm;
    imu_msg.linear_acceleration.y = aym;
    imu_msg.linear_acceleration.z = azm;
    imu_msg.angular_velocity.x = gx * DEG2RAD;
    imu_msg.angular_velocity.y = gy * DEG2RAD;
    imu_msg.angular_velocity.z = gz * DEG2RAD;
    stampNow(&imu_msg.header.stamp);
    RCSOFTCHECK(rcl_publish(&imu_pub, &imu_msg, NULL));
  }

  if (n % RESYNC_EVERY == 0) rmw_uros_sync_session(100);

  // True sampling rate, measured at the source. `ros2 topic hz` on the Jetson
  // measures ARRIVALS and so cannot distinguish "sampled fewer" from "WiFi ate
  // some" — this can. If it reads well below 500, the ESP32 is CPU-bound and
  // SAMPLE_PERIOD_NS should be relaxed.
  static uint32_t last_report_ms = 0;
  static uint32_t last_report_n = 0;
  if (now_ms - last_report_ms >= 1000) {
    Serial.printf("samples/s: %lu  taps: %lu\n",
                  (unsigned long) (n - last_report_n) * 1000 / (now_ms - last_report_ms),
                  (unsigned long) tap_count);
    last_report_ms = now_ms;
    last_report_n = n;
  }

  if (n % LCD_EVERY == 0) {
    M5.Display.setCursor(0, 40);
    M5.Display.printf("taps:%-4lu %s", (unsigned long) tap_count,
                      rmw_uros_epoch_synchronized() ? "sync " : "NOSYNC");
  }
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);                 // inits IMU + LCD
  M5.Display.setTextSize(2);
  M5.Display.println("uROS tap");
  Serial.begin(115200);

  // The MPU6886 refreshes its data registers at internal_rate/(1+SMPLRT_DIV).
  // M5Unified's begin() leaves SMPLRT_DIV=3 -> 250 Hz, which would make a
  // 500 Hz poll re-read the same sample half the time. 1 -> 500 Hz.
  //
  // Two registers deliberately left alone:
  //   ACCEL_CONFIG2 (0x1D) is already 0x00 = ~218 Hz bandwidth, wide enough to
  //     pass a 5-20 ms tap transient, so no DLPF change is needed.
  //   ACCEL_CONFIG (0x1C) stays at +-8G. Reaching +-16G means writing this
  //     register directly (setAccelFsr is protected), but M5Unified caches a
  //     scale factor from the range it believes is set -- so every REPORTED
  //     value would be halved. Taps do clip at 8 g, which is harmless here:
  //     detection only needs taps to separate from swings, and they do.
  m5::In_I2C.writeRegister8(0x68, 0x19, 0x01, 400000);   // addr, SMPLRT_DIV, val

  IPAddress agent_ip;
  agent_ip.fromString(AGENT_IP);
  set_microros_wifi_transports((char*) WIFI_SSID, (char*) WIFI_PASS, agent_ip, AGENT_PORT);
  delay(2000);

  // Disable WiFi modem sleep: by default the ESP32 parks its radio between DTIM
  // beacons, stalling transmission for ~100 ms at a time. Costs battery life,
  // which matters little here (short sessions, often USB-tethered).
  // Must come AFTER set_microros_wifi_transports(), which brings up the link.
  WiFi.setSleep(false);

  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "m5stick_node", "", &support));

  // Best-effort for the IMU stream: right for a high-rate sensor feed, where a
  // retransmit would stall newer samples. Consumers must match best-effort.
  RCCHECK(rclc_publisher_init_best_effort(
      &imu_pub, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, Imu),
      "m5stick/imu"));

  // RELIABLE for taps: these are rare, small, and each one is a game event.
  // Losing an IMU sample is invisible; losing a tap means the player's hit does
  // not register. Opposite trade-off from the stream, hence opposite QoS.
  RCCHECK(rclc_publisher_init_default(
      &tap_pub, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Header),
      "wand/tap"));

  RCCHECK(rclc_timer_init_default(&timer, &support, SAMPLE_PERIOD_NS, timer_callback));
  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));

  // One-time message fields:
  imu_msg.header.frame_id.data = imu_frame_id;
  imu_msg.header.frame_id.size = strlen(imu_frame_id);
  imu_msg.header.frame_id.capacity = sizeof(imu_frame_id);
  imu_msg.orientation.w = 1.0;              // no absolute orientation from a raw 6-axis IMU...
  imu_msg.orientation_covariance[0] = -1.0; // ...so flag it invalid (REP-145 convention)

  tap_msg.frame_id.data = tap_frame_id;
  tap_msg.frame_id.size = strlen(tap_frame_id);
  tap_msg.frame_id.capacity = sizeof(tap_frame_id);

  // Initial clock sync with the agent (maps ESP32 clock -> Jetson epoch).
  rmw_uros_sync_session(1000);
}

void loop() {
  // Tight spin: the timer period is 2 ms, so a larger budget here would let
  // timer firings bunch up and jitter the sample spacing.
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1)));
}
