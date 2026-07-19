// Milestone 2+3 — M5StickC Plus micro-ROS IMU client.
// Publishes sensor_msgs/Imu on /m5stick/imu over WiFi/UDP, timestamped in the
// AGENT's (Jetson) epoch via micro-ROS session time-sync — so its stamps are
// directly comparable to the D456's IMU (both in Jetson time). Enables the
// "thump the desk, compare the spike timestamps" cross-device sync test.
//
// M5.begin() initializes the MPU6886 IMU AND takes over the LCD (clearing the
// stale image from the previous firmware).

#include <Arduino.h>
#include <M5Unified.h>
#include <math.h>
#include <string.h>

#include <micro_ros_platformio.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>   // rmw_uros_sync_session / rmw_uros_epoch_nanos
#include <sensor_msgs/msg/imu.h>

#include "secrets.h"

rcl_publisher_t publisher;
sensor_msgs__msg__Imu msg;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

static const float ACCEL_G  = 9.80665f;          // g -> m/s^2
static const float DEG2RAD  = 3.14159265f / 180.0f;  // deg/s -> rad/s
static char frame_id[] = "m5stick_imu";

#define RCCHECK(fn)     { if ((fn) != RCL_RET_OK) { errorLoop(); } }
#define RCSOFTCHECK(fn) { rcl_ret_t rc_ = (fn); (void) rc_; }

void errorLoop() { while (true) { delay(200); } }

// Stamp the message with synced (agent/Jetson) epoch time.
void stampNow() {
  int64_t t_ns = rmw_uros_epoch_nanos();
  msg.header.stamp.sec     = (int32_t)(t_ns / 1000000000LL);
  msg.header.stamp.nanosec = (uint32_t)(t_ns % 1000000000LL);
}

void timer_callback(rcl_timer_t* t, int64_t /*last_call_time*/) {
  if (t == NULL) return;

  float ax, ay, az, gx, gy, gz;
  M5.Imu.getAccelData(&ax, &ay, &az);   // in g
  M5.Imu.getGyroData(&gx, &gy, &gz);    // in deg/s

  msg.linear_acceleration.x = ax * ACCEL_G;
  msg.linear_acceleration.y = ay * ACCEL_G;
  msg.linear_acceleration.z = az * ACCEL_G;
  msg.angular_velocity.x = gx * DEG2RAD;
  msg.angular_velocity.y = gy * DEG2RAD;
  msg.angular_velocity.z = gz * DEG2RAD;

  stampNow();
  RCSOFTCHECK(rcl_publish(&publisher, &msg, NULL));

  // Housekeeping at lower rates (100 Hz timer): re-sync the clock (drift) and
  // refresh the LCD. Cheap, and keeps stamps honest over time.
  static uint32_t n = 0;
  n++;
  if (n % 500 == 0) rmw_uros_sync_session(100);          // ~every 5 s
  if (n % 20 == 0) {                                     // ~5 Hz LCD
    M5.Display.setCursor(0, 40);
    M5.Display.printf("az:%+5.1f  %s", az * ACCEL_G,
                      rmw_uros_epoch_synchronized() ? "sync " : "NOSYNC");
  }
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);                 // inits IMU + LCD
  M5.Display.setTextSize(2);
  M5.Display.println("micro-ROS IMU");
  Serial.begin(115200);

  IPAddress agent_ip;
  agent_ip.fromString(AGENT_IP);
  set_microros_wifi_transports((char*) WIFI_SSID, (char*) WIFI_PASS, agent_ip, AGENT_PORT);
  delay(2000);

  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "m5stick_node", "", &support));
  // Best-effort QoS: right for high-rate sensor streams (no retransmit stalls).
  // NOTE: consumers must also use best-effort to match.
  RCCHECK(rclc_publisher_init_best_effort(
      &publisher, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, Imu),
      "m5stick/imu"));

  const unsigned int period_ms = 10;   // 100 Hz
  RCCHECK(rclc_timer_init_default(&timer, &support,
      RCL_MS_TO_NS(period_ms), timer_callback));
  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));

  // One-time message fields:
  msg.header.frame_id.data = frame_id;
  msg.header.frame_id.size = strlen(frame_id);
  msg.header.frame_id.capacity = sizeof(frame_id);
  msg.orientation.w = 1.0;              // no absolute orientation from a raw 6-axis IMU...
  msg.orientation_covariance[0] = -1.0; // ...so flag it invalid (REP-145 convention)

  // Initial clock sync with the agent (maps ESP32 clock -> Jetson epoch).
  rmw_uros_sync_session(1000);
}

void loop() {
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10)));
}
