// Milestone 1 — minimal micro-ROS client for the M5StickC Plus.
// Publishes an incrementing std_msgs/Int32 on /m5stick/counter over WiFi/UDP to
// the micro-ROS agent on the Jetson. Goal: verify client<->agent<->ROS-graph
// plumbing BEFORE adding the IMU.
//
// micro-ROS client structure (rclc): the C ROS client library. The lifecycle is
//   support (context+allocator) -> node -> publisher -> timer -> executor(spin).
// The executor runs callbacks (here, a periodic timer that publishes).

#include <Arduino.h>
#include <micro_ros_platformio.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/int32.h>

#include "secrets.h"

rcl_publisher_t publisher;
std_msgs__msg__Int32 msg;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

// micro-ROS convention: on a hard rcl failure, halt (a real firmware would
// reset/reconnect). RCSOFTCHECK ignores recoverable errors (e.g. a dropped publish).
#define RCCHECK(fn)     { if ((fn) != RCL_RET_OK) { errorLoop(); } }
#define RCSOFTCHECK(fn) { (void)(fn); }

void errorLoop() {
  while (true) { delay(200); }
}

// Fires every timer period; publishes the counter and increments it.
void timer_callback(rcl_timer_t* t, int64_t /*last_call_time*/) {
  if (t != NULL) {
    RCSOFTCHECK(rcl_publish(&publisher, &msg, NULL));
    msg.data++;
  }
}

void setup() {
  Serial.begin(115200);

  // Configure the WiFi/UDP transport to the agent (creds/IP from secrets.h).
  set_microros_wifi_transports(WIFI_SSID, WIFI_PASS, AGENT_IP, AGENT_PORT);
  delay(2000);  // give WiFi a moment to associate

  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "m5stick_node", "", &support));
  RCCHECK(rclc_publisher_init_default(
      &publisher, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
      "m5stick/counter"));

  const unsigned int period_ms = 1000;
  RCCHECK(rclc_timer_init_default(&timer, &support,
      RCL_MS_TO_NS(period_ms), timer_callback));

  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_timer(&executor, &timer));

  msg.data = 0;
}

void loop() {
  // Pump the executor: runs due timers/callbacks and services the transport.
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100)));
}
