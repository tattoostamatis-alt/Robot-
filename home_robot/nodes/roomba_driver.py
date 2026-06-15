#!/usr/bin/env python3
"""Roomba 692 → ROS2 driver via Create 2 Open Interface."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformBroadcaster
import pycreate2
import math
import struct
import time


class RoombaDriver(Node):
    def __init__(self):
        super().__init__('roomba_driver')

        self.declare_parameter('port', '/dev/roomba')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('wheel_base', 0.235)
        self.declare_parameter('idle_timeout', 30.0)  # seconds until passive mode
        # Calibrated 2026-06-13 via calibrate_odom.py (straight-line test:
        # odom reported 1.2372m for a real 1.0m drive). Theoretical Create 2
        # spec value was (pi*72)/508.8 = 0.44447; this unit's actual wheels
        # need ~19% less.
        self.declare_parameter('mm_per_tick', 0.359338)

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.idle_timeout = self.get_parameter('idle_timeout').value
        self.mm_per_tick = self.get_parameter('mm_per_tick').value

        self.bot = pycreate2.Create2(port, baud)
        self.bot.wake()
        self.bot.start()
        time.sleep(0.5)
        self.bot.full()
        time.sleep(0.3)
        self.get_logger().info(f'Roomba connected on {port}')

        self._idle = False
        self._stopped = True
        self._last_cmd_time = time.monotonic()

        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)
        self.dock_sub = self.create_subscription(Bool, 'dock', self._dock_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.battery_pub = self.create_publisher(BatteryState, 'battery/state', 10)
        self.charge_ratio_pub = self.create_publisher(Float32, 'battery/charge_ratio', 10)

        self.create_timer(0.05, self._publish_odom)   # 20 Hz
        self.create_timer(0.1,  self._watchdog)        # safety stop
        self.create_timer(2.0,  self._publish_battery) # 0.5 Hz
        self.create_timer(5.0,  self._check_idle)      # idle watchdog

        self._x = 0.0
        self._y = 0.0
        self._th = 0.0
        self._prev_left_enc = None
        self._prev_right_enc = None

    def _cmd_vel_cb(self, msg: Twist):
        self._last_cmd_time = time.monotonic()
        self._stopped = False
        if self._idle:
            self._idle = False
            self.bot.full()
            self.get_logger().info('Roomba: waking up (full mode)')

        linear = msg.linear.x * 1000   # m/s → mm/s
        angular = msg.angular.z         # rad/s

        right = int(linear + angular * self.wheel_base * 500)
        left  = int(linear - angular * self.wheel_base * 500)

        right = max(-500, min(500, right))
        left  = max(-500, min(500, left))

        self.bot.drive_direct(right, left)

    def _watchdog(self):
        if not self._idle and not self._stopped and (time.monotonic() - self._last_cmd_time) > 0.1:
            self.bot.drive_direct(0, 0)
            self._stopped = True

    def _check_idle(self):
        if not self._idle and (time.monotonic() - self._last_cmd_time) > self.idle_timeout:
            self._idle = True
            self.bot.drive_direct(0, 0)
            self._stopped = True
            self.get_logger().info(
                f'Roomba: idle {self.idle_timeout:.0f}s → passive mode (low power)'
            )

    def _safe_get_sensors(self):
        """get_sensors() με retry — το Roomba μερικές φορές επιστρέφει 0 bytes."""
        for attempt in range(3):
            try:
                # Discard any stale/leftover bytes so this read is aligned to
                # the response of the query we're about to send — otherwise a
                # backlog from a delayed previous cycle desyncs every field.
                self.bot.SCI.ser.reset_input_buffer()
                return self.bot.get_sensors()
            except Exception as e:
                self.get_logger().warn(f'get_sensors() attempt {attempt}: {e!r}')
                time.sleep(0.05)
        self.get_logger().error('get_sensors() failed after 3 attempts')
        return None

    def _dock_cb(self, msg: Bool):
        if msg.data:
            self.bot.seek_dock()
            self.get_logger().info('Seeking dock...')

    # Max plausible per-cycle change at 20Hz (well above the Create 2's
    # ~0.5 m/s top speed) — anything beyond this is a corrupted/misaligned
    # serial read, not real encoder data.
    MAX_DELTA_D = 0.05   # m
    MAX_DELTA_A = math.radians(30)  # rad

    # Packets 19/20 (Distance/Angle) read as 0 on this unit's firmware, so
    # odometry is computed from raw left/right encoder counts (43/44)
    # instead, scaled by the mm_per_tick parameter.

    def _safe_get_encoders(self):
        """Read packets 43/44 (left/right encoder counts) με retry."""
        for attempt in range(3):
            try:
                self.bot.SCI.ser.reset_input_buffer()
                self.bot.SCI.write(149, (2, 43, 44))  # Query List: 2 packets, IDs 43+44
                time.sleep(0.015)
                data = self.bot.SCI.read(4)
                if len(data) != 4:
                    raise Exception(f'Encoder data not 4 bytes long, it is: {len(data)}')
                return struct.unpack('>hh', data)
            except Exception as e:
                self.get_logger().warn(f'get_encoders() attempt {attempt}: {e!r}')
                time.sleep(0.05)
        self.get_logger().error('get_encoders() failed after 3 attempts')
        return None

    def _publish_odom(self):
        enc = self._safe_get_encoders()
        if enc is None:
            return
        left, right = enc

        if self._prev_left_enc is None:
            self._prev_left_enc = left
            self._prev_right_enc = right
            return

        d_left = left - self._prev_left_enc
        d_right = right - self._prev_right_enc
        # 16-bit signed counter wraparound
        if d_left > 32768:
            d_left -= 65536
        elif d_left < -32768:
            d_left += 65536
        if d_right > 32768:
            d_right -= 65536
        elif d_right < -32768:
            d_right += 65536

        self._prev_left_enc = left
        self._prev_right_enc = right

        delta_left_m  = d_left  * self.mm_per_tick / 1000.0
        delta_right_m = d_right * self.mm_per_tick / 1000.0

        delta_d = (delta_left_m + delta_right_m) / 2.0
        delta_a = (delta_right_m - delta_left_m) / self.wheel_base

        if abs(delta_d) > self.MAX_DELTA_D or abs(delta_a) > self.MAX_DELTA_A:
            self.get_logger().warn(
                f'Discarding implausible odom delta: d_left={d_left} d_right={d_right} ticks '
                '(corrupted sensor read?)',
                throttle_duration_sec=5.0,
            )
            delta_d = 0.0
            delta_a = 0.0

        self._th += delta_a
        self._x  += delta_d * math.cos(self._th)
        self._y  += delta_d * math.sin(self._th)

        import tf_transformations
        q = tf_transformations.quaternion_from_euler(0, 0, self._th)
        now = self.get_clock().now().to_msg()

        # TF: odom → base_link  (required by SLAM Toolbox)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_link'
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        self.odom_pub.publish(odom)

    def _publish_battery(self):
        sensors = self._safe_get_sensors()
        if sensors is None:
            return

        charge    = getattr(sensors, 'battery_charge', None)    # mAh
        capacity  = getattr(sensors, 'battery_capacity', None)  # mAh
        voltage   = getattr(sensors, 'voltage', None)           # mV
        current   = getattr(sensors, 'current', None)           # mA
        charging  = getattr(sensors, 'charger_state', 0)

        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        if voltage is not None:
            msg.voltage = voltage / 1000.0
        if current is not None:
            msg.current = current / 1000.0
        if charge is not None:
            msg.charge = charge / 1000.0
        if capacity is not None and capacity > 0:
            msg.capacity = capacity / 1000.0
            msg.design_capacity = msg.capacity
            ratio = charge / capacity if charge is not None else float('nan')
            msg.percentage = ratio
            self.charge_ratio_pub.publish(Float32(data=float(ratio)))
        else:
            msg.percentage = float('nan')

        msg.power_supply_status = (
            BatteryState.POWER_SUPPLY_STATUS_CHARGING
            if charging else BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        )
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_NIMH
        msg.present = True

        self.battery_pub.publish(msg)

    def destroy_node(self):
        self.bot.drive_direct(0, 0)
        self.bot.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = RoombaDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
