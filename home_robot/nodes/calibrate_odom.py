#!/usr/bin/env python3
"""Interactive odometry calibration helper for roomba_driver.py.

Run this AFTER roomba_driver.py is up and /odom is publishing (Roomba
must be awake — press the Clean button first if get_sensors() is failing).
Walks through a straight-line test and an in-place rotation test, then
prints corrected mm_per_tick / wheel_base values to pass to roomba_driver
(e.g. via --ros-args -p mm_per_tick:=... -p wheel_base:=...).

If imu_node.py is also running and publishing imu/data, the rotation
test uses the IMU's measured yaw (gyro-integrated, not subject to wheel
slip) as ground truth instead of asking the user to manually align floor
marks and guess the angle turned — this removes the main source of error
from the previous (rejected) wheel_base calibration attempt, where
in-place rotation slip made the wheel-only result physically meaningless.

Usage:
    ros2 run home_robot calibrate_odom.py
"""

import math
import re
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import tf_transformations


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def read_float(prompt):
    """input() + float(), tolerant of trailing units like '0.9999 m' or '360 deg'."""
    while True:
        raw = input(prompt)
        match = re.search(r'-?\d+\.?\d*', raw)
        if match:
            return float(match.group())
        print(f"  could not parse a number from '{raw}', try again.")


def get_param(node_name, param_name, default):
    try:
        out = subprocess.run(
            ['ros2', 'param', 'get', node_name, param_name],
            capture_output=True, text=True, timeout=5,
        )
        for tok in out.stdout.replace(',', ' ').split():
            try:
                return float(tok)
            except ValueError:
                continue
    except Exception:
        pass
    print(f'  (could not read {param_name} from {node_name}, using default {default})')
    return default


class CalibrationListener(Node):
    def __init__(self):
        super().__init__('calibrate_odom')
        self.x = None
        self.y = None
        self.yaw = None
        self.unwrapped_yaw = 0.0
        self._last_yaw = None
        self.create_subscription(Odometry, 'odom', self._cb, 10)

        self.imu_yaw = None
        self.imu_unwrapped_yaw = 0.0
        self._imu_last_yaw = None
        self._last_imu_msg_time = None
        self.create_subscription(Imu, 'imu/data', self._imu_cb, 10)

    def _cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = yaw
        if self._last_yaw is not None:
            self.unwrapped_yaw += normalize_angle(yaw - self._last_yaw)
        self._last_yaw = yaw

    def _imu_cb(self, msg):
        q = msg.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.imu_yaw = yaw
        self._last_imu_msg_time = time.monotonic()
        if self._imu_last_yaw is not None:
            self.imu_unwrapped_yaw += normalize_angle(yaw - self._imu_last_yaw)
        self._imu_last_yaw = yaw

    def imu_is_live(self):
        """True if imu/data has been received recently (last 1s)."""
        return (
            self._last_imu_msg_time is not None
            and time.monotonic() - self._last_imu_msg_time < 1.0
        )


def main():
    rclpy.init()
    node = CalibrationListener()

    # input() blocks the main thread for as long as the user is moving/
    # rotating the robot — without a background spin thread, /odom
    # callbacks would only run in the brief windows between prompts, and
    # the bounded subscription queue (depth 10) would silently drop
    # almost the entire motion. Spinning continuously in a daemon thread
    # keeps self.x/y/yaw/unwrapped_yaw up to date the whole time.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('Waiting for /odom ...')
    while node.x is None:
        time.sleep(0.1)
    print('  got /odom.\n')

    mm_per_tick_old = get_param('/roomba_driver', 'mm_per_tick', (math.pi * 72.0) / 508.8)
    wheel_base_old = get_param('/roomba_driver', 'wheel_base', 0.235)
    print(f'Current mm_per_tick = {mm_per_tick_old:.6f}, wheel_base = {wheel_base_old:.4f} m\n')

    # --- Test 1: straight-line distance ---
    print('=== Test 1: straight-line distance ===')
    print('Mark the robot\'s starting position on the floor.')
    input('Press Enter when ready, BEFORE you start moving... ')
    time.sleep(0.3)
    x0, y0 = node.x, node.y

    print("Now push/drive the robot in a straight line for ~1-2 m.")
    input('Press Enter when it has STOPPED... ')
    time.sleep(0.3)
    x1, y1 = node.x, node.y

    d_odom = math.hypot(x1 - x0, y1 - y0)
    print(f'  odom distance = {d_odom:.4f} m')
    d_true = read_float('  Enter the REAL distance measured with a tape (m): ')

    if d_odom > 1e-4:
        mm_per_tick_new = mm_per_tick_old * (d_true / d_odom)
    else:
        print('  odom distance ~0, skipping mm_per_tick update.')
        mm_per_tick_new = mm_per_tick_old
    print(f'  -> suggested mm_per_tick = {mm_per_tick_new:.6f}\n')

    # --- Test 2: in-place rotation ---
    print('=== Test 2: in-place rotation ===')
    use_imu = node.imu_is_live()
    if use_imu:
        print('imu/data is live — using the IMU\'s measured yaw as ground')
        print('truth (no need to mark/align floor tape or guess the angle).')
    else:
        print('imu/data not detected — falling back to manual angle entry.')
        print("Mark the robot's heading (e.g. tape on floor + a mark on the robot).")
    input('Press Enter when ready, BEFORE you start rotating... ')
    time.sleep(0.3)
    node.unwrapped_yaw = 0.0
    node._last_yaw = node.yaw
    node.imu_unwrapped_yaw = 0.0
    node._imu_last_yaw = node.imu_yaw

    print('Now rotate the robot in place by a known angle')
    print('(one full turn / 360 deg is easiest' +
          ('' if use_imu else ', lining the marks back up') + ').')
    input('Press Enter when it has STOPPED rotating... ')
    time.sleep(0.3)
    a_odom = node.unwrapped_yaw
    print(f'  odom rotation = {math.degrees(a_odom):.2f} deg')

    if use_imu:
        a_true = node.imu_unwrapped_yaw
        print(f'  IMU rotation (ground truth) = {math.degrees(a_true):.2f} deg')
    else:
        a_true_deg = read_float('  Enter the REAL angle turned in degrees (e.g. 360): ')
        a_true = math.radians(a_true_deg)

    if abs(a_true) > 1e-4:
        wheel_base_new = wheel_base_old * (mm_per_tick_new / mm_per_tick_old) * (a_odom / a_true)
    else:
        print('  real angle ~0, skipping wheel_base update.')
        wheel_base_new = wheel_base_old

    if wheel_base_new <= 0:
        print(f'  WARNING: computed wheel_base ({wheel_base_new:.5f} m) is not positive — '
              'this usually means severe wheel slip during the turn (encoders barely '
              'moved relative to the real rotation). Not a valid value; keeping the '
              f'old wheel_base ({wheel_base_old:.4f} m). Try a slower, smoother turn, '
              'or average several repeated full-turn runs.')
        wheel_base_new = wheel_base_old
    print(f'  -> suggested wheel_base = {wheel_base_new:.5f} m\n')

    print('=== Result ===')
    print('Pass these to roomba_driver, e.g.:')
    print('  ros2 run home_robot roomba_driver.py --ros-args '
          f'-p mm_per_tick:={mm_per_tick_new:.6f} -p wheel_base:={wheel_base_new:.5f}')
    print('Or set them as defaults in bringup.launch.py once verified.')

    # Shutdown first so the spin() loop in the background thread sees
    # the invalid context and returns on its own — destroying the node
    # while another thread is still spinning it is what caused the
    # "terminate called without an active exception" abort.
    rclpy.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()


if __name__ == '__main__':
    main()
