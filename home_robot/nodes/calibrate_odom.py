#!/usr/bin/env python3
"""Interactive odometry calibration helper for roomba_driver.py.

Run this AFTER roomba_driver.py is up and /odom is publishing (Roomba
must be awake — press the Clean button first if get_sensors() is failing).
Walks through a straight-line test and an in-place rotation test, then
prints corrected mm_per_tick / wheel_base values to pass to roomba_driver
(e.g. via --ros-args -p mm_per_tick:=... -p wheel_base:=...).

Usage:
    ros2 run home_robot calibrate_odom.py
"""

import math
import subprocess

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import tf_transformations


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


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

    def _cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = yaw
        if self._last_yaw is not None:
            self.unwrapped_yaw += normalize_angle(yaw - self._last_yaw)
        self._last_yaw = yaw


def spin_briefly(node, duration=0.3):
    end = node.get_clock().now().nanoseconds + int(duration * 1e9)
    while node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def main():
    rclpy.init()
    node = CalibrationListener()

    print('Waiting for /odom ...')
    while node.x is None:
        rclpy.spin_once(node, timeout_sec=0.5)
    print('  got /odom.\n')

    mm_per_tick_old = get_param('/roomba_driver', 'mm_per_tick', (math.pi * 72.0) / 508.8)
    wheel_base_old = get_param('/roomba_driver', 'wheel_base', 0.235)
    print(f'Current mm_per_tick = {mm_per_tick_old:.6f}, wheel_base = {wheel_base_old:.4f} m\n')

    # --- Test 1: straight-line distance ---
    print('=== Test 1: straight-line distance ===')
    print('Mark the robot\'s starting position on the floor.')
    input('Press Enter when ready, BEFORE you start moving... ')
    spin_briefly(node)
    x0, y0 = node.x, node.y

    print("Now push/drive the robot in a straight line for ~1-2 m.")
    input('Press Enter when it has STOPPED... ')
    spin_briefly(node)
    x1, y1 = node.x, node.y

    d_odom = math.hypot(x1 - x0, y1 - y0)
    print(f'  odom distance = {d_odom:.4f} m')
    d_true = float(input('  Enter the REAL distance measured with a tape (m): '))

    if d_odom > 1e-4:
        mm_per_tick_new = mm_per_tick_old * (d_true / d_odom)
    else:
        print('  odom distance ~0, skipping mm_per_tick update.')
        mm_per_tick_new = mm_per_tick_old
    print(f'  -> suggested mm_per_tick = {mm_per_tick_new:.6f}\n')

    # --- Test 2: in-place rotation ---
    print('=== Test 2: in-place rotation ===')
    print("Mark the robot's heading (e.g. tape on floor + a mark on the robot).")
    input('Press Enter when ready, BEFORE you start rotating... ')
    spin_briefly(node)
    node.unwrapped_yaw = 0.0
    node._last_yaw = node.yaw

    print('Now rotate the robot in place by a known angle')
    print('(one full turn / 360 deg, lining the marks back up, is easiest).')
    input('Press Enter when it has STOPPED rotating... ')
    spin_briefly(node)
    a_odom = node.unwrapped_yaw

    print(f'  odom rotation = {math.degrees(a_odom):.2f} deg')
    a_true_deg = float(input('  Enter the REAL angle turned in degrees (e.g. 360): '))
    a_true = math.radians(a_true_deg)

    if abs(a_true) > 1e-4:
        wheel_base_new = wheel_base_old * (mm_per_tick_new / mm_per_tick_old) * (a_odom / a_true)
    else:
        print('  real angle ~0, skipping wheel_base update.')
        wheel_base_new = wheel_base_old
    print(f'  -> suggested wheel_base = {wheel_base_new:.5f} m\n')

    print('=== Result ===')
    print('Pass these to roomba_driver, e.g.:')
    print('  ros2 run home_robot roomba_driver.py --ros-args '
          f'-p mm_per_tick:={mm_per_tick_new:.6f} -p wheel_base:={wheel_base_new:.5f}')
    print('Or set them as defaults in bringup.launch.py once verified.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
