#!/usr/bin/env python3
"""Odometry calibration helper for a chassis swap.

Reports the raw WHEEL odometry travelled since the last reset -- straight-line
distance and cumulative yaw -- so you can compare it against a tape measure /
physical turn count and back out the two driver constants.

IMPORTANT: run ONLY roomba_driver.py alongside this (NOT the EKF). /odom must be
pure wheel odometry; if the EKF is running, yaw comes from the IMU and the
wheel_base calibration is meaningless.

Procedure
  mm_per_tick  (straight-line):
    1. mark the floor at the robot's start, reset this tool (Enter)
    2. drive straight ~2-3 m, stop, tape-measure the ACTUAL distance
    3. mm_per_tick_new = mm_per_tick_old * actual / reported

  wheel_base  (rotation):
    1. mark the robot's heading, reset (Enter)
    2. spin in place an exact whole number of turns (e.g. 5 x 360 = 1800 deg),
       stop on the mark
    3. wheel_base_new = wheel_base_old * reported_yaw / actual_yaw

Cumulative yaw is unwrapped so multi-turn spins read 720, 1080, ... not -0.
"""
import math
import sys
import threading

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import tf_transformations as tft

# Current (692) values, for the correction arithmetic printout.
OLD_MM_PER_TICK = 0.401635
OLD_WHEEL_BASE = 0.23842


class Calib(Node):
    def __init__(self):
        super().__init__('calibrate_odom')
        self.x0 = self.y0 = None
        self.last_yaw = None
        self.cum_yaw = 0.0
        self.x = self.y = 0.0
        self.create_subscription(Odometry, '/odom', self.cb, 50)

    def cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        if self.x0 is None:
            self.x0, self.y0, self.last_yaw = p.x, p.y, yaw
        self.x, self.y = p.x, p.y
        d = yaw - self.last_yaw
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        self.cum_yaw += d
        self.last_yaw = yaw

    def reset(self):
        self.x0, self.y0 = self.x, self.y
        self.last_yaw = None
        self.cum_yaw = 0.0

    def dist_m(self):
        if self.x0 is None:
            return 0.0
        return math.hypot(self.x - self.x0, self.y - self.y0)


def input_thread(node, stop):
    while not stop.is_set():
        try:
            input()
        except EOFError:
            return
        node.reset()
        print('  --- reset ---')


def main():
    rclpy.init()
    node = Calib()
    stop = threading.Event()
    t = threading.Thread(target=input_thread, args=(node, stop), daemon=True)
    t.start()
    print('Press Enter to reset; Ctrl-C to quit.\n')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            dist = node.dist_m()
            ydeg = math.degrees(node.cum_yaw)
            mmpt = OLD_MM_PER_TICK  # shown so you can plug in the measured actual
            sys.stdout.write(
                f'\rstraight = {dist:6.3f} m   yaw = {ydeg:+8.2f} deg    '
                f'(mm_per_tick={mmpt:.6f}, wheel_base={OLD_WHEEL_BASE:.5f})   ')
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        node.destroy_node()
        rclpy.shutdown()
        print()


if __name__ == '__main__':
    main()
