#!/usr/bin/env python3
"""Prevent an idle PS5 controller from fighting other cmd_vel_safe sources."""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from home_robot.joy_cmd_gate import JoyCommandGate


class JoyCmdGateNode(Node):
    def __init__(self):
        super().__init__('joy_cmd_gate')
        self.declare_parameter('input_topic', '/cmd_vel_joy_raw')
        self.declare_parameter('output_topic', '/cmd_vel_safe')
        self.declare_parameter('disconnect_timeout', 0.30)
        self._timeout = float(self.get_parameter('disconnect_timeout').value)
        self._gate = JoyCommandGate()
        self._last_rx = time.monotonic()
        self._pub = self.create_publisher(
            Twist, self.get_parameter('output_topic').value, 10)
        self.create_subscription(
            Twist, self.get_parameter('input_topic').value, self._cb, 10)
        self.create_timer(0.05, self._watchdog)
        self.get_logger().info(
            'PS5 command gate ready: idle zero autorepeat is suppressed')

    @staticmethod
    def _values(msg):
        return (msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z)

    def _cb(self, msg):
        self._last_rx = time.monotonic()
        if self._gate.should_forward(self._values(msg)):
            self._pub.publish(msg)

    def _watchdog(self):
        if (time.monotonic() - self._last_rx > self._timeout
                and self._gate.stop_if_active()):
            self._pub.publish(Twist())


def main():
    rclpy.init()
    node = JoyCmdGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
