#! /usr/bin/env python3
# Adapted from open-navigation/opennav_amd_demonstrations honeybee_watchdogs
# (joystick_estop.py), Copyright 2024 Open Navigation LLC, Apache-2.0.
# Retargeted for the PS5 DualSense + home_robot cmd_vel_safe topology: the
# roomba_driver enforces the stop (teleop bypasses obstacle_safety straight to
# cmd_vel_safe, so the driver is the only guaranteed choke point). See
# project_robot_teleop_cmdvel_safe / project_robot_other_robots_survey memories.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Sticky soft e-stop from the PS5 controller.

Press the e-stop button (Circle by default) and the robot latches STOPPED: the
roomba_driver forces the wheels to zero and ignores cmd_vel until reset --
unlike the teleop R1 dead-man, this state persists even if the controller
disconnects. Reset requires a deliberate two-button chord (Share+Options) so it
can't clear accidentally.

The `emergency_stop` Bool is published latched (transient-local) so the driver
picks up the current state even if it starts after this node.

DualSense button indices (hid-playstation, 13 buttons; teleop uses 4=L1/5=R1):
  0 Cross  1 Circle  2 Triangle  3 Square  4 L1  5 R1  6 L2  7 R2
  8 Share  9 Options  10 PS  11 L3  12 R3

  ros2 run home_robot joystick_estop_node.py
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


class JoyEStop(Node):

    def __init__(self):
        super().__init__('joystick_estop')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('estop_button', 1)          # Circle (red)
        self.declare_parameter('reset_combo', [8, 9])      # Share + Options together
        joy_topic = self.get_parameter('joy_topic').value
        self.estop_button = self.get_parameter('estop_button').value
        self.reset_combo = list(self.get_parameter('reset_combo').value)

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Bool, 'emergency_stop', latched)

        self.estopped = False
        self._publish()  # latch the initial (released) state

        self.create_subscription(Joy, joy_topic, self._joy_cb, 10)
        self.get_logger().info(
            f'Soft e-stop watching {joy_topic}: button {self.estop_button} to STOP, '
            f'buttons {self.reset_combo} together to reset.')

    def _pressed(self, buttons, idx):
        return 0 <= idx < len(buttons) and buttons[idx] == 1

    def _joy_cb(self, msg: Joy):
        buttons = msg.buttons
        if not self.estopped:
            if self._pressed(buttons, self.estop_button):
                self.estopped = True
                self.get_logger().warn('E-STOP requested — robot stopped until reset!')
                self._publish()
        else:
            # only a deliberate chord (all reset buttons at once) can release it
            if self.reset_combo and all(self._pressed(buttons, b) for b in self.reset_combo):
                self.estopped = False
                self.get_logger().info('E-stop reset — robot may move again.')
                self._publish()

    def _publish(self):
        self.pub.publish(Bool(data=self.estopped))


def main():
    rclpy.init()
    node = JoyEStop()
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
