#!/usr/bin/env python3
"""Forward obstacle safety stop — consumes object_detector.py's detections.

Nav2's voxel_layer already fuses /scan + the D435 point cloud for
goal-directed navigation, but that costmap is only consulted by Nav2's own
controller — raw cmd_vel from llm_bridge_node.py/teleop bypasses it entirely.
This node is an independent safety net for exactly that gap: it passes
cmd_vel straight through to cmd_vel_safe, except it zeroes the forward
component when something is within `safety_distance` ahead (still allows
turning/reversing away from it).

Previously this node ran its own separate YOLO pass on the same camera feed
as object_detector.py — two full CPU-bound YOLO instances inferencing the
same frames (no GPU/NPU path available from Python, see project memory).
Now it just reads the 'detected_objects' topic object_detector.py already
publishes (which carries pixel boxes + a whole-box median distance for
exactly this purpose) — same safety behavior, roughly half the CPU/RAM.

Fail-safe: if object_detector.py stops publishing (crashed, use_camera off,
etc.) for longer than `detection_timeout`, forward motion is blocked rather
than silently assumed clear — this Roomba has no physical bump/cliff
sensors, so software is the only obstacle protection it has.
"""
import json
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class ObstacleSafetyNode(Node):

    def __init__(self):
        super().__init__('obstacle_safety_node')

        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('center_fraction', 0.5)
        self.declare_parameter('detection_timeout', 2.0)

        self.safety_distance = self.get_parameter('safety_distance').value
        self.center_fraction = self.get_parameter('center_fraction').value
        self.detection_timeout = self.get_parameter('detection_timeout').value

        self.obstacle_blocking = False
        self.obstacle_info = ''
        self._last_detection_time = None

        self.create_subscription(String, 'detected_objects', self._detections_cb, 10)
        self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel_safe', 10)
        self.obstacle_pub = self.create_publisher(Bool, 'obstacle_safety/obstacle_detected', 1)
        self.distance_pub = self.create_publisher(Float32, 'obstacle_safety/min_distance', 1)

        self.create_timer(0.5, self._check_staleness)

    def _detections_cb(self, msg: String):
        self._last_detection_time = time.monotonic()
        try:
            objects = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        blocking = False
        min_distance = float('inf')
        info = ''

        for obj in objects:
            distance = obj.get('box_distance')
            img_w = obj.get('img_w')
            x1, x2 = obj.get('x1'), obj.get('x2')
            if distance is None or img_w is None or x1 is None or x2 is None:
                continue

            center_margin = (1.0 - self.center_fraction) / 2.0
            center_x_min = img_w * center_margin
            center_x_max = img_w * (1.0 - center_margin)
            box_center_x = (x1 + x2) / 2.0

            in_center = center_x_min <= box_center_x <= center_x_max
            if in_center and distance < self.safety_distance:
                blocking = True
                if distance < min_distance:
                    min_distance = distance
                    info = f"{obj.get('label', '?')} at {distance:.2f}m"

        self.obstacle_blocking = blocking
        self.obstacle_info = info

        self.obstacle_pub.publish(Bool(data=blocking))
        if min_distance != float('inf'):
            self.distance_pub.publish(Float32(data=min_distance))

    def _check_staleness(self):
        if self._last_detection_time is None:
            return
        age = time.monotonic() - self._last_detection_time
        if age > self.detection_timeout and not self.obstacle_blocking:
            self.obstacle_blocking = True
            self.obstacle_info = f'object_detector silent for {age:.1f}s (fail-safe)'
            self.get_logger().warn(
                'No detections from object_detector — blocking forward motion as a fail-safe.',
                throttle_duration_sec=5.0)

    def _cmd_vel_cb(self, msg: Twist):
        out = Twist()
        out.linear.x = msg.linear.x
        out.linear.y = msg.linear.y
        out.linear.z = msg.linear.z
        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y
        out.angular.z = msg.angular.z

        if self.obstacle_blocking and msg.linear.x > 0.0:
            out.linear.x = 0.0
            self.get_logger().info(
                f'Blocking forward motion: {self.obstacle_info}', throttle_duration_sec=1.0)

        self.cmd_vel_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleSafetyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
