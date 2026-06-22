#!/usr/bin/env python3
"""YOLO + D435 forward obstacle safety stop.

Nav2's voxel_layer already fuses /scan + the D435 point cloud for
goal-directed navigation, but that costmap is only consulted by Nav2's own
controller — raw cmd_vel from llm_bridge_node.py/teleop bypasses it entirely.
This node is an independent safety net for exactly that gap: it passes
cmd_vel straight through to cmd_vel_safe, except it zeroes the forward
component when YOLO sees something within `safety_distance` ahead (still
allows turning/reversing away from it).

Confirmed (project memory, 2026-06-20): running YOLO on the NPU from
Python isn't possible here — onnxruntime's VitisAI EP segfaults on an ABI
mismatch with AMD's closed-source provider, a dead end without AMD's
private build flags. CPU is the only working path (~40+ FPS for yolo11n
on this machine, plenty for this throttled check).
"""
import os

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

from ultralytics import YOLO

DEFAULT_MODEL_PATH = os.path.expanduser('~/robot_ws/src/home_robot/yolo11n.pt')


class ObstacleSafetyNode(Node):

    def __init__(self):
        super().__init__('obstacle_safety_node')

        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('center_fraction', 0.5)
        self.declare_parameter('process_every_n', 2)
        self.declare_parameter('publish_annotated', True)

        self.device = self.get_parameter('device').value
        self.confidence = self.get_parameter('confidence').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.center_fraction = self.get_parameter('center_fraction').value
        self.process_every_n = self.get_parameter('process_every_n').value
        self.publish_annotated = self.get_parameter('publish_annotated').value

        self.get_logger().info(f'Loading YOLO model "{self.get_parameter("model_path").value}"...')
        self.model = YOLO(self.get_parameter('model_path').value)
        self.get_logger().info('YOLO model loaded.')

        self.bridge = CvBridge()
        self.latest_depth = None
        self.obstacle_blocking = False
        self.obstacle_info = ''
        self._frame_count = 0

        self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self._depth_cb, 1)
        self.create_subscription(
            Image, self.get_parameter('color_topic').value, self._color_cb, 1)
        self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel_safe', 10)
        self.obstacle_pub = self.create_publisher(Bool, 'obstacle_safety/obstacle_detected', 1)
        self.distance_pub = self.create_publisher(Float32, 'obstacle_safety/min_distance', 1)
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(Image, 'obstacle_safety/annotated_image', 1)

    def _depth_cb(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

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

    def _color_cb(self, msg: Image):
        self._frame_count += 1
        if self._frame_count % self.process_every_n != 0:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        depth = self.latest_depth
        height, width = frame.shape[:2]

        results = self.model.predict(
            frame, conf=self.confidence, device=self.device, verbose=False)[0]

        center_margin = (1.0 - self.center_fraction) / 2.0
        center_x_min = width * center_margin
        center_x_max = width * (1.0 - center_margin)

        blocking = False
        min_distance = float('inf')
        info = ''

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_center_x = (x1 + x2) / 2.0
            label = self.model.names[int(box.cls[0])]

            distance = self._box_distance(depth, x1, y1, x2, y2, width, height)
            if distance is None:
                continue

            in_center = center_x_min <= box_center_x <= center_x_max
            if in_center and distance < self.safety_distance:
                blocking = True
                if distance < min_distance:
                    min_distance = distance
                    info = f'{label} at {distance:.2f}m'

        self.obstacle_blocking = blocking
        self.obstacle_info = info

        self.obstacle_pub.publish(Bool(data=blocking))
        if min_distance != float('inf'):
            self.distance_pub.publish(Float32(data=min_distance))

        if self.publish_annotated:
            annotated = results.plot()
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.annotated_pub.publish(out_msg)

    @staticmethod
    def _box_distance(depth, x1, y1, x2, y2, width, height):
        if depth is None:
            return None
        ix1, iy1 = max(int(x1), 0), max(int(y1), 0)
        ix2, iy2 = min(int(x2), width), min(int(y2), height)
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        if depth.shape[0] != height or depth.shape[1] != width:
            return None

        region = depth[iy1:iy2, ix1:ix2].astype(np.float32)
        valid = region[region > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0  # mm -> m


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
