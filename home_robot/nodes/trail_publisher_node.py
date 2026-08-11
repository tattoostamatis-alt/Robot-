#!/usr/bin/env python3
"""Breadcrumb trail of where the robot has driven, for RViz.

Built from /amcl_pose (the localized pose) rather than raw odometry, so the
trail matches what AMCL believes the true path was, not odom drift. Points
are appended only after the robot has moved min_distance, so a stationary
robot doesn't fill the buffer with noise.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path


class TrailPublisherNode(Node):
    def __init__(self):
        super().__init__('trail_publisher_node')
        self.declare_parameter('min_distance', 0.10)
        self.declare_parameter('max_points', 5000)
        self.min_distance = self.get_parameter('min_distance').value
        self.max_points   = self.get_parameter('max_points').value

        self._path = Path()
        self._path.header.frame_id = 'map'
        self._last_xy = None

        # TRANSIENT_LOCAL so RViz gets the accumulated trail immediately on
        # connect, instead of waiting for the next /amcl_pose update.
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._pub = self.create_publisher(Path, '/robot_trail', qos)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._on_pose, 10)
        self.get_logger().info('Trail publisher ready')

    def _on_pose(self, msg: PoseWithCovarianceStamped):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if self._last_xy is not None:
            dist = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            if dist < self.min_distance:
                return
        self._last_xy = (x, y)

        pose = PoseStamped()
        pose.header = msg.header
        pose.pose   = msg.pose.pose
        self._path.poses.append(pose)
        if len(self._path.poses) > self.max_points:
            self._path.poses = self._path.poses[-self.max_points:]

        self._path.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._path)


def main():
    rclpy.init()
    node = TrailPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
