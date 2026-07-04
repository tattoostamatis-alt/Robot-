#!/usr/bin/env python3
"""Broadcast odom->base_link TF from an Odometry topic.

Sim-only helper: gz DiffDrive publishes /odom reliably but its built-in TF
output was flaky over the ros_gz bridge (the costmaps couldn't find the 'odom'
frame during activation, so the whole nav lifecycle aborted on the 4 s bond
timeout). Re-deriving the transform from /odom here is deterministic and starts
as soon as /odom flows, well before Nav2 activates. On the real robot this TF
comes from the EKF instead — this node never runs on hardware.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        self.declare_parameter('odom_topic', 'odom')
        topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self._br = TransformBroadcaster(self)
        self.create_subscription(Odometry, topic, self._on_odom, 20)
        self.get_logger().info(f'Broadcasting odom->base_link TF from {topic}')

    def _on_odom(self, msg: Odometry):
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = msg.header.frame_id or 'odom'
        t.child_frame_id = msg.child_frame_id or 'base_link'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self._br.sendTransform(t)


def main():
    rclpy.init()
    node = OdomTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
