#!/usr/bin/env python3
"""Starts/stops explore_lite on demand via Bool topic from llm_bridge."""
import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from ament_index_python.packages import get_package_share_directory
import os


class ExploreManager(Node):
    def __init__(self):
        super().__init__('explore_manager')
        self._proc = None
        self._config = os.path.join(
            get_package_share_directory('home_robot'), 'config', 'explore.yaml')
        self.create_subscription(Bool, 'explore_command', self._cb, 10)
        self.get_logger().info('Explore manager ready')

    def _cb(self, msg: Bool):
        if msg.data:
            self._start()
        else:
            self._stop()

    def _start(self):
        if self._proc and self._proc.poll() is None:
            self.get_logger().info('Explore already running')
            return
        self._proc = subprocess.Popen([
            'ros2', 'run', 'explore_lite', 'explore',
            '--ros-args', '--params-file', self._config,
            '-r', '/tf:=tf', '-r', '/tf_static:=tf_static',
        ])
        self.get_logger().info(f'Explore started (pid={self._proc.pid})')

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self.get_logger().info('Explore stopped')
        self._proc = None

    def destroy_node(self):
        self._stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = ExploreManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
