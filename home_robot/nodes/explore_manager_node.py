#!/usr/bin/env python3
"""Starts/stops explore_lite on demand via Bool topic from llm_bridge."""
import signal
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
        # start_new_session puts `ros2 run` and the explore node it spawns in
        # their own process GROUP, so _stop can signal BOTH. Signalling only the
        # `ros2` parent left the actual explore node alive and still driving.
        self._proc = subprocess.Popen([
            'ros2', 'run', 'explore_lite', 'explore',
            '--ros-args', '--params-file', self._config,
            '-r', '/tf:=tf', '-r', '/tf_static:=tf_static',
        ], start_new_session=True)
        self.get_logger().info(f'Explore started (pgid={self._proc.pid})')

    def _stop(self):
        """Stop explore for real, and reap it.

        The old version called terminate() on the parent only, never waited,
        and then dropped the handle (`self._proc = None`) — so the child was
        never reaped (zombie), the actual explore node could outlive the
        signal, and because _start tests `self._proc.poll()` on a handle that
        no longer existed, the next start happily launched a SECOND explore.
        """
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            if proc is not None:
                proc.wait()                     # reap an already-exited child
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=5.0)
            self.get_logger().info('Explore stopped')
            return
        except subprocess.TimeoutExpired:
            pass
        self.get_logger().warn('Explore ignored SIGTERM — killing')
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.get_logger().error('Explore process would not die')

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
