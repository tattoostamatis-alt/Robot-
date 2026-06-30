#!/usr/bin/env python3
"""Saves AMCL pose to disk and restores it on startup.

On startup: reads last_pose.yaml and publishes to /initialpose so AMCL
starts from the last known position instead of needing manual 2D Pose Estimate.
Retries every second until AMCL subscribes to /initialpose (i.e. is active),
then confirms by waiting for an /amcl_pose message.

On shutdown: writes the last received AMCL pose to last_pose.yaml.
Also saves every `save_interval` seconds while running.
"""

import math
import os

import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node


POSE_DIR = os.path.expanduser('~/.ros')


def _pose_file_for(map_yaml: str) -> str:
    """Per-map pose file so the saved pose of one map never restores onto another.

    last_amcl_pose_<mapstem>.yaml — e.g. maps/kela.yaml -> last_amcl_pose_kela.yaml.
    Falls back to the legacy shared file when no map path is given.
    """
    if not map_yaml:
        return os.path.join(POSE_DIR, 'last_amcl_pose.yaml')
    stem = os.path.splitext(os.path.basename(map_yaml))[0]
    return os.path.join(POSE_DIR, f'last_amcl_pose_{stem}.yaml')


def _quat_to_yaw(z, w):
    return 2.0 * math.atan2(z, w)


def _yaw_to_quat(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class PoseSaverNode(Node):
    def __init__(self):
        super().__init__('pose_saver_node')
        self.declare_parameter('save_interval', 10.0)
        # publish_delay kept for backwards compat but no longer used as a
        # one-shot timer — we now poll until AMCL is subscribed instead.
        self.declare_parameter('publish_delay', 8.0)
        # Path to the active map's .yaml — used to pick a per-map pose file.
        self.declare_parameter('map_yaml', '')
        # When False, do NOT publish the saved pose on startup — let
        # global_localizer find the position from scratch instead. This is the
        # default now: the user starts the robot anywhere on the map and it
        # self-localizes, so a stale saved pose must not be restored on top of
        # the global match (the two would fight over /initialpose). The node
        # still SAVES the live AMCL pose periodically (harmless, useful for
        # debugging / re-enabling restore later).
        self.declare_parameter('restore_pose', False)

        self._interval = self.get_parameter('save_interval').value
        self._restore_pose = self.get_parameter('restore_pose').value
        self._pose_file = _pose_file_for(self.get_parameter('map_yaml').value)
        self._last_pose = None
        self._restored = False
        self._confirmed = False   # True once /amcl_pose comes back after restore

        self._pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._on_pose, 10)
        self.create_timer(self._interval, self._save)

        # Poll every 1s until AMCL is ready, then publish the saved pose —
        # only if restore_pose is enabled; otherwise global_localizer owns
        # the initial pose.
        if self._restore_pose:
            self._restore_timer = self.create_timer(1.0, self._try_restore)
        else:
            self._restored = True
            self.get_logger().info(
                'restore_pose=False — global_localizer will set the initial '
                'pose; pose_saver will only save the live pose')

        self.get_logger().info(f'pose_saver_node started — pose file: {self._pose_file}')

    def _on_pose(self, msg: PoseWithCovarianceStamped):
        self._last_pose = msg
        if not self._confirmed and self._restored:
            self._confirmed = True
            self.get_logger().info('AMCL confirmed initial pose — localization active')

    def _try_restore(self):
        if self._restored:
            # Already published; cancel this timer
            self._restore_timer.cancel()
            return

        # Wait until AMCL has subscribed to /initialpose (= AMCL is active)
        if self._pub.get_subscription_count() == 0:
            return

        if not os.path.exists(self._pose_file):
            self.get_logger().info('No saved pose — will use global localization fallback')
            self._restored = True
            self._restore_timer.cancel()
            return

        try:
            with open(self._pose_file) as f:
                data = yaml.safe_load(f)
            x   = float(data['x'])
            y   = float(data['y'])
            yaw = float(data['yaw'])
        except Exception as e:
            self.get_logger().warn(f'Could not read saved pose: {e}')
            self._restored = True
            self._restore_timer.cancel()
            return

        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x    = x
        msg.pose.pose.position.y    = y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = _yaw_to_quat(yaw)
        # Moderate initial uncertainty — let AMCL refine it
        msg.pose.covariance[0]  = 0.25   # x
        msg.pose.covariance[7]  = 0.25   # y
        msg.pose.covariance[35] = 0.07   # yaw

        self._pub.publish(msg)
        self._restored = True
        self._restore_timer.cancel()
        self.get_logger().info(
            f'Published saved pose: x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f}°')

    def _save(self):
        if self._last_pose is None:
            return
        p = self._last_pose.pose.pose
        data = {
            'x':   p.position.x,
            'y':   p.position.y,
            'yaw': _quat_to_yaw(p.orientation.z, p.orientation.w),
        }
        try:
            with open(self._pose_file, "w") as f:
                yaml.dump(data, f)
        except Exception as e:
            self.get_logger().warn(f'Failed to save pose: {e}')


def main():
    rclpy.init()
    node = PoseSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save()   # save on shutdown
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
