#!/usr/bin/env python3
"""Restart the RealSense when it dies, because nothing else notices.

The D435 driver segfaults on roughly 2.5% of bringups (exit code -11, seen
again on 2026-08-01). When it does, the launch reports the death once and then
everything downstream simply goes quiet: object_detector logs "waiting for
camera topics" forever, the 3D tab shows an empty cloud, distance_to answers
"δεν βλέπω", and the robot tells you there is nothing in front of it. The robot
is blind and every symptom points somewhere else — the camera is the one part
that never says it is missing.

So: watch the colour stream. If frames stop for `timeout` seconds, relaunch the
camera alone (which is the fix that has always worked by hand) and say so
loudly on /rosout, where the dashboard's log tab will show it.

Deliberately narrow:
  * Only the camera is restarted, never the stack. A stack restart would drop
    localization and any goal in flight to fix a camera.
  * Restarts are rate-limited and capped. A camera that has been unplugged
    cannot be fixed by relaunching it 400 times, and the log noise would bury
    the reason.
  * It waits for the FIRST frame before arming. A camera still starting up is
    not a camera that died.
"""
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

DEFAULT_TOPIC = '/camera/camera/color/image_raw'


class CameraWatchdog(Node):
    def __init__(self):
        super().__init__('camera_watchdog')

        self.declare_parameter('topic', DEFAULT_TOPIC)
        # Frames arrive at 30 Hz, so 8 s of nothing is ~240 missed frames. Long
        # enough that a USB hiccup or a busy CPU cannot trip it, short enough
        # that a blind robot is not left driving for a minute.
        self.declare_parameter('timeout', 8.0)
        # The driver takes ~10 s to enumerate and start streaming; without this
        # the watchdog would fire again while its own restart was still coming up.
        self.declare_parameter('restart_grace', 30.0)
        self.declare_parameter('max_restarts', 3)
        self.declare_parameter('enable_restart', True)
        # Same arguments bringup uses with use_perception:=true. Kept here
        # rather than read from the launch because this has to work when the
        # launch that owned the camera is the thing that lost it.
        self.declare_parameter('launch_args', [
            'enable_color:=true', 'enable_depth:=true',
            'enable_infra1:=false', 'enable_infra2:=false',
            'align_depth.enable:=true', 'pointcloud.enable:=true',
        ])

        self.topic = self.get_parameter('topic').value
        self.timeout = float(self.get_parameter('timeout').value)
        self.grace = float(self.get_parameter('restart_grace').value)
        self.max_restarts = int(self.get_parameter('max_restarts').value)
        self.enable_restart = bool(self.get_parameter('enable_restart').value)

        self._last_frame = None        # None = never saw one; do not arm yet
        self._restarts = 0
        self._blocked_until = 0.0
        self._reported_dead = False

        # Images are BEST_EFFORT: a RELIABLE subscription matches nothing and
        # the watchdog would declare a healthy camera dead, every time.
        self.create_subscription(
            Image, self.topic, self._on_frame,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.create_timer(1.0, self._check)

        self.get_logger().info(
            f'Camera watchdog armed on {self.topic} '
            f'(timeout {self.timeout:.0f}s, restart={"on" if self.enable_restart else "off"})')

    def _on_frame(self, _msg):
        if self._last_frame is None:
            self.get_logger().info('Camera is streaming')
        elif self._reported_dead:
            self.get_logger().info('Camera is back')
            self._reported_dead = False
        self._last_frame = time.monotonic()

    def _check(self):
        if self._last_frame is None:
            return                      # never started; not our problem to fix
        now = time.monotonic()
        if now - self._last_frame < self.timeout:
            return
        if now < self._blocked_until:
            return                      # a restart is still coming up

        gap = now - self._last_frame
        if not self._reported_dead:
            self.get_logger().error(
                f'No camera frames for {gap:.0f}s on {self.topic} — the robot '
                'is BLIND. Object detection, the 3D view and distance answers '
                'are all dead until this comes back.')
            self._reported_dead = True

        if not self.enable_restart:
            return
        if self._restarts >= self.max_restarts:
            self.get_logger().error(
                f'Camera did not recover after {self._restarts} restarts — '
                'giving up. This is hardware: check the USB cable and the hub.')
            self._blocked_until = now + 3600.0
            return

        self._restarts += 1
        self._blocked_until = now + self.grace
        self.get_logger().warn(
            f'Restarting the camera (attempt {self._restarts}/{self.max_restarts})')
        self._restart_camera()

    def _restart_camera(self):
        args = list(self.get_parameter('launch_args').value or [])
        try:
            # start_new_session: this outlives the request that triggered it,
            # and must not die with this node if the node is next to be killed.
            subprocess.Popen(
                ['ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
                 'camera_name:=camera', 'camera_namespace:=camera'] + args,
                start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().error(f'Could not relaunch the camera: {exc!r}')


def main():
    rclpy.init()
    node = CameraWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
