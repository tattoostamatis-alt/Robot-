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
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

DEFAULT_TOPIC = '/camera/camera/color/image_raw'
DEFAULT_DEPTH_TOPIC = '/camera/camera/depth/image_rect_raw'


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
        # ‼️ How long to wait for the FIRST frame before treating the camera as
        # dead. The first version of this node refused to arm until it had seen
        # a frame, so that a slow boot could not be mistaken for a death — and
        # that made it useless for the exact failure it was written for: the
        # driver segfaults DURING startup (exit -11), before it has ever
        # published anything. Caught on 2026-08-01 within minutes of shipping
        # it, on the very next bringup. Generous, because a cold start under
        # load plus USB enumeration is genuinely slow.
        self.declare_parameter('startup_timeout', 45.0)
        # How long the camera has to keep streaming after a restart before that
        # restart stops counting against max_restarts. Comfortably longer than
        # the grace period, so a restart that only half worked does not buy
        # itself a fresh budget.
        self.declare_parameter('restart_count_reset', 300.0)
        # Same arguments bringup uses with use_perception:=true. Kept here
        # rather than read from the launch because this has to work when the
        # launch that owned the camera is the thing that lost it.
        self.declare_parameter('launch_args', [
            'enable_color:=true', 'enable_depth:=true',
            'enable_infra1:=false', 'enable_infra2:=false',
            'align_depth.enable:=true', 'pointcloud.enable:=true',
        ])

        # ‼️ The depth stream can die on its own while colour keeps streaming at
        # 30 Hz — measured live on 2026-08-04. Everything above watched colour
        # only, so the watchdog stayed happy while the 3D tab sat empty, the
        # pointcloud never published and the detector answered [] forever. The
        # camera looks healthy from the outside; only depth is gone.
        # The fix is NOT a relaunch: toggling enable_depth off and back on
        # brought depth (and the pointcloud) back at 25 Hz in seconds, without
        # dropping the colour stream that object detection is riding on.
        self.declare_parameter('depth_topic', DEFAULT_DEPTH_TOPIC)
        # Depth runs at 30 Hz like colour. 12 s is longer than the colour
        # timeout on purpose: a depth stall is repaired in place, so there is no
        # cost to being sure, and the pointcloud filter genuinely pauses depth
        # for a moment when the 3D tab flips the parameter.
        self.declare_parameter('depth_timeout', 12.0)
        self.declare_parameter('depth_grace', 20.0)
        self.declare_parameter('enable_depth_recover', True)

        self.topic = self.get_parameter('topic').value
        self.timeout = float(self.get_parameter('timeout').value)
        self.grace = float(self.get_parameter('restart_grace').value)
        self.max_restarts = int(self.get_parameter('max_restarts').value)
        self.enable_restart = bool(self.get_parameter('enable_restart').value)

        self.startup_timeout = float(self.get_parameter('startup_timeout').value)
        self.reset_after = float(self.get_parameter('restart_count_reset').value)
        self._last_frame = None        # None = never saw one
        self._started_at = time.monotonic()
        self._restarts = 0
        self._last_restart_at = 0.0
        self._blocked_until = 0.0
        self._reported_dead = False

        # Images are BEST_EFFORT: a RELIABLE subscription matches nothing and
        # the watchdog would declare a healthy camera dead, every time.
        self.create_subscription(
            Image, self.topic, self._on_frame,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.create_timer(1.0, self._check)

        self.depth_topic = self.get_parameter('depth_topic').value
        self.depth_timeout = float(self.get_parameter('depth_timeout').value)
        self.depth_grace = float(self.get_parameter('depth_grace').value)
        self.depth_recover = bool(self.get_parameter('enable_depth_recover').value)
        self._last_depth = None
        self._depth_blocked_until = 0.0
        self._depth_reported = False
        self._depth_on_at = 0.0         # when to flip enable_depth back on
        self._depth_cli = self.create_client(
            SetParameters, '/camera/camera/set_parameters')
        self.create_subscription(
            Image, self.depth_topic, self._on_depth,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.create_timer(1.0, self._check_depth)

        self.get_logger().info(
            f'Camera watchdog armed on {self.topic} '
            f'(timeout {self.timeout:.0f}s, restart={"on" if self.enable_restart else "off"})')

    def _on_frame(self, _msg):
        # ‼️ The first branch used to be an if/elif, so a camera that NEVER
        # streamed and was then brought back by a restart hit the first branch
        # only — `_reported_dead` stayed True forever. Two consequences, both
        # seen in the log tab on 2026-08-05: the red "the robot is BLIND" line
        # was never followed by anything saying it came back (17 s later), and
        # a SECOND failure would have been reported by nobody, because the flag
        # that guards that error was already set.
        first = self._last_frame is None
        self._last_frame = time.monotonic()
        if first:
            self.get_logger().info('Camera is streaming')
        if self._reported_dead:
            self._reported_dead = False
            if not first:
                self.get_logger().info('Camera is back')
        # A restart that worked buys back the budget. Without this the counter
        # only ever climbs: three unrelated hiccups spread over a day and the
        # watchdog gives up for an hour on a camera that recovered every time.
        if self._restarts and (self._last_frame - self._last_restart_at
                               > self.reset_after):
            self.get_logger().info(
                f'Camera has streamed for {self.reset_after:.0f}s since the '
                f'last restart — clearing the count (was {self._restarts})')
            self._restarts = 0

    def _check(self):
        now = time.monotonic()
        if self._last_frame is None:
            # Never streamed. Either still enumerating, or it segfaulted on the
            # way up — which is the common one, so it cannot be ignored.
            gap = now - self._started_at
            if gap < self.startup_timeout:
                return
            never_started = True
        else:
            gap = now - self._last_frame
            if gap < self.timeout:
                return
            never_started = False

        if now < self._blocked_until:
            return                      # a restart is still coming up

        if not self._reported_dead:
            self.get_logger().error(
                (f'Camera never streamed in {gap:.0f}s' if never_started
                 else f'No camera frames for {gap:.0f}s')
                + f' on {self.topic} — the robot is BLIND. Object detection, '
                'the 3D view and distance answers are all dead until this '
                'comes back.')
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
        self._last_restart_at = now
        self._blocked_until = now + self.grace
        self.get_logger().warn(
            f'Restarting the camera (attempt {self._restarts}/{self.max_restarts})')
        self._restart_camera()

    def _on_depth(self, _msg):
        if self._depth_reported:
            self.get_logger().info('Depth is back')
            self._depth_reported = False
        self._last_depth = time.monotonic()

    def _check_depth(self):
        # Only meaningful once colour is up: a camera that never started is the
        # other check's problem, and both firing at once would fight each other.
        now = time.monotonic()
        if self._depth_on_at and now >= self._depth_on_at:
            self._depth_on_at = 0.0
            self._toggle_depth(True)
            return
        if self._last_frame is None:
            return
        if now - self._last_frame > self.timeout:
            return                      # whole camera is down, not just depth
        if self._last_depth is None:
            # Give depth the same startup allowance as the camera itself.
            if now - self._started_at < self.startup_timeout:
                return
            gap = now - self._started_at
        else:
            gap = now - self._last_depth
            if gap < self.depth_timeout:
                return
        if now < self._depth_blocked_until:
            return

        if not self._depth_reported:
            self.get_logger().error(
                f'No depth frames for {gap:.0f}s on {self.depth_topic} while '
                'colour is still streaming — the 3D tab, the pointcloud and '
                'every distance answer are dead. Colour-only looks healthy, '
                'which is why this needs saying out loud.')
            self._depth_reported = True

        if not self.depth_recover or not self._depth_cli.service_is_ready():
            return
        self._depth_blocked_until = now + self.depth_grace
        self.get_logger().warn('Restarting the depth stream (enable_depth off/on)')
        self._toggle_depth(False)
        # Turned back on from this same 1 Hz timer, not from a sleep: this runs
        # on the executor and blocking here would stall the colour check too.
        self._depth_on_at = now + 2.0

    def _toggle_depth(self, on: bool):
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name='enable_depth',
            value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=on),
        )]
        self._depth_cli.call_async(req)

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
