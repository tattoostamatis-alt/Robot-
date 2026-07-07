#!/usr/bin/env python3
"""
recovery_manager_node.py — Physical stuck detection + Nav2 backup/spin recovery.

Watches cmd_vel_smoothed (the commanded intent, upstream of collision_monitor)
against /odom. It must NOT watch cmd_vel_safe: when collision_monitor zeroes a
command it publishes zeros for stop_pub_timeout seconds and then goes silent
entirely (seen live 2026-07-06, saloni→kouzina doorway — Spin commanded wz=0.6
for 10s, cmd_vel_safe empty, robot frozen), so the motor-side topic carries no
evidence that motion is being asked for. If motion is commanded but the robot
doesn't move for `stuck_timeout` seconds it declares STUCK, cancels any active
NavigateToPose goal, and runs the stock Nav2 recovery behaviors first, in the
order that works in tight doorways (translation before rotation):
  1. Nav2 BackUp action — collision-checked reverse; rear is usually clear
  2. Nav2 DriveOnHeading action — collision-checked forward creep
  3. Direct creep+turn nudge on cmd_vel_safe — LAST resort only: bypasses the
     collision monitor, for the case where even the checked behaviors refuse
     (e.g. footprint pinched between doorframes)
  4. Nav2 Spin action (90° rotation) — last because its full-footprint sweep
     check is exactly what aborts in doorways
Up to `max_attempts` times, then declares FAILED and asks for help.

Publishes: recovery/status (std_msgs/String: idle/stuck/recovering/recovered/failed)
           speech_response (std_msgs/String) — picked up by tts_node if running
"""

import math
import threading
import time
from collections import deque
from action_msgs.srv import CancelGoal
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import BackUp, DriveOnHeading, Spin
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String, Bool
import rclpy

STATUS_IDLE       = 'idle'
STATUS_STUCK      = 'stuck'
STATUS_RECOVERING = 'recovering'
STATUS_RECOVERED  = 'recovered'
STATUS_FAILED     = 'failed'


class RecoveryManagerNode(Node):
    def __init__(self):
        super().__init__('recovery_manager_node')

        self.declare_parameter('stuck_timeout',    5.0)   # seconds with cmd_vel but no movement
        self.declare_parameter('min_displacement', 0.03)  # meters — less than this = not moving
        self.declare_parameter('cmd_threshold',    0.01)  # m/s or rad/s — above = motion commanded
        self.declare_parameter('max_attempts',     3)
        self.declare_parameter('backup_distance',  0.25)  # meters
        self.declare_parameter('backup_speed',     0.05)  # m/s
        self.declare_parameter('drive_distance',   0.15)  # meters — DriveOnHeading creep
        self.declare_parameter('drive_speed',      0.05)  # m/s
        self.declare_parameter('spin_angle',       1.571) # radians (π/2 = 90°)
        self.declare_parameter('enabled',          True)
        self.declare_parameter('nudge_linear_speed',  0.10)  # m/s — doorway pinch-point creep
        self.declare_parameter('nudge_angular_speed', 0.10)  # rad/s
        self.declare_parameter('nudge_duration',      1.2)   # seconds per phase (turn, then creep)

        self._stuck_timeout   = self.get_parameter('stuck_timeout').value
        self._min_disp        = self.get_parameter('min_displacement').value
        self._cmd_thr         = self.get_parameter('cmd_threshold').value
        self._max_attempts    = self.get_parameter('max_attempts').value
        self._backup_dist     = self.get_parameter('backup_distance').value
        self._backup_speed    = self.get_parameter('backup_speed').value
        self._drive_dist      = self.get_parameter('drive_distance').value
        self._drive_speed     = self.get_parameter('drive_speed').value
        self._spin_angle      = self.get_parameter('spin_angle').value
        self._enabled         = self.get_parameter('enabled').value
        self._nudge_linear    = self.get_parameter('nudge_linear_speed').value
        self._nudge_angular   = self.get_parameter('nudge_angular_speed').value
        self._nudge_duration  = self.get_parameter('nudge_duration').value

        # Sliding window: deque of (timestamp_sec, x, y)
        self._positions: deque = deque(maxlen=40)   # ~20s at 0.5Hz check
        self._cmd_active = False   # True when cmd_vel_smoothed magnitude > threshold
        self._cmd_active_since: float | None = None
        self._cmd_last_rx: float | None = None  # last cmd_vel_smoothed arrival
        self._status = STATUS_IDLE
        self._lock = threading.Lock()

        # Action clients — the stock nav2_behaviors servers
        self._backup_ac = ActionClient(self, BackUp,         'backup')
        self._drive_ac  = ActionClient(self, DriveOnHeading, 'drive_on_heading')
        self._spin_ac   = ActionClient(self, Spin,           'spin')
        # Cancel any in-flight NavigateToPose via the action's cancel service.
        # An empty CancelGoal request (zero id + zero stamp) cancels all goals;
        # recovery didn't send the goal itself so it has no goal handle to use.
        self._nav_cancel_cli = self.create_client(
            CancelGoal, 'navigate_to_pose/_action/cancel_goal')

        # Publishers
        self._status_pub   = self.create_publisher(String, 'recovery/status',  10)
        self._speech_pub   = self.create_publisher(String, 'speech_response',  10)
        # Same bypass teleop uses (project_robot_teleop_cmdvel_safe): publish
        # straight onto cmd_vel_safe, past collision_monitor, for the nudge.
        self._nudge_pub    = self.create_publisher(Twist,  'cmd_vel_safe',     10)

        # Manual trigger (Bool True = force a recovery attempt)
        self.create_subscription(Bool, 'recovery/trigger', self._on_trigger, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        # Teleop that remaps straight onto cmd_vel_safe bypasses this on
        # purpose: no auto-recovery fighting the joystick.
        self.create_subscription(Twist, 'cmd_vel_smoothed', self._on_cmd_vel, 10)

        self.create_timer(0.5, self._check_stuck)

        self.get_logger().info(
            f'Recovery manager ready — stuck_timeout={self._stuck_timeout}s '
            f'min_displacement={self._min_disp}m'
        )

    # ── Subscriptions ────────────────────────────────────────────────

    def _on_odom(self, msg: Odometry):
        t = self.get_clock().now().nanoseconds / 1e9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self._positions.append((t, x, y))

    def _on_cmd_vel(self, msg: Twist):
        mag = abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z)
        active = mag > self._cmd_thr
        now = self.get_clock().now().nanoseconds / 1e9
        with self._lock:
            self._cmd_last_rx = now
            if active and not self._cmd_active:
                self._cmd_active = True
                self._cmd_active_since = now
            elif not active:
                self._cmd_active = False
                self._cmd_active_since = None

    def _on_trigger(self, msg: Bool):
        if msg.data and self._status == STATUS_IDLE:
            self.get_logger().info('Manual recovery trigger received')
            threading.Thread(target=self._run_recovery, args=(1,), daemon=True).start()

    # ── Stuck detection ──────────────────────────────────────────────

    def _check_stuck(self):
        if not self._enabled or self._status != STATUS_IDLE:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        with self._lock:
            # If the publisher went quiet with a non-zero command as its last
            # message (goal aborted mid-motion), don't let a stale "active"
            # flag fire a bogus STUCK minutes later.
            if self._cmd_last_rx is not None and now - self._cmd_last_rx > 1.0:
                self._cmd_active = False
                self._cmd_active_since = None
            active = self._cmd_active
            since  = self._cmd_active_since

        if not active or since is None:
            return

        elapsed = now - since
        if elapsed < self._stuck_timeout:
            return

        # Compute displacement over the stuck window
        window = [(t, x, y) for t, x, y in self._positions if t >= since]
        if len(window) < 2:
            return

        xs = [p[1] for p in window]
        ys = [p[2] for p in window]
        disp = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)

        if disp < self._min_disp:
            self.get_logger().warn(
                f'STUCK detected — {elapsed:.1f}s with cmd_vel, displacement={disp:.3f}m'
            )
            self._status = STATUS_STUCK
            self._publish_status(STATUS_STUCK)
            threading.Thread(
                target=self._run_recovery, args=(self._max_attempts,), daemon=True
            ).start()

    # ── Recovery sequence ────────────────────────────────────────────

    def _run_recovery(self, attempts_left: int):
        self._status = STATUS_RECOVERING
        self._speak('Κόλλησα, προσπαθώ να ξεκολλήσω.')

        if attempts_left <= 0:
            self._status = STATUS_FAILED
            self._publish_status(STATUS_FAILED)
            self._speak('Δεν μπορώ να ξεκολλήσω. Χρειάζομαι βοήθεια.')
            return

        # Cancel any active navigation goal
        self._cancel_navigation()

        # Stock Nav2 behaviors first, translation before rotation: BackUp and
        # DriveOnHeading collision-check only the straight-line motion, so they
        # work in doorways where Spin's full-footprint 90° sweep check aborts
        # ("Collision Ahead - Exiting Spin"). The raw creep+turn nudge (proven
        # by hand on kela3, 2026-07-06) bypasses the collision monitor and is
        # kept strictly as the step before giving up; Spin goes last.
        steps = [
            ('BackUp',         self._do_backup),
            ('DriveOnHeading', self._do_drive_on_heading),
            ('nudge',          self._do_nudge),
            ('Spin',           self._do_spin),
        ]
        for name, step in steps:
            self.get_logger().info(f'Recovery step: {name}')
            step()
            time.sleep(1.0)
            if self._has_moved_recently(window=5.0, threshold=0.01):
                self._recovered()
                return

        self.get_logger().warn(f'Still stuck, {attempts_left - 1} attempt(s) left')
        self._run_recovery(attempts_left - 1)

    def _recovered(self):
        self._status = STATUS_RECOVERED
        self._publish_status(STATUS_RECOVERED)
        self._speak('Ξεκόλλησα!')
        self.get_logger().info('Recovery succeeded')
        # Reset to IDLE after a pause
        time.sleep(2.0)
        self._status = STATUS_IDLE
        self._cmd_active = False
        self._cmd_active_since = None

    def _await_future(self, future, timeout_sec):
        """Wait for an async future from this worker thread WITHOUT spinning —
        the node's main executor (rclpy.spin in main) services the callbacks
        that complete it. Spinning here would attach the node to a second
        executor and the future would never complete. Returns None on timeout."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)
        return future.result() if future.done() else None

    def _cancel_navigation(self):
        if not self._nav_cancel_cli.wait_for_service(timeout_sec=2.0):
            return
        # Empty request → cancel all active goals on navigate_to_pose.
        future = self._nav_cancel_cli.call_async(CancelGoal.Request())
        self._await_future(future, 3.0)

    def _do_nudge(self):
        self.get_logger().info('Nudge: turn then creep')
        self._publish_twist_for(self._nudge_duration, angular_z=self._nudge_angular)
        self._publish_twist_for(self._nudge_duration, linear_x=self._nudge_linear)
        self.get_logger().info('Nudge done')

    def _publish_twist_for(self, seconds: float, linear_x: float = 0.0, angular_z: float = 0.0):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        rate_hz = 10.0
        for _ in range(int(seconds * rate_hz)):
            self._nudge_pub.publish(twist)
            time.sleep(1.0 / rate_hz)
        self._nudge_pub.publish(Twist())  # stop

    def _do_backup(self) -> bool:
        if not self._backup_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('backup action server not available')
            return False
        goal = BackUp.Goal()
        goal.target = Point(x=self._backup_dist, y=0.0, z=0.0)
        goal.speed  = float(self._backup_speed)
        goal.time_allowance = Duration(sec=int(self._backup_dist / self._backup_speed) + 5)
        future = self._backup_ac.send_goal_async(goal)
        gh = self._await_future(future, 10.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn('BackUp goal rejected')
            return False
        result_future = gh.get_result_async()
        self._await_future(result_future, 15.0)
        self.get_logger().info('BackUp done')
        return True

    def _do_drive_on_heading(self) -> bool:
        if not self._drive_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('drive_on_heading action server not available')
            return False
        goal = DriveOnHeading.Goal()
        goal.target = Point(x=float(self._drive_dist), y=0.0, z=0.0)
        goal.speed  = float(self._drive_speed)
        goal.time_allowance = Duration(sec=int(self._drive_dist / self._drive_speed) + 5)
        future = self._drive_ac.send_goal_async(goal)
        gh = self._await_future(future, 10.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn('DriveOnHeading goal rejected')
            return False
        result_future = gh.get_result_async()
        self._await_future(result_future, 15.0)
        self.get_logger().info('DriveOnHeading done')
        return True

    def _do_spin(self) -> bool:
        if not self._spin_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('spin action server not available')
            return False
        goal = Spin.Goal()
        goal.target_yaw      = float(self._spin_angle)
        goal.time_allowance  = Duration(sec=10)
        future = self._spin_ac.send_goal_async(goal)
        gh = self._await_future(future, 10.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn('Spin goal rejected')
            return False
        result_future = gh.get_result_async()
        self._await_future(result_future, 15.0)
        self.get_logger().info('Spin done')
        return True

    def _has_moved_recently(self, window: float, threshold: float) -> bool:
        now = self.get_clock().now().nanoseconds / 1e9
        recent = [(t, x, y) for t, x, y in self._positions if t >= now - window]
        if len(recent) < 2:
            return False
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        return math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) >= threshold

    # ── Helpers ──────────────────────────────────────────────────────

    def _publish_status(self, status: str):
        self._status_pub.publish(String(data=status))

    def _speak(self, text: str):
        self._speech_pub.publish(String(data=text))
        self.get_logger().info(f'[speech] {text}')


def main():
    rclpy.init()
    node = RecoveryManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
