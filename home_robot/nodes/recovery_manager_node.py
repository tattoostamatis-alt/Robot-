#!/usr/bin/env python3
"""
recovery_manager_node.py — Physical stuck detection + Nav2 backup/spin recovery.

Watches cmd_vel_safe (what actually reaches the motors) against /odom. If the
robot receives motion commands but doesn't move for `stuck_timeout` seconds it
declares STUCK, cancels any active NavigateToPose goal, and executes:
  1. Nav2 BackUp action (0.25 m reverse)
  2. Nav2 Spin action (90° rotation)
Up to `max_attempts` times, then declares FAILED and asks for help.

Publishes: recovery/status (std_msgs/String: idle/stuck/recovering/recovered/failed)
           speech_response (std_msgs/String) — picked up by tts_node if running
"""

import math
import threading
from collections import deque
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import BackUp, Spin, NavigateToPose
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
        self.declare_parameter('spin_angle',       1.571) # radians (π/2 = 90°)
        self.declare_parameter('enabled',          True)

        self._stuck_timeout   = self.get_parameter('stuck_timeout').value
        self._min_disp        = self.get_parameter('min_displacement').value
        self._cmd_thr         = self.get_parameter('cmd_threshold').value
        self._max_attempts    = self.get_parameter('max_attempts').value
        self._backup_dist     = self.get_parameter('backup_distance').value
        self._backup_speed    = self.get_parameter('backup_speed').value
        self._spin_angle      = self.get_parameter('spin_angle').value
        self._enabled         = self.get_parameter('enabled').value

        # Sliding window: deque of (timestamp_sec, x, y)
        self._positions: deque = deque(maxlen=40)   # ~20s at 0.5Hz check
        self._cmd_active = False   # True when cmd_vel_safe magnitude > threshold
        self._cmd_active_since: float | None = None
        self._status = STATUS_IDLE
        self._lock = threading.Lock()

        # Action clients
        self._backup_ac = ActionClient(self, BackUp, 'backup')
        self._spin_ac   = ActionClient(self, Spin,   'spin')
        self._nav_ac    = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers
        self._status_pub   = self.create_publisher(String, 'recovery/status',  10)
        self._speech_pub   = self.create_publisher(String, 'speech_response',  10)

        # Manual trigger (Bool True = force a recovery attempt)
        self.create_subscription(Bool, 'recovery/trigger', self._on_trigger, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(Twist, 'cmd_vel_safe', self._on_cmd_vel, 10)

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

        with self._lock:
            active = self._cmd_active
            since  = self._cmd_active_since

        if not active or since is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
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

        backed_up = self._do_backup()
        if backed_up:
            self._do_spin()

        # Check if we can move now
        self.get_logger().info('Recovery actions done — monitoring for 2s')
        import time; time.sleep(2.0)

        if self._has_moved_recently(window=2.0, threshold=0.01):
            self._status = STATUS_RECOVERED
            self._publish_status(STATUS_RECOVERED)
            self._speak('Ξεκόλλησα!')
            self.get_logger().info('Recovery succeeded')
            # Reset to IDLE after a pause
            import time; time.sleep(2.0)
            self._status = STATUS_IDLE
            self._cmd_active = False
            self._cmd_active_since = None
        else:
            self.get_logger().warn(f'Still stuck, {attempts_left - 1} attempt(s) left')
            self._run_recovery(attempts_left - 1)

    def _cancel_navigation(self):
        if not self._nav_ac.wait_for_server(timeout_sec=2.0):
            return
        future = self._nav_ac._cancel_all_goals()
        if future:
            try:
                rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            except Exception:
                pass

    def _do_backup(self) -> bool:
        if not self._backup_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('backup action server not available')
            return False
        goal = BackUp.Goal()
        goal.target = Point(x=self._backup_dist, y=0.0, z=0.0)
        goal.speed  = float(self._backup_speed)
        goal.time_allowance = Duration(sec=int(self._backup_dist / self._backup_speed) + 5)
        future = self._backup_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        gh = future.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn('BackUp goal rejected')
            return False
        result_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
        self.get_logger().info('BackUp done')
        return True

    def _do_spin(self) -> bool:
        if not self._spin_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('spin action server not available')
            return False
        goal = Spin.Goal()
        goal.target_yaw      = float(self._spin_angle)
        goal.time_allowance  = Duration(sec=10)
        future = self._spin_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        gh = future.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn('Spin goal rejected')
            return False
        result_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
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
