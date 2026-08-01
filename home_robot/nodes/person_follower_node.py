#!/usr/bin/env python3
"""Person follower node — follows a person using RealSense D435 depth and DoA angle.

Subscribes:
  /follow_command              (std_msgs/Bool)     — True = start following, False = stop
  /doa/wake                    (std_msgs/Float32)  — wake angle [0-359°]; only updates the
                                                     target direction, does NOT activate
  /camera/camera/depth/image_rect_raw (sensor_msgs/Image, 16UC1) — depth in mm

Publishes:
  /cmd_vel                     (geometry_msgs/Twist) — velocity commands

Following logic:
  - Activates ONLY on an explicit follow_command=True (the llm_bridge 'follow' tool,
    i.e. the user said "ακολούθησέ με"). Stays active up to follow_timeout seconds.
  - Uses the latest DoA wake angle to aim at the speaker (projected onto the depth
    image column, D435 87° HFOV) — a fresh wake fired just before the command.
  - Reads median depth in a vertical strip around that column.
  - Turns to bring that bearing to the image centre, and moves forward when the
    person is too far, stops when too close, holds in range.
  - Deactivates on follow_command=False or timeout; publishes a zero-velocity stop.

Note: chmod +x this file after creation.
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


_IMAGE_WIDTH  = 640
_IMAGE_HEIGHT = 480
_CENTER_COL   = _IMAGE_WIDTH // 2  # 320


class PersonFollowerNode(Node):
    def __init__(self):
        super().__init__('person_follower')

        # --- Parameters ---------------------------------------------------
        self.declare_parameter('follow_distance_min', 0.8)
        self.declare_parameter('follow_distance_max', 1.5)
        self.declare_parameter('follow_speed',        0.15)
        self.declare_parameter('follow_timeout',      30.0)
        self.declare_parameter('depth_strip_width',   60)
        self.declare_parameter('camera_hfov_deg',     87.0)
        # ‼️ Turning. The docstring has always claimed this node "aims at the
        # speaker", but _control only ever set linear.x — the DoA column picked
        # which depth strip to MEASURE and nothing more, so the robot drove
        # dead ahead on a distance read from 40° off to one side. Without a turn
        # the forward motion is not just unhelpful, it is aimed at something the
        # node never looked at.
        self.declare_parameter('turn_gain',        0.0025)  # rad/s per pixel of error
        self.declare_parameter('turn_deadband_px',   40)    # ignore small errors
        # This chassis cannot rotate below ~0.31 rad/s — commanding less just
        # stalls the wheels (see the rotation-floor measurements in project
        # memory). So a turn is either a real one or none at all.
        self.declare_parameter('min_turn_speed',     0.35)  # rad/s
        self.declare_parameter('max_turn_speed',     0.60)  # rad/s

        self._dist_min    = self.get_parameter('follow_distance_min').value
        self._dist_max    = self.get_parameter('follow_distance_max').value
        self._speed       = self.get_parameter('follow_speed').value
        self._timeout     = self.get_parameter('follow_timeout').value
        self._strip_width = self.get_parameter('depth_strip_width').value
        self._hfov        = self.get_parameter('camera_hfov_deg').value
        self._turn_gain   = self.get_parameter('turn_gain').value
        self._turn_dead   = self.get_parameter('turn_deadband_px').value
        self._turn_min    = self.get_parameter('min_turn_speed').value
        self._turn_max    = self.get_parameter('max_turn_speed').value

        # --- Internal state (all access via _lock) ------------------------
        self._lock         = threading.Lock()
        self._active       = False
        self._active_until = 0.0       # monotonic timestamp when mode expires
        self._doa_col      = _CENTER_COL  # image column derived from DoA angle

        self._bridge = CvBridge()

        # --- Publishers ---------------------------------------------------
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- Subscribers --------------------------------------------------
        self.create_subscription(Bool, 'follow_command',
                                 self._follow_cmd_cb, 10)
        self.create_subscription(Float32, '/doa/wake',
                                 self._doa_cb, 10)
        # ‼️ /camera/CAMERA/... — the realsense2_camera launch nests the camera
        # name under the camera namespace, so every other node in this package
        # uses the doubled prefix (object_detector, global_localizer,
        # web_dashboard). This one was missing a level and subscribed to a topic
        # NOBODY PUBLISHES, with no remapping in bringup.launch.py to save it:
        # "ακολούθησέ με" logged "Following activated", never received a single
        # frame, never published a single Twist, and silently timed out 30 s
        # later. The whole feature was dead, on by default, since it was written.
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw',
                                 self._depth_cb, 10)

        self.get_logger().info(
            f'PersonFollowerNode ready '
            f'(dist={self._dist_min}-{self._dist_max}m, '
            f'speed={self._speed}m/s, timeout={self._timeout}s)'
        )

    # ------------------------------------------------------------------
    # DoA callback — activates following mode
    # ------------------------------------------------------------------
    def _doa_cb(self, msg: Float32):
        """Update the target direction from the DoA wake angle. Does NOT activate
        following — that only happens on an explicit follow_command=True."""
        angle_deg = float(msg.data)

        # Convert clockwise 0-359° to signed angle: positive = right, negative = left
        signed_angle = angle_deg if angle_deg <= 180.0 else angle_deg - 360.0

        # Project to pixel column; clamp to valid image range
        px_per_deg = _IMAGE_WIDTH / self._hfov
        col = int(_CENTER_COL + signed_angle * px_per_deg)
        col = max(0, min(_IMAGE_WIDTH - 1, col))

        with self._lock:
            self._doa_col = col

    # ------------------------------------------------------------------
    # Follow command callback — activates / deactivates following mode
    # ------------------------------------------------------------------
    def _follow_cmd_cb(self, msg: Bool):
        """Start/stop following on the llm_bridge 'follow'/'stop_follow' tools."""
        if msg.data:
            with self._lock:
                self._active       = True
                self._active_until = time.monotonic() + self._timeout
                col = self._doa_col
            self.get_logger().info(
                f'Following activated (target col={col}, timeout={self._timeout:.0f}s)')
        else:
            with self._lock:
                was_active   = self._active
                self._active = False
            if was_active:
                self._publish_stop()
                self.get_logger().info('Following stopped: stop command received')

    # ------------------------------------------------------------------
    # Depth image callback — main control loop
    # ------------------------------------------------------------------
    def _depth_cb(self, msg: Image):
        """Process depth frame and issue cmd_vel; only acts when active."""
        # Fast gate — check active flag under lock
        with self._lock:
            if not self._active:
                return
            now = time.monotonic()
            if now >= self._active_until:
                self._active = False
                self.get_logger().info('Following stopped: 30-second timeout')
                self._publish_stop()
                return
            col = self._doa_col

        # Convert ROS Image → numpy array (16UC1, depth in millimetres)
        try:
            depth_img = self._bridge.imgmsg_to_cv2(msg, '16UC1')
        except Exception as exc:
            self.get_logger().warn(f'cv_bridge conversion error: {exc}')
            return

        # Extract vertical strip around the target column
        half      = self._strip_width // 2
        col_start = max(0, col - half)
        col_end   = min(depth_img.shape[1], col + half)
        strip     = depth_img[:, col_start:col_end]

        # Flatten and discard invalid (zero) readings
        flat  = strip.flatten()
        valid = flat[flat > 0]

        if valid.size == 0:
            self.get_logger().debug(
                f'No valid depth pixels in strip (col={col_start}-{col_end}), skipping'
            )
            return

        median_mm = float(np.median(valid))
        median_m  = median_mm / 1000.0

        self.get_logger().debug(
            f'Strip col={col_start}-{col_end}: {valid.size} valid px, '
            f'median={median_m:.3f}m'
        )

        self._control(median_m, col)

    # ------------------------------------------------------------------
    # Control logic
    # ------------------------------------------------------------------
    def _turn_for(self, col: int) -> float:
        """Angular velocity that brings `col` toward the image centre.

        Quantized to the chassis' real capability: below min_turn_speed the
        wheels do not move at all, so anything smaller is commanded as zero
        rather than as a stall.
        """
        err = col - _CENTER_COL            # +ve: target is right of centre
        if abs(err) <= self._turn_dead:
            return 0.0
        wz = -self._turn_gain * err        # +z is left (CCW) in REP-103
        mag = min(max(abs(wz), self._turn_min), self._turn_max)
        return math.copysign(mag, wz)

    def _control(self, depth_m: float, col: int):
        """Publish Twist based on measured distance to, and bearing of, person."""
        twist = Twist()
        twist.angular.z = self._turn_for(col)

        if depth_m > self._dist_max:
            # Person is too far away — drive toward them
            twist.linear.x = self._speed
            self.get_logger().debug(
                f'depth={depth_m:.2f}m > {self._dist_max}m → forward'
            )
        elif depth_m < self._dist_min:
            # Person is too close — stop completely
            twist.linear.x = 0.0
            self.get_logger().debug(
                f'depth={depth_m:.2f}m < {self._dist_min}m → too close, stop'
            )
        else:
            # Good following distance — hold position
            twist.linear.x = 0.0
            self.get_logger().debug(
                f'depth={depth_m:.2f}m in range [{self._dist_min},{self._dist_max}]m → hold'
            )

        self._cmd_pub.publish(twist)

    def _publish_stop(self):
        """Publish a zero-velocity Twist to halt the robot."""
        self._cmd_pub.publish(Twist())


# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
