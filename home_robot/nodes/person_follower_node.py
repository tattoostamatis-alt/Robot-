#!/usr/bin/env python3
"""Person follower node — follows a person using object_detector's detections.

Subscribes:
  /follow_command   (std_msgs/Bool)    — True = start following, False = stop
  /doa/wake         (std_msgs/Float32) — wake angle [0-359°]; seeds WHICH person to
                                         lock onto, does NOT activate following
  detected_objects  (std_msgs/String)  — object_detector.py's JSON detections,
                                         carrying label + pixel box + box_distance

Publishes:
  /cmd_vel_safe     (geometry_msgs/Twist) — velocity commands. NOT /cmd_vel;
                                            see the note at the publisher.

Following logic:
  - Activates ONLY on an explicit follow_command=True (the llm_bridge 'follow' tool,
    i.e. the user said "ακολούθησέ με"). Stays active up to follow_timeout seconds.
  - Each detection frame, picks the 'person' box nearest the current target column
    (seeded from the DoA wake angle) and re-aims at it.
  - Turns to bring that person to the image centre, and moves forward when they are
    too far, stops when too close, holds in range.
  - Deactivates on follow_command=False or timeout; publishes a zero-velocity stop.

‼️ This used to run open-loop off a vertical depth strip at the DoA column, and
`_doa_col` was written ONLY by the wake callback — so during a follow the bearing
error was a CONSTANT. `_turn_for` therefore returned the same non-zero rate on
every frame and the robot spun on the spot for the full 30 s, measuring depth at
an image column it had long since turned away from. Nothing in the loop could
ever reduce the error, because nothing in the loop looked at where the person
actually was. The fix is this file's whole point: the bearing now comes from the
detector every frame, so turning toward the person is what makes the error shrink.
Found 2026-08-01.
"""

import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String
from geometry_msgs.msg import Twist


_IMAGE_WIDTH  = 640                # DoA seed projection only; frames carry img_w
_CENTER_COL   = _IMAGE_WIDTH // 2  # 320


class PersonFollowerNode(Node):
    def __init__(self):
        super().__init__('person_follower')

        # --- Parameters ---------------------------------------------------
        self.declare_parameter('follow_distance_min', 0.8)
        # Start closing the gap once the person is more than one metre away.
        self.declare_parameter('follow_distance_max', 1.0)
        self.declare_parameter('follow_speed',        0.15)
        self.declare_parameter('follow_timeout',      30.0)
        self.declare_parameter('camera_hfov_deg',     87.0)
        # How long a person may be out of view before the robot stops moving.
        # ‼️ Not cosmetic: acting on a stale bearing is exactly what made the
        # old open-loop version spin forever. No fresh sighting, no motion.
        self.declare_parameter('person_timeout',      1.0)
        # When the person leaves the camera image, do not drive on a stale
        # depth/bearing.  A short in-place turn is enough to reacquire someone
        # who has just crossed an edge of the image, without turning a lost
        # target into blind navigation through the house.
        self.declare_parameter('reacquire_turn_seconds', 4.0)
        self.declare_parameter('reacquire_turn_speed',   0.35)
        self.declare_parameter('motion_deadband_px_s',   20.0)
        # Do not drive forward while the person is still far off-centre: with
        # the D435 pitched down, a combined forward+turn manoeuvre keeps only
        # the lower body in frame and loses the target exactly when someone
        # turns away across the edge. Turn first, then advance.
        self.declare_parameter('forward_turn_gate_px',   90)
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
        # Divisor for the pixels-per-degree conversion.
        self._hfov        = max(1.0, float(self.get_parameter('camera_hfov_deg').value))
        self._lost_after  = self.get_parameter('person_timeout').value
        self._reacquire_for = self.get_parameter('reacquire_turn_seconds').value
        self._reacquire_speed = self.get_parameter('reacquire_turn_speed').value
        self._motion_deadband = self.get_parameter('motion_deadband_px_s').value
        self._forward_turn_gate = self.get_parameter('forward_turn_gate_px').value
        self._turn_gain   = self.get_parameter('turn_gain').value
        self._turn_dead   = self.get_parameter('turn_deadband_px').value
        self._turn_min    = self.get_parameter('min_turn_speed').value
        self._turn_max    = self.get_parameter('max_turn_speed').value

        # --- Internal state (all access via _lock) ------------------------
        self._lock         = threading.Lock()
        self._active       = False
        self._active_until = 0.0          # monotonic timestamp when mode expires
        self._doa_col      = _CENTER_COL  # image column derived from DoA angle
        # Where the followed person was last SEEN. Seeded from _doa_col when a
        # follow starts, then re-written from the detector on every frame —
        # that write is what closes the control loop.
        self._target_col   = _CENTER_COL
        self._last_seen    = 0.0          # monotonic; staleness clock for _lost_after
        self._last_col_velocity = 0.0      # pixels/s; direction before occlusion
        self._lost_warned  = False
        self._pose_people  = []
        self._pose_seen    = 0.0

        # --- Publishers ---------------------------------------------------
        # ‼️ cmd_vel_safe, NOT cmd_vel. /cmd_vel has FOUR publishers and ZERO
        # subscribers on this graph: roomba_driver listens on cmd_vel_safe alone,
        # and Nav2 reaches it by its own chain (cmd_vel_nav -> velocity_smoother
        # -> cmd_vel_smoothed -> collision_monitor -> cmd_vel_safe). So every
        # twist this node published went nowhere and "ακολούθησέ με" could not
        # move the robot no matter how correct the control loop was — which is
        # the cruel part, because the loop above had already been rewritten once
        # (2026-08-01) to fix a real spinning bug that this masked entirely.
        #
        # This is the FOURTH time the same wrong topic has been used here (the
        # PS5 teleop, the keyboard teleop and the web D-pad were the others).
        # Before adding anything that moves the robot: `ros2 topic info -v
        # /cmd_vel_safe` and check your publisher is in the list.
        #
        # Like the joystick and the web D-pad, this path bypasses
        # collision_monitor. What still guards it: roomba_driver's bumper/cliff/
        # wheel-drop stops, its 0.25 s stale-command watchdog, the latched
        # e-stop, and this node's own follow_timeout.
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)

        # --- Subscribers --------------------------------------------------
        self.create_subscription(Bool, 'follow_command',
                                 self._follow_cmd_cb, 10)
        self.create_subscription(Float32, '/doa/wake',
                                 self._doa_cb, 10)
        # object_detector.py already runs YOLO on the colour frame and pairs each
        # box with a median depth, which is both halves of what following needs
        # (bearing + distance) from a node that is up anyway. Reading the depth
        # image directly here — as this node used to — meant a second consumer of
        # the heaviest topic on the bus that still could not tell a person from a
        # wall, which is how it ended up steering by a stale DoA angle.
        self.create_subscription(String, 'detected_objects',
                                 self._detections_cb, 10)
        self.create_subscription(String, 'pose_detections',
                                 self._poses_cb, 10)

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
                # The wake angle picks WHICH person to lock onto on the first
                # frame; from there the detector takes over.
                self._target_col   = self._doa_col
                # Start the staleness clock now, so a follow where nobody is
                # ever detected says so after person_timeout instead of sitting
                # silent for the full 30 s.
                self._last_seen    = time.monotonic()
                self._last_col_velocity = 0.0
                self._lost_warned  = False
                col = self._target_col
            self.get_logger().info(
                f'Following activated (seed col={col}, timeout={self._timeout:.0f}s)')
        else:
            with self._lock:
                was_active   = self._active
                self._active = False
            if was_active:
                self._publish_stop()
                self.get_logger().info('Following stopped: stop command received')

    # ------------------------------------------------------------------
    # Pose callback — upper-body centring for a downward-pitched camera
    # ------------------------------------------------------------------
    def _poses_cb(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            persons = payload.get('persons') or []
        else:
            persons = payload
        if not isinstance(persons, list):
            return
        with self._lock:
            self._pose_people = [p for p in persons if isinstance(p, dict)]
            self._pose_seen = time.monotonic()

    # ------------------------------------------------------------------
    # Person selection
    # ------------------------------------------------------------------
    @staticmethod
    def _pose_anchor(person):
        """A horizontal target column from the upper body when pose is available.

        With the camera pitched down, the box centre is biased toward the legs.
        Steering from shoulders/hips/nose keeps the followed person visually
        "higher" in the frame and preserves lock better while turning.
        """
        keypoints = person.get('keypoints') or []
        by_name = {
            kp.get('name'): kp for kp in keypoints
            if isinstance(kp, dict) and kp.get('v', 0.0) >= 0.5
        }
        cols = []
        for name in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'nose'):
            kp = by_name.get(name)
            if kp and kp.get('x') is not None:
                cols.append(float(kp['x']))
        if cols:
            return sum(cols) / len(cols)
        x1, x2 = person.get('x1'), person.get('x2')
        if x1 is None or x2 is None:
            return None
        return (float(x1) + float(x2)) / 2.0

    @classmethod
    def _pick_pose_person(cls, persons, target_col):
        """Pose person nearest target_col, returned as centre column only."""
        best = None
        for person in persons:
            centre = cls._pose_anchor(person)
            if centre is None:
                continue
            gap = abs(centre - target_col)
            if best is None or gap < best[0]:
                best = (gap, centre)
        return None if best is None else best[1]

    @staticmethod
    def _pick_person(objects, target_col):
        """The 'person' box nearest `target_col`, as (centre_col, distance_m, img_w).

        Nearest-to-last-seen rather than nearest-to-camera or biggest: with two
        people in frame those two rules swap targets the moment someone walks
        past, whereas continuity keeps the robot on whoever it was already
        following. On the first frame `target_col` is the DoA seed, so the
        person the speaker's voice came from wins.

        Detections without a usable box or distance are skipped — a person with
        no depth (all-zero pixels, too close, too far) gives a bearing we could
        turn to but no distance to decide forward motion on, and half a control
        input is worse than none.
        """
        best = None
        for obj in objects:
            if obj.get('label') != 'person':
                continue
            x1, x2 = obj.get('x1'), obj.get('x2')
            distance, img_w = obj.get('box_distance'), obj.get('img_w')
            if None in (x1, x2, distance, img_w):
                continue
            centre = (x1 + x2) / 2.0
            gap = abs(centre - target_col)
            if best is None or gap < best[0]:
                best = (gap, centre, float(distance), int(img_w))
        if best is None:
            return None
        _, centre, distance, img_w = best
        return centre, distance, img_w

    @staticmethod
    def _apply_pose_centre(found, pose_centre):
        """Replace bbox centre with pose-based upper-body centre when sane."""
        if found is None or pose_centre is None:
            return found
        centre, distance, img_w = found
        if abs(float(pose_centre) - centre) > img_w * 0.25:
            return found
        return float(pose_centre), distance, img_w

    # ------------------------------------------------------------------
    # Detections callback — main control loop
    # ------------------------------------------------------------------
    def _detections_cb(self, msg: String):
        """Re-aim at the followed person and issue cmd_vel; only acts when active."""
        with self._lock:
            if not self._active:
                return
            now = time.monotonic()
            if now >= self._active_until:
                self._active = False
                self.get_logger().info(
                    f'Following stopped: {self._timeout:.0f}-second timeout')
                self._publish_stop()
                return
            target_col = self._target_col
            last_seen  = self._last_seen

        try:
            objects = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            pose_people = list(self._pose_people)
            pose_seen = self._pose_seen

        found = self._pick_person(objects, target_col)
        if pose_people and (now - pose_seen) <= 0.7:
            found = self._apply_pose_centre(
                found, self._pick_pose_person(pose_people, target_col))

        if found is None:
            # Out of view. Never move forwards from a stale sighting.  For a
            # few seconds turn in place toward the side/direction where the
            # person disappeared; this reacquires ordinary camera-edge losses
            # while preserving the hard stop when the person is truly gone.
            if last_seen and (now - last_seen) > self._lost_after:
                with self._lock:
                    warn, self._lost_warned = not self._lost_warned, True
                    target_col = self._target_col
                    velocity = self._last_col_velocity
                lost_for = now - last_seen
                if lost_for <= self._reacquire_for:
                    self._publish_reacquire_turn(target_col, velocity)
                else:
                    self._publish_stop()
                if warn:
                    self.get_logger().info(
                        'Δεν σε βλέπω — ψάχνω γύρω μου χωρίς να προχωρώ')
            return

        centre, distance_m, img_w = found
        with self._lock:
            # Remembers WHO to follow next frame (see _pick_person). The control
            # loop itself is closed one line further down, by handing _control
            # `centre` — this frame's measurement — rather than this stored one.
            previous_col = self._target_col
            previous_seen = self._last_seen
            elapsed = max(now - previous_seen, 1e-3)
            # Clamp detector jitter: a single bad box must not choose the
            # search direction after an occlusion.
            self._last_col_velocity = max(
                -img_w, min(img_w, (centre - previous_col) / elapsed))
            self._target_col  = centre
            self._last_seen   = now
            was_lost, self._lost_warned = self._lost_warned, False
        if was_lost:
            self.get_logger().info('Σε ξαναβρήκα')

        self._control(distance_m, centre, img_w)

    # ------------------------------------------------------------------
    # Control logic
    # ------------------------------------------------------------------
    def _turn_for(self, col: float, img_w: int = _IMAGE_WIDTH) -> float:
        """Angular velocity that brings `col` toward the image centre.

        Quantized to the chassis' real capability: below min_turn_speed the
        wheels do not move at all, so anything smaller is commanded as zero
        rather than as a stall. That quantization is also why the deadband
        matters — without it the robot would hunt around centre at 0.35 rad/s,
        never able to command the small correction it actually needs.
        """
        err = col - img_w / 2.0            # +ve: target is right of centre
        if abs(err) <= self._turn_dead:
            return 0.0
        wz = -self._turn_gain * err        # +z is left (CCW) in REP-103
        mag = min(max(abs(wz), self._turn_min), self._turn_max)
        return math.copysign(mag, wz)

    def _control(self, depth_m: float, col: float, img_w: int = _IMAGE_WIDTH):
        """Publish Twist based on measured distance to, and bearing of, person."""
        twist = Twist()
        twist.angular.z = self._turn_for(col, img_w)
        centred_enough = abs(col - img_w / 2.0) <= self._forward_turn_gate

        if depth_m > self._dist_max:
            # Person is too far away — but only advance once the turn has
            # mostly completed, or the downward-pitched camera keeps the legs
            # in frame and loses the rest of the person on a turn.
            twist.linear.x = self._speed if centred_enough else 0.0
            self.get_logger().debug(
                f'depth={depth_m:.2f}m > {self._dist_max}m → '
                f'{"forward" if centred_enough else "turn-first"}'
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

    def _publish_reacquire_turn(self, last_col: float, velocity_px_s: float):
        """Turn in place toward the last observed travel/bearing, never drive."""
        if abs(velocity_px_s) >= self._motion_deadband:
            # Person moving right in the image -> turn right (negative z).
            direction = -math.copysign(1.0, velocity_px_s)
        else:
            # With no reliable motion estimate, use the side where they were
            # last visible.  A centred target gives no safe direction to infer.
            turn = self._turn_for(last_col)
            if turn == 0.0:
                self._publish_stop()
                return
            direction = math.copysign(1.0, turn)
        twist = Twist()
        twist.angular.z = direction * self._reacquire_speed
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
