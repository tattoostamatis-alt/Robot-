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
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import BackUp, DriveOnHeading, NavigateToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
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
        self.declare_parameter('min_rotation',     0.05)  # radians — rotation-in-place is movement too
        self.declare_parameter('cmd_threshold',    0.01)  # m/s or rad/s — above = motion commanded
        # Tiny controller settling commands near a goal are not evidence that
        # the chassis should be moving.  Auto-recovery there can cancel a
        # successful approach and nudge the robot away from the object.
        self.declare_parameter('stuck_linear_cmd_threshold',  0.04)
        self.declare_parameter('stuck_angular_cmd_threshold', 0.15)
        self.declare_parameter('max_attempts',     3)
        # 0.25/0.15 originally — dropped 2026-08-09 after a live stuck event
        # where a requested BackUp/DriveOnHeading as small as 8cm still hit
        # "Collision Ahead" instantly (see _clear_costmaps): the smaller ask
        # completes inside whatever real gap exists instead of demanding a
        # guaranteed-clear 25cm runway that a tight doorway may not have.
        self.declare_parameter('backup_distance',  0.10)  # meters
        self.declare_parameter('backup_speed',     0.05)  # m/s
        self.declare_parameter('drive_distance',   0.08)  # meters — DriveOnHeading creep
        self.declare_parameter('drive_speed',      0.05)  # m/s
        self.declare_parameter('spin_angle',       1.571) # radians (π/2 = 90°)
        self.declare_parameter('enabled',          True)
        # Speak the running commentary («Κόλλησα» / «Ξεκόλλησα»)? Off: it fires
        # on every pinch point and says nothing the person in the room cannot
        # already see. The give-up line is spoken regardless — see _speak.
        self.declare_parameter('announce_progress', False)
        self.declare_parameter('nudge_linear_speed',  0.10)  # m/s — doorway pinch-point creep
        # ‼️ 2026-08-09: was 0.10 rad/s, which this chassis cannot execute AT
        # ALL — the Roomba 879 has a hard rotation floor around 0.31 rad/s
        # (project_robot_rotation_floor), below which the wheels simply do not
        # turn. So the nudge step — the last raw-cmd_vel escape before the
        # recovery gives up, deliberately placed below the collision monitor
        # precisely because everything above it has already refused — published
        # 1.2 s of a command the robot ignored, every single time, and then
        # reported that it had tried. 0.45 clears the floor with margin.
        self.declare_parameter('nudge_angular_speed', 0.45)  # rad/s
        self.declare_parameter('nudge_duration',      1.2)   # seconds per phase (turn, then creep)
        # After a successful escape, re-send the navigation goal that was
        # cancelled to run the recovery. recovery_manager cancels the active
        # NavigateToPose to nudge (owner gets CANCELED and stops), so without
        # this the robot just sits free after each pinch — the goto never
        # completes hands-off. We learn the goal from the last /plan endpoint
        # (map frame, published by planner_server for every owner) and re-issue
        # it ourselves. Capped at reissue_max consecutive re-issues *without
        # net progress* so a truly unreachable (in-wall) goal can't loop forever;
        # any odom advance > reissue_progress_dist between two pinches resets the
        # count, so a multi-doorway journey re-issues as many times as it needs.
        self.declare_parameter('reissue_after_recovery', True)
        self.declare_parameter('reissue_max',            4)     # re-issues w/o progress before giving up
        self.declare_parameter('reissue_progress_dist',  0.25)  # m odom advance that resets the count
        self.declare_parameter('goal_max_age',           30.0)  # s — don't re-issue a stale plan endpoint
        # _check_stuck can only fire while cmd_vel is actively commanded — it
        # has nothing to watch when the planner refuses to produce a plan AT
        # ALL from the robot's current pose (footprint wedged in a genuine
        # inflation cell, not a stale costmap mark — clear_costmaps alone
        # doesn't free that). bt_navigator's own number_of_recoveries feedback
        # keeps climbing in that case with zero cmd_vel ever published; this
        # is the threshold (recoveries, on the CURRENT goal, with cmd_vel
        # never once active) before _on_nav_feedback escalates to the same
        # nudge/backup _run_recovery uses. Found 2026-08-11 replaying a
        # fetch_sim_gazebo run that looped "clearing costmaps too" #1-#15
        # and never once nudged.
        self.declare_parameter('planner_stuck_recoveries', 3)

        self._stuck_timeout   = self.get_parameter('stuck_timeout').value
        self._min_disp        = self.get_parameter('min_displacement').value
        self._min_rotation    = self.get_parameter('min_rotation').value
        self._cmd_thr         = self.get_parameter('cmd_threshold').value
        self._stuck_linear_cmd = self.get_parameter('stuck_linear_cmd_threshold').value
        self._stuck_angular_cmd = self.get_parameter('stuck_angular_cmd_threshold').value
        self._max_attempts    = self.get_parameter('max_attempts').value
        self._backup_dist     = self.get_parameter('backup_distance').value
        # Divisors when the time allowance is computed; 0 would crash the
        # recovery that is meant to be rescuing a stuck robot.
        self._backup_speed    = max(0.01, float(self.get_parameter('backup_speed').value))
        self._drive_dist      = self.get_parameter('drive_distance').value
        self._drive_speed     = max(0.01, float(self.get_parameter('drive_speed').value))
        self._spin_angle      = self.get_parameter('spin_angle').value
        self._enabled         = self.get_parameter('enabled').value
        self._announce_progress = bool(self.get_parameter('announce_progress').value)
        self._nudge_linear    = self.get_parameter('nudge_linear_speed').value
        self._nudge_angular   = self.get_parameter('nudge_angular_speed').value
        self._nudge_duration  = self.get_parameter('nudge_duration').value
        self._reissue_enabled = self.get_parameter('reissue_after_recovery').value
        self._reissue_max     = self.get_parameter('reissue_max').value
        self._reissue_prog    = self.get_parameter('reissue_progress_dist').value
        self._goal_max_age    = self.get_parameter('goal_max_age').value
        self._planner_stuck_recoveries = self.get_parameter('planner_stuck_recoveries').value

        # Last global-plan endpoint = the goal we're driving to (map frame),
        # captured from /plan so re-issue works for any goal owner.
        self._last_goal: PoseStamped | None = None
        self._last_goal_rx: float | None = None
        self._reissue_count = 0
        self._reissue_odom: tuple | None = None   # odom (x,y) at the last re-issue
        # bt_navigator's own number_of_recoveries feedback — see _on_nav_feedback.
        self._last_num_recoveries: int | None = None
        self._nav_feedback_goal_id: bytes | None = None

        # Sliding window: deque of (timestamp_sec, x, y, yaw).  Translation
        # alone cannot distinguish a legitimately rotating robot from a stuck
        # one, which used to cancel valid Nav2 goals after five seconds of an
        # initial in-place turn.
        self._positions: deque = deque(maxlen=40)   # ~20s at 0.5Hz check
        self._cmd_active = False   # True when cmd_vel_smoothed magnitude > threshold
        self._cmd_active_since: float | None = None
        self._cmd_last_rx: float | None = None  # last cmd_vel_smoothed arrival
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
        # Per-CURRENT-goal: has cmd_vel EVER gone active since bt_navigator
        # started counting recoveries from 0? Reset on that edge in
        # _on_nav_feedback, not on cmd_vel going idle — see _on_nav_feedback.
        self._cmd_ever_active_this_goal = False
        self._planner_stuck_triggered = False  # one escalation per goal attempt
        self._status = STATUS_IDLE
        self._cancelled = False    # set by /mission/cancel; see _on_cancel
        # Goal handle of the Nav2 behavior currently running, so a cancel can
        # stop it mid-motion instead of waiting out its time_allowance. BackUp
        # alone is ~10 s of the robot reversing after the person said stop.
        self._active_behavior = None
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
        # Our own NavigateToPose client to re-issue the goal after recovery.
        self._nav_ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # Cleared at the top of every recovery — see _clear_costmaps(). A STUCK
        # event with real free space on all sides (confirmed live 2026-08-09:
        # BackUp/DriveOnHeading both refused with "Collision Ahead" at a
        # requested distance as small as 8cm, and the planner could not find a
        # path even FROM the robot's own current cell) means the costmap, not
        # the world, is blocking it — most likely stale voxel_layer marks from
        # a corrupted RealSense frame (the camera logs "Frame Corrupted"
        # continuously). Clearing before every attempt is what unstuck it by
        # hand that day; this makes the same fix automatic.
        self._clear_local_cli  = self.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')
        self._clear_global_cli = self.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap')

        # Publishers
        self._status_pub   = self.create_publisher(String, 'recovery/status',  10)
        self._speech_pub   = self.create_publisher(String, 'speech_response',  10)
        # Same bypass teleop uses (project_robot_teleop_cmdvel_safe): publish
        # straight onto cmd_vel_safe, past collision_monitor, for the nudge.
        self._nudge_pub    = self.create_publisher(Twist,  'cmd_vel_safe',     10)

        # Manual trigger (Bool True = force a recovery attempt)
        self.create_subscription(Bool, 'recovery/trigger', self._on_trigger, 10)
        # ‼️ A human cancel must beat the re-issue. This node learns the goal
        # from /plan and re-sends it after every escape, which is exactly right
        # for a multi-doorway journey and exactly wrong when someone has just
        # pressed "✕ Ακύρωση στόχου": the goal came back a few seconds later and
        # the robot carried on (reported 2026-08-04). Nothing was listening to
        # /mission/cancel at all — not this node, and not the `cancel` gesture's
        # publisher either.
        self.create_subscription(String, '/mission/cancel', self._on_cancel, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        # Only consumer is the nudge step, which needs a direction to escape in
        # when the costmap-based behaviors have all refused.
        self._scan = None
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        # Teleop that remaps straight onto cmd_vel_safe bypasses this on
        # purpose: no auto-recovery fighting the joystick.
        self.create_subscription(Twist, 'cmd_vel_smoothed', self._on_cmd_vel, 10)
        # Learn the active goal (last plan pose, map frame) so we can re-issue it.
        self.create_subscription(Path, 'plan', self._on_plan, 10)
        # bt_navigator's own recovery count for the ACTIVE goal, whoever owns
        # it — catches the case _check_stuck cannot: a planner that keeps
        # failing to find a path (own footprint pinned by a stale costmap
        # mark, see _clear_costmaps) never commands cmd_vel, so the
        # cmd_vel-vs-displacement watch never starts and STUCK never fires.
        self.create_subscription(
            NavigateToPose.Impl.FeedbackMessage, 'navigate_to_pose/_action/feedback',
            self._on_nav_feedback, 10)

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
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._positions.append((t, x, y, yaw))

    def _on_scan(self, msg: LaserScan):
        self._scan = msg

    def _on_cmd_vel(self, msg: Twist):
        mag = abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z)
        active = mag > self._cmd_thr
        now = self.get_clock().now().nanoseconds / 1e9
        with self._lock:
            self._cmd_last_rx = now
            self._cmd_linear = math.hypot(msg.linear.x, msg.linear.y)
            self._cmd_angular = abs(msg.angular.z)
            if active:
                self._cmd_ever_active_this_goal = True
            if active and not self._cmd_active:
                self._cmd_active = True
                self._cmd_active_since = now
            elif not active:
                self._cmd_active = False
                self._cmd_active_since = None

    def _on_plan(self, msg: Path):
        # The planner republishes the full path each cycle; its last pose is the
        # current navigation goal in the map frame. Stamp it so a stale plan from
        # a finished navigation isn't re-issued much later.
        if not msg.poses:
            return
        goal = PoseStamped()
        goal.header.frame_id = msg.header.frame_id or 'map'
        goal.pose = msg.poses[-1].pose
        with self._lock:
            prev = self._last_goal
            self._last_goal = goal
            self._last_goal_rx = self.get_clock().now().nanoseconds / 1e9
        # A goal endpoint that moved is a NEW navigation target, not a fresh
        # /plan for the one we were already chasing (the planner republishes
        # every cycle). The reissue budget belongs to that old goal, not this
        # one — without this, a goal sent straight to navigate_to_pose
        # (bypassing /mission/cancel, e.g. from the CLI or after an
        # /emergency_stop, which this node doesn't watch) inherits a stale,
        # possibly near-exhausted count from an unrelated earlier goal and
        # can give up on a fresh goal almost immediately (seen 2026-08-09:
        # "attempt 4/4" on the very first stuck event of a brand-new goal).
        changed = prev is None or self._pose_dist(prev.pose, goal.pose) > self._reissue_prog
        if changed:
            self._reissue_count = 0
            self._reissue_odom = None
            self._last_num_recoveries = None
            # A mission owner may react to our cancelled NavigateToPose by
            # choosing a different target (retry, delivery, or next room)
            # before this recovery thread has completed all of its physical
            # steps.  That new plan supersedes the old recovery.  Without this
            # guard the stale thread can cancel or nudge underneath the new
            # goal — observed in Room4 when a fetch started returning home.
            if prev is not None and self._status in (STATUS_STUCK, STATUS_RECOVERING):
                self._supersede_recovery()

    @staticmethod
    def _pose_dist(a, b) -> float:
        return math.hypot(a.position.x - b.position.x, a.position.y - b.position.y)

    def _on_nav_feedback(self, msg):
        goal_id = bytes(msg.goal_id.uuid)
        previous_goal_id = self._nav_feedback_goal_id
        self._nav_feedback_goal_id = goal_id
        if (previous_goal_id is not None and goal_id != previous_goal_id
                and self._status in (STATUS_STUCK, STATUS_RECOVERING)):
            self._supersede_recovery()

        # /plan only republishes on a SUCCESSFUL compute_path_to_pose, so a
        # goal stuck in repeated planning failures (see above) never refreshes
        # _last_goal_rx there. This feedback arrives for the goal regardless —
        # proof it's still live — so touch the timestamp here too, or
        # _maybe_reissue_goal silently drops a goal that is very much still
        # active once goal_max_age (30s) has passed without a plan.
        with self._lock:
            if self._last_goal is not None:
                self._last_goal_rx = self.get_clock().now().nanoseconds / 1e9
        n = msg.feedback.number_of_recoveries
        if n == 0 and self._last_num_recoveries != 0:
            # bt_navigator numbers recoveries from 0 for every NEW
            # NavigateToPose goal — this edge is the per-goal reset point.
            # _on_plan's reset can't be used here: /plan only republishes on
            # a SUCCESSFUL compute_path_to_pose, so a goal whose planner
            # never once succeeds (the exact case below) never fires it.
            with self._lock:
                self._cmd_ever_active_this_goal = False
            self._planner_stuck_triggered = False
        if self._last_num_recoveries is not None and n > self._last_num_recoveries:
            # bt_navigator's own RoundRobin clears costmaps on only 1 of its 5
            # recovery branches (BackUp/DriveOnHeading/Spin/Wait are the other
            # 4), so a planner stuck on a stale costmap mark can burn through
            # several recoveries between clears. Run from a fresh thread, same
            # as _run_recovery — this callback is on the executor thread and
            # _clear_costmaps' _await_future needs that thread free to spin.
            self.get_logger().info(f'bt_navigator recovery #{n} — clearing costmaps too')
            threading.Thread(target=self._clear_costmaps, daemon=True).start()
        self._last_num_recoveries = n

        # Planner-only failure: costmap-clearing above fixes a STALE mark, but
        # a footprint genuinely wedged in a real inflation cell just keeps
        # failing to plan — with cmd_vel NEVER active, _check_stuck has
        # nothing to trigger on, so nothing ever escalates to the physical
        # nudge/backup that would actually free it. One escalation per goal
        # attempt (_planner_stuck_triggered, reset on the n==0 edge above).
        with self._lock:
            never_moved = not self._cmd_ever_active_this_goal
        if (self._enabled and self._status == STATUS_IDLE
                and not self._planner_stuck_triggered
                and never_moved and n >= self._planner_stuck_recoveries):
            self._planner_stuck_triggered = True
            self._declare_stuck(
                f'{n} bt_navigator recoveries on this goal, cmd_vel never active '
                f'(planner refuses to start a plan)')

    def _on_cancel(self, _msg: String):
        """Someone cancelled the navigation. Forget the goal; do not resume it.

        Also clears the no-progress counter: the next goal is a fresh journey
        and must not inherit the count from the one that was abandoned.
        """
        with self._lock:
            had_goal = self._last_goal is not None
            self._last_goal = None
            self._last_goal_rx = None
        self._reissue_count = 0
        self._reissue_odom = None
        self._cancelled = True          # stops a recovery already in flight
        gh = self._active_behavior
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception as exc:                      # noqa: BLE001
                self.get_logger().warn(f'could not cancel the running behavior: {exc!r}')
        self._nudge_pub.publish(Twist())
        if had_goal or gh is not None:
            self.get_logger().info('Navigation cancelled — dropping the goal, not re-issuing')

    def _supersede_recovery(self):
        """Stop recovery work belonging to a navigation goal that was replaced."""
        self._cancelled = True
        gh = self._active_behavior
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception as exc:                      # noqa: BLE001
                self.get_logger().warn(
                    f'could not stop superseded recovery behavior: {exc!r}')
        self._nudge_pub.publish(Twist())
        self.get_logger().info('New navigation target superseded the in-flight recovery')

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
        window = [(t, x, y, yaw) for t, x, y, yaw in self._positions if t >= since]
        if len(window) < 2:
            return

        xs = [p[1] for p in window]
        ys = [p[2] for p in window]
        disp = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
        # Sum wrapped yaw deltas instead of max-min: crossing +pi/-pi must not
        # look like a 2*pi jump, and a turn that changes direction still counts.
        yaw_motion = sum(abs(math.atan2(math.sin(b[3] - a[3]),
                                        math.cos(b[3] - a[3])))
                         for a, b in zip(window, window[1:]))
        with self._lock:
            wants_linear = self._cmd_linear >= self._stuck_linear_cmd
            wants_rotation = self._cmd_angular >= self._stuck_angular_cmd

        if not wants_linear and not wants_rotation:
            # MPPI often emits a few hundredths near its final tolerance.  Do
            # not turn that harmless settling into a physical recovery move.
            with self._lock:
                self._cmd_active_since = now
            return

        progressed = ((wants_linear and disp >= self._min_disp) or
                      (wants_rotation and yaw_motion >= self._min_rotation))
        if progressed:
            # Start a fresh window.  Keeping the original start forever makes
            # old motion mask a later genuine stall during one continuous goal.
            with self._lock:
                self._cmd_active_since = now
            return

        self._declare_stuck(
            f'{elapsed:.1f}s with cmd_vel, displacement={disp:.3f}m, '
            f'rotation={yaw_motion:.3f}rad')

    def _declare_stuck(self, reason: str):
        """Shared entry point for both stuck-detection paths.

        _check_stuck (cmd_vel active but no displacement) and _on_nav_feedback
        (planner never even produces a cmd_vel) both land here so there is
        exactly one place that flips STATUS_STUCK and kicks off _run_recovery.
        """
        self.get_logger().warn(f'STUCK detected — {reason}')
        self._status = STATUS_STUCK
        self._publish_status(STATUS_STUCK)
        self._cancelled = False     # a new pinch, not the cancelled one
        threading.Thread(
            target=self._run_recovery, args=(self._max_attempts,), daemon=True
        ).start()

    # ── Recovery sequence ────────────────────────────────────────────

    def _run_recovery(self, attempts_left: int):
        # ‼️ Announce and give up BEFORE claiming to be recovering. This used to
        # say "Κόλλησα, προσπαθώ να ξεκολλήσω" on entry — including on the
        # recursive call that has no attempts left, so a full failure spoke the
        # line once per attempt and then contradicted itself in the same breath.
        if attempts_left <= 0:
            self._publish_status(STATUS_FAILED)
            self._speak('Δεν μπορώ να ξεκολλήσω. Χρειάζομαι βοήθεια.')
            # ‼️ Back to IDLE, otherwise this node is finished for the session:
            # both entry points (_check_stuck and _on_trigger) require IDLE, so
            # leaving _status at FAILED meant the FIRST exhausted recovery
            # silently disabled stuck detection until the next restart — no
            # log, no symptom, the robot simply never tries to free itself
            # again. The status is still published as FAILED above so whoever
            # is listening hears the outcome; only the internal gate reopens.
            self._status = STATUS_IDLE
            self._reset_cmd_tracking()
            return

        self._status = STATUS_RECOVERING
        self._speak('Κόλλησα, προσπαθώ να ξεκολλήσω.', progress=True)

        # Cancel any active navigation goal
        self._cancel_navigation()
        # Stale costmap data (esp. camera-fed voxel_layer marks left by a
        # corrupted depth frame) can pin the robot's own footprint against a
        # phantom obstacle — every stock recovery step then refuses with
        # "Collision Ahead" no matter how short the requested distance,
        # because the very first simulated step already overlaps it. A clear
        # costmap fixes exactly that without touching the map or amcl.
        self._clear_costmaps()

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
            # Checked between steps, not only at the top: the whole sequence is
            # ~20 s of the robot shuffling about, and a person who has just
            # pressed cancel is watching it do that.
            if self._cancelled:
                self.get_logger().info('Recovery abandoned — navigation was cancelled')
                self._abandon_recovery()
                return
            self.get_logger().info(f'Recovery step: {name}')
            step()
            time.sleep(1.0)
            if self._has_moved_recently(window=5.0, threshold=0.01):
                self._recovered()
                return

        if self._cancelled:
            self._abandon_recovery()
            return
        self.get_logger().warn(f'Still stuck, {attempts_left - 1} attempt(s) left')
        self._run_recovery(attempts_left - 1)

    def _abandon_recovery(self):
        self._nudge_pub.publish(Twist())        # whatever the last step left running
        self._publish_status(STATUS_IDLE)
        self._status = STATUS_IDLE
        self._reset_cmd_tracking()

    def _recovered(self):
        self._status = STATUS_RECOVERED
        self._publish_status(STATUS_RECOVERED)
        self._speak('Ξεκόλλησα!', progress=True)
        self.get_logger().info('Recovery succeeded')
        # Let the body settle, then resume the goal that was cancelled to run
        # the recovery — otherwise the robot just sits free after each pinch.
        time.sleep(2.0)
        if not self._cancelled:
            self._maybe_reissue_goal()
        self._status = STATUS_IDLE
        self._reset_cmd_tracking()

    def _reset_cmd_tracking(self):
        """Forget the command window that triggered this recovery.

        Under the lock: _on_cmd_vel writes these two fields from the executor
        thread while recovery runs on its own, so clearing them unlocked could
        drop a fresh 'active since' that arrived in between and re-fire STUCK
        immediately on a robot that had just started moving again.
        """
        with self._lock:
            self._cmd_active = False
            self._cmd_active_since = None
            self._cmd_linear = 0.0
            self._cmd_angular = 0.0

    def _maybe_reissue_goal(self):
        """Re-send the cancelled navigation goal (learned from /plan) so a
        multi-doorway goto completes hands-off. Capped by consecutive re-issues
        without net odom progress so an unreachable goal can't loop forever."""
        if not self._reissue_enabled:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        with self._lock:
            goal = self._last_goal
            age = None if self._last_goal_rx is None else now - self._last_goal_rx
        if goal is None or age is None or age > self._goal_max_age:
            return  # no fresh navigation to resume (e.g. a manual trigger test)

        # If the robot advanced since the last re-issue, this is a new pinch
        # further along the route → reset the no-progress counter.
        cur = self._positions[-1][1:3] if self._positions else None
        if cur is not None and self._reissue_odom is not None:
            if math.hypot(cur[0] - self._reissue_odom[0],
                          cur[1] - self._reissue_odom[1]) > self._reissue_prog:
                self._reissue_count = 0

        if self._reissue_count >= self._reissue_max:
            self.get_logger().warn(
                f'Not re-issuing: {self._reissue_count} re-issues without progress '
                f'— goal likely unreachable')
            self._speak('Δεν μπορώ να φτάσω στον στόχο.')
            return

        if not self._nav_ac.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('navigate_to_pose server unavailable; cannot re-issue')
            return

        ps = PoseStamped()
        ps.header.frame_id = goal.header.frame_id or 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = goal.pose
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = ps
        self._reissue_count += 1
        self._reissue_odom = cur
        self.get_logger().info(
            f're-issuing navigation goal (attempt {self._reissue_count}/'
            f'{self._reissue_max}) to ({goal.pose.position.x:.2f}, '
            f'{goal.pose.position.y:.2f})')
        self._nav_ac.send_goal_async(nav_goal)  # fire-and-forget; monitor re-catches re-sticks

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

    def _clear_costmaps(self):
        for name, client in (('local', self._clear_local_cli),
                              ('global', self._clear_global_cli)):
            if not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn(f'{name} costmap clear service not available')
                continue
            future = client.call_async(ClearEntireCostmap.Request())
            if self._await_future(future, 2.0) is None:
                self.get_logger().warn(f'{name} costmap clear timed out')

    def _roomiest_bearing(self):
        """Bearing (rad, robot frame) of the widest gap the lidar can see, or
        None without a usable scan. Sectors of 30°, scored by their CLOSEST
        return — a sector is only as passable as its nearest obstacle."""
        scan = self._scan
        if scan is None:
            return None
        width = math.radians(30)
        best_a, best_r = None, 0.0
        for k in range(int(2 * math.pi / width)):
            lo = -math.pi + k * width
            closest = float('inf')
            for i, r in enumerate(scan.ranges):
                if not (scan.range_min <= r <= scan.range_max):
                    continue
                a = scan.angle_min + i * scan.angle_increment
                a = math.atan2(math.sin(a), math.cos(a))
                if lo <= a < lo + width:
                    closest = min(closest, r)
            if closest != float('inf') and closest > best_r:
                best_a, best_r = lo + width / 2, closest
        if best_a is None:
            return None
        self.get_logger().info(
            f'Nudge: roomiest bearing {math.degrees(best_a):+.0f}° '
            f'({best_r:.2f} m clear)')
        return best_a

    def _do_nudge(self):
        """Raw cmd_vel escape, aimed rather than blind.

        This runs only after BackUp and DriveOnHeading have both refused, which
        on this robot means the costmap believes the footprint is already in
        contact — so there is no point asking the costmap where to go. Steer by
        the live scan instead: turn toward the widest gap, then creep into it.
        Turning a circular chassis in place sweeps no new ground
        (project_robot_collision_monitor_corners), so the turn itself cannot
        hit anything the robot is not already touching.
        """
        bearing = self._roomiest_bearing()
        if bearing is None:
            self.get_logger().info('Nudge: no scan — blind turn then creep')
            self._publish_twist_for(self._nudge_duration,
                                    angular_z=self._nudge_angular)
        else:
            turn_time = min(abs(bearing) / self._nudge_angular, 4.0)
            self._publish_twist_for(
                turn_time, angular_z=math.copysign(self._nudge_angular, bearing))
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
        self._active_behavior = gh
        result_future = gh.get_result_async()
        self._await_future(result_future, 15.0)
        self._active_behavior = None
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
        self._active_behavior = gh
        result_future = gh.get_result_async()
        self._await_future(result_future, 15.0)
        self._active_behavior = None
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
        self._active_behavior = gh
        result_future = gh.get_result_async()
        self._await_future(result_future, 15.0)
        self._active_behavior = None
        self.get_logger().info('Spin done')
        return True

    def _has_moved_recently(self, window: float, threshold: float) -> bool:
        now = self.get_clock().now().nanoseconds / 1e9
        recent = [(t, x, y) for t, x, y, _yaw in self._positions
                  if t >= now - window]
        if len(recent) < 2:
            return False
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        return math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) >= threshold

    # ── Helpers ──────────────────────────────────────────────────────

    def _publish_status(self, status: str):
        self._status_pub.publish(String(data=status))

    def _speak(self, text: str, progress: bool = False):
        """Say something. `progress` marks the running commentary.

        ‼️ 2026-08-04, from the owner: "λέει όλη την ώρα κόλλησα προσπαθώ να
        ξεκολλήσω, σταμάτα το να το λέει αυτό". One pinch point produced eight
        announcements in fifteen minutes — «Κόλλησα» and «Ξεκόλλησα» in pairs,
        every time, while the robot was working the problem perfectly well.
        The commentary is worthless to a person in the room: they can SEE it is
        stuck. What is worth saying out loud is the outcome it cannot fix, and
        that is still spoken. Everything else stays in the log and in
        recovery_status, which the dashboard shows.
        """
        if progress and not self._announce_progress:
            self.get_logger().info(f'[silent] {text}')
            return
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
