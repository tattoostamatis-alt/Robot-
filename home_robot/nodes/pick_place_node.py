#!/usr/bin/env python3
"""Pick-and-place — bridges object_detector.py detections to arm_driver.py.

NOT YET HW-VERIFIED end-to-end. The RoArm-M3 is connected (use_arm:=true), but
every motion still goes through arm_driver.py's documented unknowns (T:105
feedback field names, T:210 torque semantics). Re-verify slowly and by hand
before trusting a full grasp.

‼️ This paragraph used to add "and tf_base_arm is a guessed placeholder, not a
measurement". That was written 2026-07-06 and went stale on 2026-07-21
(9957842), when the transform WAS measured for the 879 chassis: 120 mm forward
of the wheel axle, centred, 150 mm above it (+36 mm wheel radius = z 0.186).
The stale line survived long enough to be read as a live blocker on
2026-08-03 and stall pick/fetch planning. If the arm is ever remounted, remeasure
it — but it is not an open item today.

Two modes on `pick_command` (JSON): default {"label":"cup"} picks and drops at
the drop pose; {"label":"cup","hold":true} grasps and lifts but keeps the object
in the gripper for transport (the fetch mission), and a later `place_command`
({} or {"x","y","z"}) releases it. `place_result` reports the release.

Flow, triggered by `pick_command` (std_msgs/String, JSON {"label": "cup"}
or {} for "whatever clutter object_detector.py last saw"):
  1. Look up the target in the latest `detected_objects` message
     (object_detector.py — x/y/z are metric, in the
     camera_color_optical_frame convention: x right, y down, z forward).
  2. Transform that point into `arm_base` via tf2 (the TF chain is
     object_detector's camera_color_optical_frame -> camera_link, published
     by the realsense driver itself, -> base_link -> arm_base, the latter
     two both static_transform_publishers from bringup.launch.py).
  3. Open gripper, hover above the target, then **visually servo** the XY
     alignment (see below) before descending, grasping, lifting, moving to
     the drop-off pose, releasing, and returning to init (T:100).

Visual servoing (closed-loop XY correction, param `servo_enabled`):
The D435 is body-mounted, not on the arm (eye-to-hand), so the object stays
in view while the arm moves. Rather than committing to one noisy snapshot,
the node re-reads detected_objects while hovering, re-locks the *same*
object (nearest same-label detection to the current estimate, so it can't
jump to a different cup), and nudges the hover XY toward it until the
estimate settles within `servo_tolerance` or `servo_max_iters` runs out.
This refines out per-frame detection jitter; it does NOT correct a
systematic base_link->arm_base calibration bias (the camera can't see the
gripper) — calibrate tf_base_arm for that. If detections stop mid-servo the
loop keeps the last good estimate and proceeds (graceful open-loop fallback).

Hand-eye correction (param `gripper_tag_enabled`):
The visual servoing above refines out per-frame detection jitter, but cannot
see (and so cannot correct) a systematic base_link->arm_base calibration
bias, because the body-mounted D435 never has the gripper in frame — see
module docstring above. A wrist-mounted AprilTag (id 1, frame `gripper_tag`,
config/apriltag.yaml) fixes that blind spot: once the arm reaches its hover
point, we look up where the tag says the gripper *actually* is (transformed
into arm_base the same tf2 way as any other target) and compare it against
where it was commanded to be. Any gap is exactly the calibration bias the
servoing loop can't touch, and gets cancelled with a few proportional
corrections (home_robot/servo_filter.py's hand_eye_correction) before the
object-detection servo runs. `gripper_tag_offset_xyz` is the tag's own
local-frame lever arm to the fingertip centre — measured 2026-08-10 as
150 mm along the tag's +Z (out of the printed face, which faces the same way
the gripper reaches). Off by default: NOT yet HW-verified, enable only after
checking it drives toward the target and not away from it.

Grasp orientation (param `grasp_orient_enabled`):
When object_detector.py runs the -seg model it attaches a grasp axis to each
clutter detection as two 3D fingertip contacts (grasp_contact_a/b, in the
camera frame) plus a grasp_width. We transform both contacts into arm_base via
the same tf2 path as the centroid, take the direction between them, and set the
wrist roll so the gripper closes ACROSS the object's short axis instead of at a
fixed orientation. `grasp_roll_sign`/`grasp_roll_offset` calibrate the gripper's
zero-roll convention; leave orientation off until roll is verified by hand on
the RoArm-M3 (T:104 'r' semantics are among arm_driver.py's documented unknowns).
grasp_width is published for future gripper pre-shaping but not yet commanded.

arm_driver.py exposes no "motion complete" feedback (T:105 reports
joint/EE pose, not a busy flag), so each step waits a fixed settle time
instead of polling for arrival — generous by design until real timing is
observed.
"""

import json
import math
import threading
import time

import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Float32, String
from tf2_geometry_msgs import do_transform_point

from home_robot.servo_filter import median_point, spread, converged, hand_eye_correction
from home_robot.stop_command import is_stop_command


CAMERA_FRAME = 'camera_color_optical_frame'
ARM_FRAME = 'arm_base'
# Wrist-mounted AprilTag (id 1, config/apriltag.yaml) — see module docstring,
# "Hand-eye correction".
GRIPPER_TAG_FRAME = 'gripper_tag'
# Where sightings are pinned. `odom` and not `map`: it is continuous and always
# available, and a grasp only cares about the last few metres of driving, which
# is exactly the span odom is trusted over.
ODOM_FRAME = 'odom'


class _Cancelled(Exception):
    """A stop arrived between two arm motions."""


class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')

        self.declare_parameter('approach_height', 0.10)   # m above target before descending
        self.declare_parameter('grasp_z_offset', 0.0)      # m added to target z at grasp (fingertip vs. detected centroid)
        self.declare_parameter('drop_x', 0.15)             # arm_base frame — placeholder, no tray/bin measured yet
        self.declare_parameter('drop_y', -0.15)
        self.declare_parameter('drop_z', 0.10)
        self.declare_parameter('gripper_open', 1.2)        # rad — JOINT_LIMITS['hand'] is 1.08..3.14, lower = more open
        self.declare_parameter('gripper_closed', 3.0)
        self.declare_parameter('arm_speed', 0)             # passed through to T:104 "spd", 0 = firmware default
        self.declare_parameter('movement_settle_time', 2.0)  # s — no completion feedback from arm_driver.py, see module docstring
        self.declare_parameter('tf_timeout', 2.0)
        self.declare_parameter('max_reach', 0.52)          # m from arm_base — see _run_pick
        self.declare_parameter('open_vocab_hold', 3.0)     # s — ride out flickering open-vocab frames
        self.declare_parameter('coco_hold', 3.0)            # s — same flicker fix as open_vocab_hold, for /detected_objects
        self.declare_parameter('pin_ttl', 30.0)            # s — how long a lost sighting stays usable
        # Visual servoing (closed-loop XY refinement while hovering)
        self.declare_parameter('servo_enabled', True)
        self.declare_parameter('servo_tolerance', 0.015)   # m — XY estimate settled → descend
        self.declare_parameter('servo_max_iters', 4)       # correction moves before giving up
        self.declare_parameter('servo_settle_time', 0.8)   # s — shorter than a full move; just a nudge
        self.declare_parameter('servo_relock_radius', 0.15)  # m — max jump to still count as the same object
        # Grasp orientation (align wrist roll to the object's short axis). Off by
        # default: T:104 'r' semantics and the gripper's zero-roll convention are
        # unverified on the RoArm-M3 — enable only after checking roll by hand.
        self.declare_parameter('grasp_orient_enabled', False)
        self.declare_parameter('grasp_roll_sign', 1.0)     # flip if roll turns the wrong way
        self.declare_parameter('grasp_roll_offset', 0.0)   # rad — gripper zero-roll correction
        # Hand-eye correction against the wrist AprilTag (see module docstring).
        # Off by default: NOT yet HW-verified.
        self.declare_parameter('gripper_tag_enabled', False)
        self.declare_parameter('gripper_tag_tolerance', 0.01)   # m — good enough, stop correcting
        self.declare_parameter('gripper_tag_max_iters', 3)      # correction moves before giving up
        self.declare_parameter('gripper_tag_settle_time', 0.8)  # s — a nudge, not a full move
        # Tag-local-frame lever arm to the fingertip centre — the tag sits at
        # the wrist, not at the grasp point. Measured 2026-08-10: 150 mm along
        # the tag's own +Z (out of the printed face), which faces the same way
        # the gripper reaches — x/y stay 0 (tag centred on the wrist, no
        # measured side-to-side or up/down offset).
        self.declare_parameter('gripper_tag_offset_x', 0.0)
        self.declare_parameter('gripper_tag_offset_y', 0.0)
        self.declare_parameter('gripper_tag_offset_z', 0.15)

        self.approach_height = self.get_parameter('approach_height').value
        self.grasp_z_offset = self.get_parameter('grasp_z_offset').value
        self.drop_pose = (self.get_parameter('drop_x').value,
                           self.get_parameter('drop_y').value,
                           self.get_parameter('drop_z').value)
        self.gripper_open = self.get_parameter('gripper_open').value
        self.gripper_closed = self.get_parameter('gripper_closed').value
        self.arm_speed = self.get_parameter('arm_speed').value
        self.movement_settle_time = self.get_parameter('movement_settle_time').value
        self.tf_timeout = self.get_parameter('tf_timeout').value
        self.max_reach = self.get_parameter('max_reach').value
        self.open_vocab_hold = self.get_parameter('open_vocab_hold').value
        self.coco_hold = self.get_parameter('coco_hold').value
        self.pin_ttl = self.get_parameter('pin_ttl').value
        # label -> ((x, y, z) in odom, monotonic stamp of the sighting)
        self._pinned = {}
        self.servo_enabled = self.get_parameter('servo_enabled').value
        self.servo_tolerance = self.get_parameter('servo_tolerance').value
        self.servo_max_iters = self.get_parameter('servo_max_iters').value
        self.servo_settle_time = self.get_parameter('servo_settle_time').value
        self.servo_relock_radius = self.get_parameter('servo_relock_radius').value
        self.grasp_orient_enabled = self.get_parameter('grasp_orient_enabled').value
        self.grasp_roll_sign = self.get_parameter('grasp_roll_sign').value
        self.grasp_roll_offset = self.get_parameter('grasp_roll_offset').value
        self.gripper_tag_enabled = self.get_parameter('gripper_tag_enabled').value
        self.gripper_tag_tolerance = self.get_parameter('gripper_tag_tolerance').value
        self.gripper_tag_max_iters = self.get_parameter('gripper_tag_max_iters').value
        self.gripper_tag_settle_time = self.get_parameter('gripper_tag_settle_time').value
        self.gripper_tag_offset = (self.get_parameter('gripper_tag_offset_x').value,
                                   self.get_parameter('gripper_tag_offset_y').value,
                                   self.get_parameter('gripper_tag_offset_z').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.raw_cmd_pub = self.create_publisher(String, 'arm/raw_cmd', 10)
        self.gripper_pub = self.create_publisher(Float32, 'arm/gripper_cmd', 10)
        self.response_pub = self.create_publisher(String, 'speech_response', 10)
        self.result_pub = self.create_publisher(String, 'pick_result', 10)
        self.place_result_pub = self.create_publisher(String, 'place_result', 10)

        self._latest_objects = None
        # Open-vocabulary hits are kept SEPARATE from the COCO ones, not merged
        # into _latest_objects: they arrive on their own topic at their own
        # (much slower) rate, and _servo/_relock must be able to re-read them
        # without a COCO frame overwriting the list in between.
        self._latest_open_vocab = None
        self._open_vocab_stamp = 0.0      # monotonic time of the last non-empty frame
        self._coco_stamp = 0.0            # monotonic time of the last non-empty COCO frame
        self._busy = threading.Lock()
        # A pick is ~10 s of blocking arm moves with no way to interrupt it.
        # Cancellation here is deliberately CONSERVATIVE: it stops before
        # starting the next motion, and only up to the moment the gripper
        # closes. Once the object is held, the lift completes — abandoning the
        # arm mid-descent, or opening the gripper in mid-air, is worse than
        # finishing the step.
        self._cancel = threading.Event()

        self.create_subscription(String, 'detected_objects', self._on_detected_objects, 10)
        # ‼️ Without this the arm could only ever grasp the 80 COCO classes, and
        # anything the user names outside them ("η παντόφλα") died at
        # _select_target with «δεν βλέπω κάτι να σηκώσω» — the failure observed
        # on 2026-08-06. open_vocab_detector publishes the SAME schema
        # (label/x/y/z in the camera optical frame), so the rest of the pipeline
        # needs no changes.
        self.create_subscription(String, 'open_vocab_detections',
                                 self._on_open_vocab_detections, 10)
        self.create_subscription(String, 'pick_command', self._on_pick_command, 10)
        # Release a held object (used by the fetch mission after carrying it to
        # the user). JSON {} → drop at the default drop pose, or {"x","y","z"}.
        self.create_subscription(String, 'place_command', self._on_place_command, 10)
        self.create_subscription(String, 'speech_text', self._on_speech_text, 10)

        self.get_logger().info('Pick-place node started (HW-unverified — see module docstring)')

    def _on_detected_objects(self, msg: String):
        """Latest COCO hits, with the same empty-frame hold-off as open_vocab.

        object_detector flickers the same way open_vocab_detector does (2026-08-14:
        a bottle at conf 0.74 was reported in only 5 of 37 consecutive frames) — an
        empty frame here does not erase a recent non-empty one until it is
        `coco_hold` seconds stale. See _on_open_vocab_detections for the original
        fix and why overwriting on every message is wrong.
        """
        try:
            objects = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        now = time.monotonic()
        if objects:
            self._latest_objects = objects
            self._coco_stamp = now
        elif now - self._coco_stamp > self.coco_hold:
            self._latest_objects = objects

    def _on_open_vocab_detections(self, msg: String):
        """Latest open-vocabulary hits, with an empty frame held off briefly.

        ‼️ These detections flicker. Confidences sit near the threshold (a
        slipper on the floor measured 0.25-0.33), so the detector emits an
        empty list every few frames for an object that is plainly there and
        being reported the rest of the time. Overwriting on every message meant
        a pick that arrived during one of those gaps answered «δεν βλέπω κάτι
        να σηκώσω» — seen on 2026-08-06 immediately after a successful approach
        drive, which made the approach itself look broken when it had worked.

        So an empty frame does not erase a recent non-empty one; it only takes
        effect once the last real sighting is `open_vocab_hold` seconds old,
        which is what stops a genuinely removed object from lingering forever.
        """
        try:
            objects = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        now = time.monotonic()
        if objects:
            self._latest_open_vocab = objects
            self._open_vocab_stamp = now
        elif now - self._open_vocab_stamp > self.open_vocab_hold:
            self._latest_open_vocab = []

    def _on_speech_text(self, msg: String):
        if is_stop_command(msg.data) and self._busy.locked():
            self.get_logger().warn('Spoken stop — cancelling the arm sequence')
            self._cancel.set()

    def _on_pick_command(self, msg: String):
        try:
            data = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            data = {}

        if data.get('cancel'):
            self._cancel.set()
            return

        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already executing a pick, ignoring new request')
            self._publish_result('error', 'busy with another pick')
            return
        # hold=true: grasp and lift but do NOT place/return — the object stays in
        # the gripper for transport (fetch). Default false = legacy pick-and-drop.
        hold = bool(data.get('hold', False))
        self._cancel.clear()
        threading.Thread(target=self._wrapped, args=(data.get('label'), hold), daemon=True).start()

    def _on_place_command(self, msg: String):
        try:
            data = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            data = {}
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Busy, ignoring place request')
            self.place_result_pub.publish(String(data=json.dumps({'status': 'error', 'detail': 'busy'})))
            return

        def _run():
            try:
                pose = (data.get('x', self.drop_pose[0]),
                        data.get('y', self.drop_pose[1]),
                        data.get('z', self.drop_pose[2]))
                self._release_sequence(pose)
                self._say('Ορίστε.')
                self.place_result_pub.publish(String(data=json.dumps({'status': 'ok', 'detail': 'placed'})))
            finally:
                self._busy.release()
        threading.Thread(target=_run, daemon=True).start()

    def _wrapped(self, label, hold):
        try:
            self._run_pick(label, hold)
        except _Cancelled:
            self.get_logger().warn('Pick cancelled before the grasp — arm returned to init')
            self._raw({'T': 100})            # known-safe pose, nothing held
            self._say('Σταμάτησα τον βραχίονα.')
            self._publish_result('cancelled', 'stopped by user')
        finally:
            self._busy.release()

    def _abort_point(self):
        """Raise if a stop arrived. Only ever called BEFORE a motion starts."""
        if self._cancel.is_set():
            raise _Cancelled()

    def _graspable(self, objects):
        """Detections with a real 3D position — the only ones the arm can use.

        open_vocab_detector reports x/y/z as None when depth was missing for
        that box (see its _locate docstring), and object_detector can do the
        same. Those would sail into _transform_to_arm_frame and fail there, or
        worse, be read as a point at the camera origin.
        """
        return [o for o in (objects or [])
                if o.get('x') is not None and o.get('y') is not None
                and o.get('z') is not None]

    def _select_target(self, label):
        coco = self._graspable(self._latest_objects)
        open_vocab = self._graspable(self._latest_open_vocab)

        # An explicitly named target is matched across BOTH detectors, and a
        # miss is now reported as a miss. It used to fall back to "any clutter
        # object", which meant «πιάσε την παντόφλα» could have the robot
        # confidently pick up an unrelated cup — worse than admitting it cannot
        # see the thing. The no-label case ({} = "whatever clutter", used by
        # sort/tidy) keeps the old behaviour.
        if label:
            named = [o for o in coco + open_vocab if o.get('label') == label]
            # ‼️ Most-confident, not first-seen. Live run 2026-08-06: asked for
            # a slipper, the open-vocabulary detector returned the real one at
            # 0.90 m (conf 0.33) AND four phantoms on a far wall at ~3 m (conf
            # 0.07-0.23). Taking whatever came first would send the arm at a
            # wall three metres away. Low-confidence open-vocab hits are normal
            # — the ranking is what makes them harmless.
            return max(named, key=lambda o: o.get('conf', 0.0)) if named else None

        clutter = [o for o in coco if o.get('clutter')]
        return max(clutter, key=lambda o: o.get('conf', 0.0)) if clutter else None

    def _run_pick(self, label, hold=False):
        target = self._select_target(label)
        if target is not None:
            self._remember_target(label, target)
            arm_point = self._transform_to_arm_frame(
                target['x'], target['y'], target['z'])
            if arm_point is None:
                self._publish_result('error', f'TF lookup {CAMERA_FRAME} -> {ARM_FRAME} failed (is tf_base_arm running?)')
                return
        else:
            # Out of view. For a floor object that is the NORMAL state once the
            # robot is close enough to grasp it, so fall back to the pinned
            # sighting rather than refusing — see _remember_target.
            arm_point = self._recall_arm_point(label) if label else None
            if arm_point is None:
                self._say('Δεν βλέπω κάτι να σηκώσω αυτή τη στιγμή.')
                self._publish_result('error', 'δεν βλέπω τέτοιο αντικείμενο μπροστά μου')
                return
            target = {'label': label}

        ax, ay, az = arm_point
        az += self.grasp_z_offset
        hover_z = az + self.approach_height

        # ‼️ Refuse what the arm physically cannot touch, BEFORE announcing the
        # pick. Without this the node answered «Πάω να σηκώσω: slipper» for a
        # detection three metres away, sent T:104 at an unreachable point and
        # left the user watching a motionless arm — the same "says it acted,
        # did not act" failure that made «πιάσε την παντόφλα» so confusing on
        # 2026-08-06. The reported reach of the RoArm-M3 is ~0.51 m from the
        # base; max_reach used to stay under it because that figure is for a
        # straight arm with nothing in the gripper.
        #
        # 2026-08-14: replaced with a real tape measurement — user measured
        # 640mm from the vacuum centre (base_link) straight out and DOWN to
        # the floor, the actual low reach used to grasp something on the
        # ground, not a spec-sheet straight-arm figure. Converted into the
        # arm_base frame this check runs in (arm_base sits 0.10m forward /
        # 0.18m up from base_link, see tf_base_arm in bringup.launch.py):
        # horizontal 0.64-0.10=0.54m, vertical drop to the floor 0.18m,
        # 3D distance sqrt(0.54^2 + 0.18^2) = 0.57m. max_reach stays a margin
        # under that measured ceiling, same as it stayed under the old spec
        # figure.
        distance = math.sqrt(ax * ax + ay * ay + az * az)
        if distance > self.max_reach:
            self._say(f'Το βλέπω, αλλά είναι πολύ μακριά για τον βραχίονα '
                      f'— {distance:.1f} μέτρα. Πρέπει να πλησιάσω πρώτα.')
            self.get_logger().warn(
                f'Target "{target["label"]}" at {distance:.2f} m exceeds '
                f'max_reach {self.max_reach:.2f} m — not moving the arm')
            # ‼️ Greek, because this string is handed to the LLM as the reason
            # the tool failed and it answers the user from it. With the English
            # 'out of reach (0.88 m)' the model said «Δεν ξέρω γιατί απέτυχα» —
            # it had the reason and did not use it.
            self._publish_result(
                'out_of_reach',
                f'το αντικείμενο είναι {distance:.2f} μέτρα μακριά, πιο πέρα '
                f'από όσο φτάνει ο βραχίονας ({self.max_reach:.2f} m) — '
                f'πρέπει να πλησιάσεις πρώτα',
                distance=round(distance, 3), max_reach=self.max_reach)
            return

        self._say(f'Πάω να σηκώσω: {target["label"]}.')
        self.get_logger().info(
            f'Pick target "{target["label"]}" at arm_base ({ax:.3f}, {ay:.3f}, {az:.3f})')

        self._abort_point()
        self._gripper(self.gripper_open)
        self._abort_point()

        # Hand-eye correction against the wrist tag runs FIRST: it targets a
        # systematic bias (see module docstring), and the object-detection
        # servo below should refine around the corrected point, not the
        # raw open-loop one.
        if self.gripper_tag_enabled:
            self._abort_point()
            ax, ay = self._hand_eye_correct(ax, ay, hover_z)
        else:
            self._cartesian(ax, ay, hover_z)

        # Closed-loop XY refinement while hovering (see module docstring).
        # servo_confident stays None (never downgrade the report) when servo
        # is disabled outright — only a run that actually lost the target or
        # never converged should make the final report less than "ok".
        servo_confident = None
        if self.servo_enabled:
            self._abort_point()
            ax, ay, az, servo_confident = self._servo(target['label'], ax, ay, az, hover_z)

        # Align the wrist to the object's short axis (grasp geometry from -seg).
        roll = self._grasp_roll(target)
        if roll is not None:
            self.get_logger().info(f'Grasp roll aligned to object axis: {math.degrees(roll):.0f}°')
            self._abort_point()
            self._cartesian(ax, ay, hover_z, roll=roll)  # re-orient before descending
        r = roll or 0.0

        # ‼️ LAST abort point. Past the descend+close pair the object is in the
        # gripper, and the lift below has to finish.
        self._abort_point()
        self._cartesian(ax, ay, az, roll=r)
        self._gripper(self.gripper_closed)
        self._cartesian(ax, ay, hover_z, roll=r)   # lift, object grasped

        # servo_confident is False when the loop lost sight of the target or
        # never converged — the grasp point was a stale open-loop estimate,
        # so the gripper may well have closed on nothing. Say so instead of
        # claiming success we never actually saw (2026-08-11 slipper: reported
        # «Τακτοποίησα» after "lost sight of target, using best estimate").
        uncertain = servo_confident is False

        # ‼️ servo_confident only certifies the XY lock BEFORE descending — it
        # says nothing about whether the close actually caught the object.
        # 2026-08-14 baby bottle, HW: servo converged (spread=0mm) and this
        # still reported «Τακτοποίησα» while the bottle sat knocked over on
        # the floor exactly where it started — the descend+close+lift has no
        # feedback of its own (arm/joint_states is not trustworthy either,
        # see project_robot_simple_command_gates memory). The one signal we
        # do have: if the object is still detected right where we grasped it,
        # closing on it obviously failed, confident servo or not.
        still_there = self._relock(target['label'], ax, ay)
        if still_there is not None:
            self.get_logger().warn(
                f'"{target["label"]}" still detected at the grasp point after '
                f'lifting — the gripper closed on nothing')
            uncertain = True

        if hold:
            # Keep it in the gripper for transport (fetch). A later place_command
            # releases it. Stay at hover so nothing drags on the way up.
            if uncertain:
                self._say(f'Ίσως δεν κράτησα σωστά το {target["label"]} — '
                          f'το έχασα από τα μάτια μου καθώς πλησίαζα.')
                self._publish_result('uncertain', target['label'])
            else:
                self._say(f'Το κρατάω: {target["label"]}.')
                self._publish_result('ok', target['label'])
            return

        # Legacy pick-and-drop: deposit at the drop pose and return to init.
        self._release_sequence(self.drop_pose)
        if uncertain:
            self._say(f'Ίσως δεν έπιασα σωστά το {target["label"]} — '
                      f'το έχασα από τα μάτια μου καθώς πλησίαζα.')
            self._publish_result('uncertain', target['label'])
        else:
            self._say(f'Τακτοποίησα: {target["label"]}.')
            self._publish_result('ok', target['label'])

    def _release_sequence(self, pose):
        """Move to `pose`, open the gripper to release, return to the init pose."""
        px, py, pz = pose
        self._cartesian(px, py, pz)
        self._gripper(self.gripper_open)
        self._raw({'T': 100})  # back to init pose
        time.sleep(self.movement_settle_time)

    def _servo(self, label, ax, ay, az, hover_z):
        """Refine (ax, ay, az) toward the live detection while hovering.

        Collects several relocked estimates and returns their robust
        (component-wise median) arm-frame target, so a single noisy frame —
        especially a bad RealSense depth — can't decide the grasp point. The
        hover XY tracks the running median between reads; the loop stops once the
        recent frames agree to within servo_tolerance. Keeps the best estimate so
        far if detections or TF drop out mid-loop (graceful open-loop fallback).

        Returns (x, y, z, confident). `confident` is only True when the loop
        actually converged on live detections — a lost-sight or exhausted-iters
        fallback still returns a usable point (the caller still tries the grasp)
        but the caller must not report success as confidently as a real lock,
        since the point can be a stale, systematically offset estimate (see
        module docstring)."""
        samples: list[tuple] = []          # raw arm-frame relock points (no z offset)
        ref_x, ref_y = ax, ay
        confident = False
        for i in range(self.servo_max_iters):
            locked = self._relock(label, ref_x, ref_y)
            if locked is None:
                self.get_logger().info('Servo: lost sight of target, using best estimate so far')
                break
            samples.append(locked)
            ref_x, ref_y, _ = median_point(samples)
            if converged(samples, self.servo_tolerance):
                self.get_logger().info(
                    f'Servo converged (spread={spread(samples)*1000:.0f}mm) after {i+1} frames')
                confident = True
                break
            self.get_logger().info(
                f'Servo frame {i+1}: est=({ref_x:.3f}, {ref_y:.3f}) '
                f'spread={spread(samples)*1000:.0f}mm')
            self._cartesian(ref_x, ref_y, hover_z, settle=self.servo_settle_time)
        est = median_point(samples)
        if est is None:
            return ax, ay, az, False       # never got a lock — keep the original
        return est[0], est[1], est[2] + self.grasp_z_offset, confident

    def _gripper_actual_point(self):
        """Arm-frame position of the true grasp point, from the wrist
        AprilTag — None if it isn't visible right now (out of the camera's
        frame, motion blur, occluded by the object being approached).
        `gripper_tag_offset` is applied in the TAG's own frame, so it rotates
        with the wrist the same way the tag's translation does."""
        ox, oy, oz = self.gripper_tag_offset
        return self._transform(ox, oy, oz, GRIPPER_TAG_FRAME, ARM_FRAME)

    def _hand_eye_correct(self, target_x, target_y, hover_z):
        """Drive the wrist tag's OBSERVED position onto (target_x, target_y),
        instead of trusting that commanding the arm there puts it there (see
        module docstring, "Hand-eye correction"). Falls back to the open-loop
        command untouched if the tag never comes into view — the same
        graceful degradation as _servo."""
        cmd_x, cmd_y = target_x, target_y
        for i in range(self.gripper_tag_max_iters):
            self._abort_point()
            self._cartesian(cmd_x, cmd_y, hover_z, settle=self.gripper_tag_settle_time)
            actual = self._gripper_actual_point()
            if actual is None:
                self.get_logger().info(
                    'Hand-eye: wrist tag not visible, keeping the open-loop estimate')
                return cmd_x, cmd_y
            (cmd_x, cmd_y), done = hand_eye_correction(
                (cmd_x, cmd_y), (target_x, target_y), (actual[0], actual[1]),
                self.gripper_tag_tolerance)
            if done:
                self.get_logger().info(f'Hand-eye converged after {i + 1} move(s)')
                return cmd_x, cmd_y
            self.get_logger().info(
                f'Hand-eye frame {i + 1}: tag at ({actual[0]:.3f}, {actual[1]:.3f}), '
                f'next command ({cmd_x:.3f}, {cmd_y:.3f})')
        return cmd_x, cmd_y

    def _grasp_roll(self, target):
        """Wrist-roll angle (rad) that aligns the gripper opening across the
        object's short axis, from object_detector.py's grasp contacts. Both
        contacts go through the same camera->arm_base tf2 path as the centroid,
        so the roll is computed in the arm frame. None if orientation is
        disabled, the detection has no grasp axis, or TF drops out."""
        if not self.grasp_orient_enabled:
            return None
        ca, cb = target.get('grasp_contact_a'), target.get('grasp_contact_b')
        if not ca or not cb:
            return None
        pa = self._transform_to_arm_frame(*ca)
        pb = self._transform_to_arm_frame(*cb)
        if pa is None or pb is None:
            return None
        yaw = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
        roll = self.grasp_roll_sign * yaw + self.grasp_roll_offset
        # Wrap to [-pi/2, pi/2]: a parallel-jaw grasp axis is symmetric, so a
        # 180deg flip is the same grasp — keep the wrist move small.
        while roll > math.pi / 2:
            roll -= math.pi
        while roll < -math.pi / 2:
            roll += math.pi
        return roll

    def _relock(self, label, ref_ax, ref_ay):
        """Among the latest detections of `label`, return the arm-frame (x,y,z)
        of the one nearest the current estimate (so we track the *same* object,
        not jump to another). None if none within servo_relock_radius."""
        best, best_d = None, self.servo_relock_radius
        # Both detectors, same as _select_target — an open-vocab target that
        # could be selected must also be re-lockable, or servoing silently
        # degrades to open loop for exactly the objects that need it most.
        for o in (self._graspable(self._latest_objects)
                  + self._graspable(self._latest_open_vocab)):
            if o.get('label') != label:
                continue
            p = self._transform_to_arm_frame(o['x'], o['y'], o['z'])
            if p is None:
                continue
            d = math.hypot(p[0] - ref_ax, p[1] - ref_ay)
            if d <= best_d:
                best, best_d = p, d
        return best

    def _transform_to_arm_frame(self, x, y, z):
        return self._transform(x, y, z, CAMERA_FRAME, ARM_FRAME)

    def _transform(self, x, y, z, src, dst):
        point = PointStamped()
        point.header.frame_id = src
        point.header.stamp = rclpy.time.Time().to_msg()  # latest available transform
        point.point.x, point.point.y, point.point.z = x, y, z
        try:
            transform = self.tf_buffer.lookup_transform(
                dst, src, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout))
        except tf2_ros.TransformException as e:
            self.get_logger().error(f'TF lookup {src} -> {dst} failed: {e}')
            return None
        out = do_transform_point(point, transform)
        return out.point.x, out.point.y, out.point.z

    def _remember_target(self, label, target):
        """Pin a sighting to `odom`, which does not move when the robot does.

        ‼️ This is what makes approaching a floor object possible at all. The
        D435 is body-mounted and looks FORWARD, so something on the floor
        leaves the bottom of the frame as the robot closes in — measured
        2026-08-06, a slipper tracked cleanly at 1.17 m, 0.97 m and 0.70 m and
        then vanished, at exactly the point where it was finally near enough to
        grasp. Detections are in the camera frame, so they are worthless the
        moment the robot moves; in `odom` the sighting stays put and TF does
        the arithmetic. Losing sight of the target is then normal and expected,
        not a failure.
        """
        pinned = self._transform(target['x'], target['y'], target['z'],
                                 CAMERA_FRAME, ODOM_FRAME)
        if pinned is not None:
            self._pinned[label] = (pinned, time.monotonic())

    def _recall_arm_point(self, label):
        """Arm-frame position of the last sighting, if it is still fresh."""
        entry = self._pinned.get(label)
        if entry is None:
            return None
        (ox, oy, oz), stamp = entry
        age = time.monotonic() - stamp
        if age > self.pin_ttl:
            return None
        point = self._transform(ox, oy, oz, ODOM_FRAME, ARM_FRAME)
        if point is not None:
            self.get_logger().info(
                f'"{label}" out of view — using the sighting from {age:.1f}s ago')
        return point

    def _cartesian(self, x, y, z, settle=None, roll=0.0):
        self._raw({'T': 104, 'x': x, 'y': y, 'z': z, 't': 0, 'r': roll, 'spd': self.arm_speed})
        time.sleep(self.movement_settle_time if settle is None else settle)

    def _gripper(self, pos):
        self.gripper_pub.publish(Float32(data=pos))
        time.sleep(self.movement_settle_time)

    def _raw(self, cmd: dict):
        self.raw_cmd_pub.publish(String(data=json.dumps(cmd)))

    def _say(self, text):
        self.get_logger().info(f'Pick-place: {text}')
        self.response_pub.publish(String(data=text))

    def _publish_result(self, status, detail, **extra):
        """`extra` carries machine-readable fields alongside the spoken detail.

        out-of-reach sends distance/max_reach so the caller can work out how far
        to drive, instead of parsing metres back out of a Greek sentence.
        """
        payload = {'status': status, 'detail': detail}
        payload.update(extra)
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def main():
    rclpy.init()
    node = PickPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
