#!/usr/bin/env python3
"""
mission_executor_node.py — Multi-step mission state machine for the home robot.

Missions are triggered by publishing to `mission/start` (String).
Format: "patrol" | "find:<label>" | "fetch:<label>" | "dock" | "check_rooms"
Cancel any running mission: publish "" (empty) or "cancel" to `mission/start`.

`fetch:<label>` ("φέρε μου το X") composes the perception + nav + arm stacks:
resolve the object's map position from object memory (falling back to a live
room search), drive to an approach pose short of it, verify it's actually
there, grasp-and-hold it (pick_command hold=true), carry it back to where the
command was given, and release it (place_command). Needs use_perception
(object memory + detector) + use_arm + Nav2 all up. HW-untested.

States per mission: NAVIGATE → INSPECT → REPORT → (next waypoint) → DONE | FAILED

Publishes:
  mission/status  (String) — idle / navigating / inspecting / done / failed / cancelled
  speech_response (String) — TTS feedback
  /navigate_to_pose action → Nav2

Also monitors speech_text for "ακύρωσε" / "σταμάτα αποστολή" to cancel.

Locations are loaded from config/locations.yaml.
Objects are taken from the most recent `detected_objects` / `tracked_objects` message.
"""

import json
import math
import os
import threading
import time
from enum import Enum, auto

import yaml
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String

from home_robot.fetch_planner import (approach_pose, memory_target,
                                      nearest_detection, homing_twist)


class State(Enum):
    IDLE        = auto()
    NAVIGATING  = auto()
    INSPECTING  = auto()
    REPORTING   = auto()
    DONE        = auto()
    FAILED      = auto()
    CANCELLED   = auto()


def _load_locations() -> dict:
    try:
        path = os.path.join(
            get_package_share_directory('home_robot'), 'config', 'locations.yaml'
        )
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


LOCATION_NAMES_EL = {
    'living_room': 'σαλόνι',
    'bedroom':     'κρεβατοκάμαρα',
    'kitchen':     'κουζίνα',
    'dock':        'βάση φόρτισης',
}

CANCEL_PHRASES = {'ακύρωσε', 'ακυρωσε', 'σταμάτα', 'σταματα', 'σταμάτησε', 'σταματησε', 'cancel'}


class MissionExecutorNode(Node):
    def __init__(self):
        super().__init__('mission_executor_node')

        # fetch tuning
        self.declare_parameter('fetch_approach_dist', 0.4)   # m — park this short of the object
        self.declare_parameter('fetch_grasp_range', 1.0)     # m — object must be within this to grasp
        self.declare_parameter('memory_max_age', 300.0)      # s — older object-memory hits are re-verified live
        self.declare_parameter('pick_timeout', 40.0)         # s — wait for the arm to report
        self.declare_parameter('place_timeout', 30.0)
        self.declare_parameter('fetch_max_retries', 1)       # approach+pick retries
        self.declare_parameter('inspect_settle', 2.5)        # s — let the detector stabilise
        # Delivery: 'start_pose' returns to where the command was given; 'follow'
        # then homes in on the user's *current* position (they may have moved).
        self.declare_parameter('delivery_mode', 'start_pose')
        self.declare_parameter('deliver_distance', 0.8)      # m — how close to get to the user
        self.declare_parameter('homing_timeout', 25.0)       # s — give up homing, place anyway
        self.declare_parameter('search_speed', 0.4)          # rad/s — rotate to find the user
        self._fetch_approach_dist = self.get_parameter('fetch_approach_dist').value
        self._fetch_grasp_range   = self.get_parameter('fetch_grasp_range').value
        self._memory_max_age      = self.get_parameter('memory_max_age').value
        self._pick_timeout        = self.get_parameter('pick_timeout').value
        self._place_timeout       = self.get_parameter('place_timeout').value
        self._fetch_max_retries   = self.get_parameter('fetch_max_retries').value
        self._inspect_settle      = self.get_parameter('inspect_settle').value
        self._delivery_mode       = self.get_parameter('delivery_mode').value
        self._deliver_distance    = self.get_parameter('deliver_distance').value
        self._homing_timeout      = self.get_parameter('homing_timeout').value
        self._search_speed        = self.get_parameter('search_speed').value

        self._locations  = _load_locations()
        self._state      = State.IDLE
        self._cancel_flag = threading.Event()
        self._lock       = threading.Lock()
        self._latest_objects: list = []

        self._nav_ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Request/response state for the fetch mission (callbacks run on the main
        # executor; the mission worker thread waits on these Events — same
        # no-spin pattern as _await_future).
        self._mem_event   = threading.Event(); self._mem_value = None
        self._pick_event  = threading.Event(); self._pick_value = None
        self._place_event = threading.Event(); self._place_value = None

        # Subscribers
        self.create_subscription(String, 'mission/start',     self._on_mission_start,   10)
        self.create_subscription(String, 'speech_text',       self._on_speech_text,     10)
        self.create_subscription(String, 'detected_objects',  self._on_objects,         10)
        self.create_subscription(String, 'tracked_objects',   self._on_objects,         10)
        self.create_subscription(String, 'object_memory/answer', self._on_mem_answer,   10)
        self.create_subscription(String, 'pick_result',       self._on_pick_result,     10)
        self.create_subscription(String, 'place_result',      self._on_place_result,    10)

        # Publishers
        self._status_pub  = self.create_publisher(String, 'mission/status', 10)
        self._speech_pub  = self.create_publisher(String, 'speech_response', 10)
        self._mem_query_pub = self.create_publisher(String, 'object_memory/query', 10)
        self._pick_pub    = self.create_publisher(String, 'pick_command', 10)
        self._place_pub   = self.create_publisher(String, 'place_command', 10)
        # For the follow-delivery final approach (Nav2 gets us to the area; this
        # closes the last metre onto the user). obstacle_safety relays it to
        # cmd_vel_safe, same as person_follower.
        self._cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info('Mission executor ready — topics: mission/start, mission/status')

    # ── Subscriptions ────────────────────────────────────────────────

    def _on_objects(self, msg: String):
        try:
            self._latest_objects = json.loads(msg.data) or []
        except json.JSONDecodeError:
            pass

    def _on_mem_answer(self, msg: String):
        try:
            self._mem_value = json.loads(msg.data)
        except json.JSONDecodeError:
            self._mem_value = None
        self._mem_event.set()

    def _on_pick_result(self, msg: String):
        try:
            self._pick_value = json.loads(msg.data)
        except json.JSONDecodeError:
            self._pick_value = None
        self._pick_event.set()

    def _on_place_result(self, msg: String):
        try:
            self._place_value = json.loads(msg.data)
        except json.JSONDecodeError:
            self._place_value = None
        self._place_event.set()

    def _on_speech_text(self, msg: String):
        text = msg.data.lower().strip()
        if any(p in text for p in CANCEL_PHRASES) and 'αποστολ' in text:
            self._cancel_flag.set()

    def _on_mission_start(self, msg: String):
        cmd = msg.data.strip()
        if not cmd or cmd.lower() in {'cancel', 'ακύρωσε'}:
            self._cancel_flag.set()
            return

        with self._lock:
            if self._state != State.IDLE:
                self.get_logger().warn(f'Mission already running ({self._state}), ignoring "{cmd}"')
                return

        self._cancel_flag.clear()
        threading.Thread(target=self._dispatch, args=(cmd,), daemon=True).start()

    def _dispatch(self, cmd: str):
        if cmd == 'patrol':
            self._mission_patrol()
        elif cmd.startswith('find:'):
            label = cmd[5:].strip()
            self._mission_find(label)
        elif cmd.startswith('fetch:'):
            label = cmd[6:].strip()
            self._mission_fetch(label)
        elif cmd == 'dock':
            self._mission_dock()
        elif cmd == 'check_rooms':
            self._mission_check_rooms()
        else:
            self.get_logger().warn(f'Unknown mission: {cmd}')
            self._speak(f'Δεν ξέρω αποστολή "{cmd}".')

    # ── Mission: patrol ──────────────────────────────────────────────

    def _mission_patrol(self):
        self.get_logger().info('Mission: patrol')
        locs = [n for n in self._locations if n != 'dock']
        if not locs:
            self._speak('Δεν έχω καταχωρισμένα δωμάτια ακόμα.')
            return

        self._set_state(State.NAVIGATING)
        self._speak('Ξεκινώ περιπολία.')

        findings = {}
        for room in locs:
            if self._cancel_flag.is_set():
                break
            name_el = LOCATION_NAMES_EL.get(room, room)
            self._speak(f'Πηγαίνω {name_el}.')
            if not self._navigate_to(room):
                continue

            self._set_state(State.INSPECTING)
            time.sleep(2.0)  # let detector stabilize

            nearby = [o['label'] for o in self._latest_objects if o.get('z', 99.) < 3.5]
            findings[name_el] = nearby

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Η περιπολία ακυρώθηκε.')
            return

        # Report
        self._set_state(State.REPORTING)
        if any(findings.values()):
            parts = [f'{r}: {", ".join(items)}' for r, items in findings.items() if items]
            self._speak('Αναφορά περιπολίας. ' + '. '.join(parts))
        else:
            self._speak('Ολοκλήρωσα περιπολία. Δεν βρήκα τίποτα ξεχωριστό.')

        self._finish(State.DONE, '')

    # ── Mission: find ────────────────────────────────────────────────

    def _mission_find(self, label: str):
        self.get_logger().info(f'Mission: find "{label}"')
        locs = [n for n in self._locations if n != 'dock']
        if not locs:
            self._speak('Δεν έχω καταχωρισμένα δωμάτια.')
            return

        self._set_state(State.NAVIGATING)
        self._speak(f'Ψάχνω για {label}.')

        for room in locs:
            if self._cancel_flag.is_set():
                break
            name_el = LOCATION_NAMES_EL.get(room, room)
            self._speak(f'Ελέγχω {name_el}.')
            if not self._navigate_to(room):
                continue

            self._set_state(State.INSPECTING)
            time.sleep(2.5)

            found = [o for o in self._latest_objects
                     if o.get('label', '').lower() == label.lower() and o.get('z', 99.) < 4.]
            if found:
                dist = min(o.get('z', 0.) for o in found)
                self._finish(State.DONE, f'Βρήκα {label} στο {name_el}, περίπου {dist:.1f} μέτρα μπροστά!')
                return

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Αναζήτηση ακυρώθηκε.')
        else:
            self._finish(State.FAILED, f'Δεν βρήκα {label} πουθενά.')

    # ── Mission: fetch ("φέρε μου το X") ─────────────────────────────

    def _mission_fetch(self, label: str):
        self.get_logger().info(f'Mission: fetch "{label}"')
        # Deliver back to where the command was given = the robot's pose now.
        start_pose = self._lookup_base_pose()

        self._set_state(State.NAVIGATING)
        self._speak(f'Πάω να φέρω {label}.')

        # 1. RESOLVE — where is it?
        target = self._resolve_location(label)
        if target is None:
            self._finish(State.CANCELLED if self._cancel_flag.is_set() else State.FAILED,
                         'Ακυρώθηκε.' if self._cancel_flag.is_set() else f'Δεν ξέρω πού είναι {label}.')
            return

        # 2-4. approach → verify → grasp-and-hold, with retries
        picked = False
        for attempt in range(self._fetch_max_retries + 1):
            if self._cancel_flag.is_set():
                break
            rob = self._lookup_base_pose() or (target['x'] - 1.0, target['y'], 0.0)
            ax, ay, yaw = approach_pose((rob[0], rob[1]), (target['x'], target['y']),
                                        self._fetch_approach_dist)
            if not self._navigate_to_xy(ax, ay, yaw):
                continue

            self._set_state(State.INSPECTING)
            time.sleep(self._inspect_settle)
            det = nearest_detection(self._latest_objects, label, self._fetch_grasp_range)
            if det is None:
                # Memory was stale or the approach overshot — re-query and retry.
                self._speak(f'Δεν βλέπω {label} εδώ, ξανακοιτάω.')
                t2 = self._query_object_memory(label)
                if t2:
                    target = t2
                continue

            if self._do_pick(label, hold=True):
                picked = True
                break

        if not picked:
            self._finish(State.CANCELLED if self._cancel_flag.is_set() else State.FAILED,
                         'Ακυρώθηκε.' if self._cancel_flag.is_set() else f'Δεν κατάφερα να πιάσω {label}.')
            return

        # 5. carry back to the user (skip the drive if cancelled, but still
        #    release below so the arm isn't left holding the object)
        if not self._cancel_flag.is_set():
            self._set_state(State.NAVIGATING)
            self._speak(f'Σου φέρνω {label}.')
            if start_pose is not None:
                self._navigate_to_xy(*start_pose)      # Nav2 to the area
            if self._delivery_mode == 'follow':
                self._home_in_on_person()              # close onto the user now

        # 6. DELIVER — release the object
        self._do_place()

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, f'Ακυρώθηκε — άφησα {label}.')
        else:
            self._finish(State.DONE, f'Ορίστε {label}.')

    def _resolve_location(self, label: str):
        """Return {'x','y','room',...} for `label`, or None. Object memory first,
        then a live room-by-room search (object memory updates as we look)."""
        t = self._query_object_memory(label)
        if t:
            return t
        self._speak('Δεν το έχω στη μνήμη, ψάχνω στα δωμάτια.')
        for room in [n for n in self._locations if n != 'dock']:
            if self._cancel_flag.is_set():
                return None
            if not self._navigate_to(room):
                continue
            self._set_state(State.INSPECTING)
            time.sleep(self._inspect_settle)
            t = self._query_object_memory(label)
            if t:
                return t
        return None

    def _query_object_memory(self, label: str, timeout: float = 1.5):
        self._mem_event.clear()
        self._mem_value = None
        self._mem_query_pub.publish(String(data=label))
        if not self._mem_event.wait(timeout):
            return None
        return memory_target(self._mem_value, now=self._now(), max_age=self._memory_max_age)

    def _do_pick(self, label: str, hold: bool) -> bool:
        self._pick_event.clear()
        self._pick_value = None
        self._pick_pub.publish(String(data=json.dumps({'label': label, 'hold': hold})))
        if not self._pick_event.wait(self._pick_timeout):
            self._speak('Ο βραχίονας άργησε να απαντήσει.')
            return False
        return (self._pick_value or {}).get('status') == 'ok'

    def _do_place(self) -> bool:
        self._place_event.clear()
        self._place_value = None
        self._place_pub.publish(String(data='{}'))
        if not self._place_event.wait(self._place_timeout):
            return False
        return (self._place_value or {}).get('status') == 'ok'

    def _home_in_on_person(self):
        """Follow-delivery final approach: rotate to find the user, then drive
        onto them until within deliver_distance. Bounded by homing_timeout;
        always stops the base on exit. Falls through (places where it is) if the
        user is never found — better than driving forever."""
        self._set_state(State.INSPECTING)
        self._speak('Σε ψάχνω για να σου το δώσω.')
        deadline = time.monotonic() + self._homing_timeout
        arrived = False
        while time.monotonic() < deadline and not self._cancel_flag.is_set():
            person = nearest_detection(self._latest_objects, 'person', max_z=5.0)
            tw = Twist()
            if person is None:
                tw.angular.z = self._search_speed          # scan for the user
            else:
                lin, ang, arrived = homing_twist(
                    person, deliver_distance=self._deliver_distance)
                tw.linear.x, tw.angular.z = lin, ang
                if arrived:
                    self._cmd_vel_pub.publish(Twist())
                    break
            self._cmd_vel_pub.publish(tw)
            time.sleep(0.2)
        self._cmd_vel_pub.publish(Twist())                 # always stop
        if not arrived and not self._cancel_flag.is_set():
            self._speak('Δεν σε βρήκα, το αφήνω εδώ.')

    def _lookup_base_pose(self):
        """Current map-frame (x, y, yaw) of the robot, or None if TF isn't ready."""
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2. * (q.w * q.z + q.x * q.y),
                         1. - 2. * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # ── Mission: dock ────────────────────────────────────────────────

    def _mission_dock(self):
        if 'dock' not in self._locations:
            self._speak('Δεν έχω καταχωρίσει τη βάση φόρτισης.')
            return
        self._set_state(State.NAVIGATING)
        self._speak('Πηγαίνω στη βάση φόρτισης.')
        ok = self._navigate_to('dock')
        if ok and not self._cancel_flag.is_set():
            self._finish(State.DONE, 'Έφτασα στη βάση φόρτισης.')
        elif self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Ακύρωση πλεύσης προς βάση.')
        else:
            self._finish(State.FAILED, 'Δεν μπόρεσα να φτάσω στη βάση.')

    # ── Mission: check_rooms ─────────────────────────────────────────

    def _mission_check_rooms(self):
        self.get_logger().info('Mission: check_rooms')
        locs = [n for n in self._locations if n != 'dock']
        if not locs:
            self._speak('Δεν έχω δωμάτια.')
            return

        self._set_state(State.NAVIGATING)
        self._speak('Ελέγχω όλα τα δωμάτια.')

        for room in locs:
            if self._cancel_flag.is_set():
                break
            name_el = LOCATION_NAMES_EL.get(room, room)
            self._speak(f'Ελέγχω {name_el}.')
            if not self._navigate_to(room):
                continue
            self._set_state(State.INSPECTING)
            time.sleep(2.0)
            nearby = [o['label'] for o in self._latest_objects if o.get('z', 99.) < 3.5]
            if nearby:
                self._speak(f'{name_el}: βλέπω {", ".join(nearby[:3])}.')
            else:
                self._speak(f'{name_el}: τίποτα.')

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Έλεγχος ακυρώθηκε.')
        else:
            self._finish(State.DONE, 'Ολοκλήρωσα έλεγχο δωματίων.')

    # ── Navigation ───────────────────────────────────────────────────

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

    def _navigate_to(self, location_name: str) -> bool:
        pose_data = self._locations.get(location_name)
        if not pose_data:
            self.get_logger().warn(f'Location "{location_name}" not in locations.yaml')
            return False
        return self._navigate_to_xy(float(pose_data.get('x', 0.)),
                                    float(pose_data.get('y', 0.)),
                                    float(pose_data.get('yaw', 0.)))

    def _navigate_to_xy(self, x: float, y: float, yaw: float) -> bool:
        if not self._nav_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('navigate_to_pose action server not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.)

        future = self._nav_ac.send_goal_async(goal)
        gh = self._await_future(future, 10.)
        if gh is None or not gh.accepted:
            return False

        # Poll until done or cancelled. Do NOT spin here — the node's main
        # executor (rclpy.spin in main) services the action callbacks that
        # complete result_future; spinning from this worker thread would
        # attach the node to a second executor and stall the future.
        result_future = gh.get_result_async()
        while not result_future.done():
            if self._cancel_flag.is_set():
                gh.cancel_goal_async()
                return False
            time.sleep(0.1)

        result = result_future.result()
        return result is not None and result.status == GoalStatus.STATUS_SUCCEEDED

    # ── Helpers ──────────────────────────────────────────────────────

    def _set_state(self, state: State):
        self._state = state
        self._status_pub.publish(String(data=state.name.lower()))

    def _finish(self, final_state: State, message: str):
        self._set_state(final_state)
        if message:
            self._speak(message)
        self.get_logger().info(f'Mission finished: {final_state.name}')
        time.sleep(2.0)
        self._state = State.IDLE
        self._status_pub.publish(String(data='idle'))

    def _speak(self, text: str):
        self._speech_pub.publish(String(data=text))
        self.get_logger().info(f'[speech] {text}')


def main():
    rclpy.init()
    node = MissionExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
