#!/usr/bin/env python3
"""
mission_executor_node.py — Multi-step mission state machine for the home robot.

Missions are triggered by publishing to `mission/start` (String).
Format: "patrol" | "find:<label>" | "dock" | "check_rooms"
Cancel any running mission: publish "" (empty) or "cancel" to `mission/start`.

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
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String


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

        self._locations  = _load_locations()
        self._state      = State.IDLE
        self._cancel_flag = threading.Event()
        self._lock       = threading.Lock()
        self._latest_objects: list = []

        self._nav_ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subscribers
        self.create_subscription(String, 'mission/start',     self._on_mission_start,   10)
        self.create_subscription(String, 'speech_text',       self._on_speech_text,     10)
        self.create_subscription(String, 'detected_objects',  self._on_objects,         10)
        self.create_subscription(String, 'tracked_objects',   self._on_objects,         10)

        # Publishers
        self._status_pub  = self.create_publisher(String, 'mission/status', 10)
        self._speech_pub  = self.create_publisher(String, 'speech_response', 10)

        self.get_logger().info('Mission executor ready — topics: mission/start, mission/status')

    # ── Subscriptions ────────────────────────────────────────────────

    def _on_objects(self, msg: String):
        try:
            self._latest_objects = json.loads(msg.data) or []
        except json.JSONDecodeError:
            pass

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

    def _navigate_to(self, location_name: str) -> bool:
        pose_data = self._locations.get(location_name)
        if not pose_data:
            self.get_logger().warn(f'Location "{location_name}" not in locations.yaml')
            return False

        if not self._nav_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('navigate_to_pose action server not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(pose_data.get('x', 0.))
        goal.pose.pose.position.y = float(pose_data.get('y', 0.))
        yaw = float(pose_data.get('yaw', 0.))
        goal.pose.pose.orientation.z = math.sin(yaw / 2.)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.)

        future = self._nav_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.)
        gh = future.result()
        if gh is None or not gh.accepted:
            return False

        # Poll until done or cancelled
        result_future = gh.get_result_async()
        while not result_future.done():
            if self._cancel_flag.is_set():
                gh.cancel_goal_async()
                return False
            time.sleep(0.2)
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=0.1)

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
