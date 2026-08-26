#!/usr/bin/env python3
"""Task planner — executes `tidy`/`patrol` requests from llm_bridge_node by
driving Nav2 (NavigateToPose) to named locations and reporting detected
clutter at each stop on `speech_response`.

Subscribes:
- `tidy_command` (std_msgs/String, JSON {'room': <a key of locations.yaml>
  |'all'}) — published by llm_bridge_node's `tidy` tool.
- `patrol_command` (std_msgs/Bool) — published by llm_bridge_node's `patrol`
  tool. Visits every known room once.
- `detected_objects` (std_msgs/String, JSON list from object_detector.py),
  cached for the clutter check after each arrival.

Publishes:
- `speech_response` (std_msgs/String) — progress narration, picked up by
  tts_node. Independent of llm_bridge_node's own immediate "started" reply.
- `pick_command` (std_msgs/String, JSON {'label': ..., 'hold': true}) — only
  when the `use_arm` parameter is true, one per clutter item found at a stop,
  consumed by pick_place_node.py. `hold:true` keeps the item in the gripper
  instead of pick_place_node's own local drop pose.
- `place_command` (std_msgs/String, JSON {}) — sent once the robot has
  navigated to `drop_location` (a locations.yaml entry, default "gonia")
  carrying a held item. Releases it at pick_place_node's default drop pose,
  which — with the robot physically parked at the corner — leaves the item
  there.

Scope: navigate + look for clutter ("Plan -> Execute -> Verify" for the
"go check on a room" part of roadmap item 6), and — when use_arm:=true — for
each item found: pick it (held, not dropped in place), carry it to
`drop_location`, place it, then navigate back to the room before the next
item (pick_place_node re-detects from wherever the robot is standing, so it
has to physically be back in the room to find what's left).

‼️ 2026-08-15, NOT YET HW-VERIFIED END-TO-END. Individual pick/hold/place
calls are each proven separately (pick_place_node.py, HW-verified twice for
plain pick-and-drop), but this multi-item, multi-trip loop through Nav2 has
never run live. `drop_location` also has to be taught first —
`record_location.py gonia` (drive there) or `record_location.py --click
gonia` (click on the map) — tidy silently drops items in place at
whatever room it's in if that name is missing from locations.yaml.
"""

import json
import math
import os
import threading
import time

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool, String

from home_robot.status_query import (room_el, room_locative_el,
                                     rooms_from_locations)
from home_robot.stop_command import is_stop_command


def _yaw_to_quaternion(yaw):
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # 120 s was not enough for one leg of the house: the first patrol on
        # malou2 (2026-07-31) gave up on diadromos at exactly 120.0 s, more
        # than 5 m out, while the robot was still driving and Nav2 had not
        # failed. This is the give-up guard for a leg, not a target time.
        self.declare_parameter('nav_timeout', 240.0)
        self.declare_parameter('detect_wait', 2.0)
        self.declare_parameter('use_arm', False)
        self.declare_parameter('pick_timeout', 30.0)
        # Named entry in locations.yaml where tidied clutter gets dropped off.
        # Teach it like any other room: drive there and
        # `record_location.py gonia`, or `record_location.py --click gonia`.
        self.declare_parameter('drop_location', 'gonia')

        self.nav_timeout = self.get_parameter('nav_timeout').value
        self.detect_wait = self.get_parameter('detect_wait').value
        self.use_arm = self.get_parameter('use_arm').value
        self.pick_timeout = self.get_parameter('pick_timeout').value
        self.drop_location = self.get_parameter('drop_location').value

        locations_path = os.path.join(get_package_share_directory('home_robot'),
                                        'config', 'locations.yaml')
        with open(locations_path) as f:
            self.locations = yaml.safe_load(f)
        self.room_order = rooms_from_locations(self.locations)

        self.response_pub = self.create_publisher(String, 'speech_response', 10)
        self.pick_pub = self.create_publisher(String, 'pick_command', 10)
        self.place_pub = self.create_publisher(String, 'place_command', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self._latest_objects = None
        self._objects_rx = 0.0            # monotonic time of the last detection msg
        self._busy = threading.Lock()
        self._pick_event = threading.Event()
        self._pick_result = None
        self._place_event = threading.Event()
        self._place_result = None
        # ‼️ There was NO way to stop a running task. llm_bridge's emergency stop
        # publishes patrol_command=False, and _on_patrol dropped it on the floor
        # ("if not msg.data: return"), so a patrol kept visiting rooms and
        # issuing fresh Nav2 goals: the robot halted for a fraction of a second
        # on the zeroed cmd_vel and then drove off again on the next goal.
        self._cancel = threading.Event()

        self.create_subscription(String, 'tidy_command', self._on_tidy, 10)
        self.create_subscription(Bool, 'patrol_command', self._on_patrol, 10)
        self.create_subscription(String, 'detected_objects', self._on_detected_objects, 10)
        self.create_subscription(String, 'pick_result', self._on_pick_result, 10)
        self.create_subscription(String, 'place_result', self._on_place_result, 10)
        # Belt and braces: honour a spoken stop directly, so a task still stops
        # when llm_bridge is down (or busy failing) and never relays the cancel.
        self.create_subscription(String, 'speech_text', self._on_speech_text, 10)

        self.get_logger().info('Task planner started')

    def _on_detected_objects(self, msg: String):
        try:
            self._latest_objects = json.loads(msg.data)
            self._objects_rx = time.monotonic()
        except json.JSONDecodeError:
            pass

    def _on_speech_text(self, msg: String):
        if is_stop_command(msg.data) and self._busy.locked():
            self.get_logger().warn('Spoken stop — cancelling the running task')
            self._cancel.set()

    def _on_pick_result(self, msg: String):
        try:
            self._pick_result = json.loads(msg.data)
        except json.JSONDecodeError:
            self._pick_result = None
        self._pick_event.set()

    def _on_place_result(self, msg: String):
        try:
            self._place_result = json.loads(msg.data)
        except json.JSONDecodeError:
            self._place_result = None
        self._place_event.set()

    def _on_tidy(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = {}
        if data.get('cancel'):
            self._cancel.set()
            return
        room = data.get('room', 'all')
        rooms = self.room_order if room == 'all' else [room]
        self._run(self._tidy_run, rooms)

    def _on_patrol(self, msg: Bool):
        if not msg.data:
            # ‼️ THE cancel signal, not a no-op. This is what llm_bridge's
            # emergency stop sends, and there is one _busy lock for the whole
            # node, so cancelling "the patrol" cancels whatever task is running
            # — tidy included, which has no stop signal of its own.
            self._cancel.set()
            return
        self._run(self._patrol_run, self.room_order)

    def _run(self, target, rooms):
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already executing a task, ignoring new request')
            return
        self._cancel.clear()
        threading.Thread(target=self._wrapped, args=(target, rooms), daemon=True).start()

    def _wrapped(self, target, rooms):
        try:
            target(rooms)
        finally:
            self._busy.release()

    def _tidy_run(self, rooms):
        for room in rooms:
            if self._cancel.is_set():
                break
            self._visit_and_report(room)
        if self._cancel.is_set():
            self._say('Σταμάτησα την τακτοποίηση.')
        elif len(rooms) > 1:
            self._say('Ολοκλήρωσα τον γύρο τακτοποίησης σε όλο το σπίτι.')

    def _patrol_run(self, rooms):
        self._say('Ξεκινάω περιπολία.')
        for room in rooms:
            if self._cancel.is_set():
                break
            self._visit_and_report(room)
        if self._cancel.is_set():
            self._say('Σταμάτησα την περιπολία.')
        else:
            self._say('Η περιπολία τελείωσε.')

    def _visit_and_report(self, room):
        loc = self.locations.get(room)
        room_name = room_locative_el(room)
        if loc is None:
            self._say(f'Δεν ξέρω πού βρίσκεται το δωμάτιο {room_el(room)}.')
            return

        ok, reason = self._navigate(loc)
        if self._cancel.is_set():
            return          # the run loop announces the cancellation once
        if not ok:
            self._say(f'Δεν κατάφερα να φτάσω {room_name} ({reason}).')
            return

        clutter = self._check_clutter()
        if not clutter:
            self._say(f'Έφτασα {room_name}. Δεν βρήκα ακαταστασία.')
            return

        items = ', '.join(clutter)
        self._say(f'Έφτασα {room_name}. Βρήκα ακαταστασία: {items}.')

        if self.use_arm:
            for label in clutter:
                if self._cancel.is_set():
                    return
                # Each item gets its own full round trip: pick here (held in
                # the gripper, not dropped locally), carry it to drop_location,
                # place, then come BACK to this room's pose before the next
                # item — pick_place_node re-detects from wherever the robot is
                # standing, so it has to physically be here again to find what
                # is left.
                if not self._pick(label, hold=True):
                    continue
                if self._cancel.is_set():
                    return
                self._deliver_to_corner()
                if self._cancel.is_set():
                    return
                ok, reason = self._navigate(loc)
                if self._cancel.is_set():
                    return
                if not ok:
                    self._say(f'Δεν κατάφερα να επιστρέψω {room_name} ({reason}) — '
                              'σταματάω την τακτοποίηση εδώ.')
                    return

    def _pick(self, label, hold=False):
        """Returns True only if the grasp succeeded."""
        self._pick_event.clear()
        self._pick_result = None
        self.pick_pub.publish(String(data=json.dumps({'label': label, 'hold': hold})))
        if not self._pick_event.wait(timeout=self.pick_timeout):
            self._say(f'Δεν κατάφερα να σηκώσω το {label} (λήξη χρόνου).')
            return False
        if not self._pick_result or self._pick_result.get('status') != 'ok':
            self._say(f'Δεν κατάφερα να σηκώσω το {label}.')
            return False
        return True

    def _deliver_to_corner(self):
        loc = self.locations.get(self.drop_location)
        if loc is None:
            self._say(f'Δεν έχω διδαχθεί τη θέση "{self.drop_location}" — '
                      'αφήνω το αντικείμενο εδώ.')
            self._place()
            return
        ok, reason = self._navigate(loc)
        if self._cancel.is_set():
            return
        if not ok:
            self._say(f'Δεν έφτασα στη γωνιά ({reason}) — αφήνω το αντικείμενο εδώ.')
        self._place()

    def _place(self):
        self._place_event.clear()
        self._place_result = None
        self.place_pub.publish(String(data=json.dumps({})))
        if not self._place_event.wait(timeout=self.pick_timeout):
            self._say('Χρονο-όριο στην απόθεση.')
            return
        if not self._place_result or self._place_result.get('status') != 'ok':
            self._say('Δεν κατάφερα να αφήσω το αντικείμενο.')

    def _await_future(self, future, timeout_sec):
        """Wait for an async future from this worker thread WITHOUT spinning —
        the node's main executor (rclpy.spin in main) services the callbacks
        that complete it. Spinning here would attach the node to a second
        executor and the future would never complete. Returns None on timeout.
        (Note: rclpy Future.result() takes no timeout argument.)"""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)
        return future.result() if future.done() else None

    def _navigate(self, loc):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            return False, 'το Nav2 δεν είναι έτοιμο'

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(loc['x'])
        goal.pose.pose.position.y = float(loc['y'])
        goal.pose.pose.orientation = _yaw_to_quaternion(float(loc['yaw']))

        send_future = self.nav_client.send_goal_async(goal)
        goal_handle = self._await_future(send_future, 10.0)
        if goal_handle is None:
            return False, 'καμία απάντηση από το Nav2'
        if not goal_handle.accepted:
            return False, 'ο στόχος απορρίφθηκε'

        result_future = goal_handle.get_result_async()
        # ‼️ Poll so a cancel can interrupt the leg, and — the part that was
        # missing — CANCEL THE GOAL on the way out. Giving up on the future
        # while leaving the goal active meant "λήξη χρόνου πλοήγησης" was a lie:
        # Nav2 kept driving to that room while the planner moved to the next
        # one, and on the last room of a patrol nothing preempted it at all.
        deadline = time.monotonic() + self.nav_timeout
        while rclpy.ok() and not result_future.done():
            if self._cancel.is_set():
                goal_handle.cancel_goal_async()
                return False, 'ακυρώθηκε'
            if time.monotonic() >= deadline:
                goal_handle.cancel_goal_async()
                return False, 'λήξη χρόνου πλοήγησης'
            time.sleep(0.05)

        result = result_future.result()
        if result is None:
            return False, 'η πλοήγηση απέτυχε'
        if result.status == GoalStatus.STATUS_CANCELED:
            return False, 'ακυρώθηκε'
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            return False, 'η πλοήγηση απέτυχε'
        return True, None

    def _check_clutter(self):
        # Interruptible settle: a cancel during the detector wait should not
        # have to sit through it before the next room is skipped.
        self._cancel.wait(self.detect_wait)
        if self._cancel.is_set() or not self._latest_objects:
            return []
        # ‼️ Only trust detections that arrived while we were standing HERE.
        # _latest_objects is whatever came last, so a detector that went quiet
        # (camera crashed, YOLO wedged) left the previous room's objects in
        # place and this room got reported with the last room's clutter.
        age = time.monotonic() - self._objects_rx
        if age > self.detect_wait * 2:
            self.get_logger().warn(
                f'detected_objects is {age:.1f}s stale — reporting no clutter '
                'rather than the previous room\'s')
            return []
        return [o['label'] for o in self._latest_objects if o.get('clutter')]

    def _say(self, text):
        self.get_logger().info(f'Task planner: {text}')
        self.response_pub.publish(String(data=text))


def main():
    rclpy.init()
    node = TaskPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
