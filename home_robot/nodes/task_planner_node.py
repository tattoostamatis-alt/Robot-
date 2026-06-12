#!/usr/bin/env python3
"""Task planner — executes `tidy`/`patrol` requests from llm_bridge_node by
driving Nav2 (NavigateToPose) to named locations and reporting detected
clutter at each stop on `speech_response`.

Subscribes:
- `tidy_command` (std_msgs/String, JSON {'room': 'living_room'|'bedroom'|
  'kitchen'|'all'}) — published by llm_bridge_node's `tidy` tool.
- `patrol_command` (std_msgs/Bool) — published by llm_bridge_node's `patrol`
  tool. Visits every known room once.
- `detected_objects` (std_msgs/String, JSON list from object_detector.py),
  cached for the clutter check after each arrival.

Publishes:
- `speech_response` (std_msgs/String) — progress narration, picked up by
  tts_node. Independent of llm_bridge_node's own immediate "started" reply.

Scope: navigate + look for clutter ("Plan -> Execute -> Verify" for the
"go check on a room" part of roadmap item 6). Does not physically pick
anything up — that needs the arm (roadmap items 16-21), not yet connected.
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


ROOM_ORDER = ['living_room', 'bedroom', 'kitchen']

ROOM_NAMES_EL = {
    'living_room': 'στο σαλόνι',
    'bedroom': 'στην κρεβατοκάμαρα',
    'kitchen': 'στην κουζίνα',
}


def _yaw_to_quaternion(yaw):
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        self.declare_parameter('nav_timeout', 120.0)
        self.declare_parameter('detect_wait', 2.0)

        self.nav_timeout = self.get_parameter('nav_timeout').value
        self.detect_wait = self.get_parameter('detect_wait').value

        locations_path = os.path.join(get_package_share_directory('home_robot'),
                                        'config', 'locations.yaml')
        with open(locations_path) as f:
            self.locations = yaml.safe_load(f)

        self.response_pub = self.create_publisher(String, 'speech_response', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self._latest_objects = None
        self._busy = threading.Lock()

        self.create_subscription(String, 'tidy_command', self._on_tidy, 10)
        self.create_subscription(Bool, 'patrol_command', self._on_patrol, 10)
        self.create_subscription(String, 'detected_objects', self._on_detected_objects, 10)

        self.get_logger().info('Task planner started')

    def _on_detected_objects(self, msg: String):
        try:
            self._latest_objects = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_tidy(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = {}
        room = data.get('room', 'all')
        rooms = ROOM_ORDER if room == 'all' else [room]
        self._run(self._tidy_run, rooms)

    def _on_patrol(self, msg: Bool):
        if not msg.data:
            return
        self._run(self._patrol_run, ROOM_ORDER)

    def _run(self, target, rooms):
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already executing a task, ignoring new request')
            return
        threading.Thread(target=self._wrapped, args=(target, rooms), daemon=True).start()

    def _wrapped(self, target, rooms):
        try:
            target(rooms)
        finally:
            self._busy.release()

    def _tidy_run(self, rooms):
        for room in rooms:
            self._visit_and_report(room)
        if len(rooms) > 1:
            self._say('Ολοκλήρωσα τον γύρο τακτοποίησης σε όλο το σπίτι.')

    def _patrol_run(self, rooms):
        self._say('Ξεκινάω περιπολία.')
        for room in rooms:
            self._visit_and_report(room)
        self._say('Η περιπολία τελείωσε.')

    def _visit_and_report(self, room):
        loc = self.locations.get(room)
        room_name = ROOM_NAMES_EL.get(room, room)
        if loc is None:
            self._say(f'Δεν ξέρω πού βρίσκεται το δωμάτιο "{room}".')
            return

        ok, reason = self._navigate(loc)
        if not ok:
            self._say(f'Δεν κατάφερα να φτάσω {room_name} ({reason}).')
            return

        clutter = self._check_clutter()
        if clutter:
            items = ', '.join(clutter)
            self._say(f'Έφτασα {room_name}. Βρήκα ακαταστασία: {items}.')
        else:
            self._say(f'Έφτασα {room_name}. Δεν βρήκα ακαταστασία.')

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
        goal_handle = send_future.result(timeout=10.0)
        if goal_handle is None:
            return False, 'καμία απάντηση από το Nav2'
        if not goal_handle.accepted:
            return False, 'ο στόχος απορρίφθηκε'

        result_future = goal_handle.get_result_async()
        result = result_future.result(timeout=self.nav_timeout)
        if result is None:
            return False, 'λήξη χρόνου πλοήγησης'
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            return False, 'η πλοήγηση απέτυχε'
        return True, None

    def _check_clutter(self):
        time.sleep(self.detect_wait)
        if not self._latest_objects:
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
