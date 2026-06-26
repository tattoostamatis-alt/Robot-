#!/usr/bin/env python3
"""
situational_awareness_node.py — Aggregates sensor state into a compact JSON
context that llm_bridge_node prepends to every LLM turn automatically.

Sources:
  - /odom       → nearest named room (from config/locations.yaml)
  - detected_objects / tracked_objects → nearby objects summary
  - battery/state → charge percentage
  - psutil       → CPU / RAM

Publishes: situation_context (std_msgs/String, JSON) at ~1 Hz

llm_bridge_node subscribes to this and includes it as an extra system message
so Max always knows his environment without needing explicit tool calls.
"""

import json
import math
import os

import psutil
import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory


def _load_locations() -> dict:
    try:
        path = os.path.join(
            get_package_share_directory('home_robot'), 'config', 'locations.yaml'
        )
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _nearest_room(x: float, y: float, locations: dict) -> str:
    best_name, best_dist = 'άγνωστο', float('inf')
    for name, pose in locations.items():
        lx = pose.get('x', 0.)
        ly = pose.get('y', 0.)
        # Skip placeholder coordinates (all zeros = location not set)
        if lx == 0. and ly == 0.:
            continue
        d = math.sqrt((x - lx)**2 + (y - ly)**2)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


ROOM_NAMES_EL = {
    'living_room': 'σαλόνι',
    'bedroom':     'κρεβατοκάμαρα',
    'kitchen':     'κουζίνα',
}


class SituationalAwarenessNode(Node):
    def __init__(self):
        super().__init__('situational_awareness_node')

        self.declare_parameter('update_hz',      1.0)
        self.declare_parameter('max_obj_range',  3.5)   # meters — ignore farther objects
        self.declare_parameter('max_obj_count',  5)     # cap objects listed in context

        self._max_range  = self.get_parameter('max_obj_range').value
        self._max_obj    = self.get_parameter('max_obj_count').value
        hz               = self.get_parameter('update_hz').value

        self._locations  = _load_locations()
        self._odom_x     = 0.
        self._odom_y     = 0.
        self._objects    = []
        self._battery    = None   # BatteryState

        self.create_subscription(Odometry,     '/odom',           self._on_odom,    10)
        self.create_subscription(BatteryState, 'battery/state',   self._on_battery, 10)
        # Prefer tracked_objects (stable IDs) but fall back to detected_objects
        self.create_subscription(String, 'tracked_objects',  self._on_objects, 10)
        self.create_subscription(String, 'detected_objects', self._on_objects, 10)

        self._pub = self.create_publisher(String, 'situation_context', 10)
        self.create_timer(1.0 / hz, self._publish)

        self.get_logger().info('Situational awareness node ready')

    # ── Subscriptions ────────────────────────────────────────────────

    def _on_odom(self, msg: Odometry):
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y

    def _on_battery(self, msg: BatteryState):
        self._battery = msg

    def _on_objects(self, msg: String):
        try:
            self._objects = json.loads(msg.data) or []
        except json.JSONDecodeError:
            pass

    # ── Context assembly ─────────────────────────────────────────────

    def _publish(self):
        # Room
        room_key = _nearest_room(self._odom_x, self._odom_y, self._locations)
        room_el  = ROOM_NAMES_EL.get(room_key, room_key)

        # Nearby objects (sorted by distance, capped)
        nearby = sorted(
            [o for o in self._objects if o.get('z', 99.) <= self._max_range],
            key=lambda o: o.get('z', 99.)
        )[:self._max_obj]

        if nearby:
            obj_parts = [f"{o['label']}@{o.get('z', 0.):.1f}m" for o in nearby]
            objects_str = ', '.join(obj_parts)
        else:
            objects_str = 'κανένα αντικείμενο κοντά'

        # Battery
        batt_pct = None
        if self._battery is not None:
            pct = self._battery.percentage
            if not math.isnan(pct) and 0. <= pct <= 1.:
                batt_pct = round(pct * 100., 1)

        # System
        cpu  = psutil.cpu_percent(interval=None)
        ram  = psutil.virtual_memory().percent

        ctx = {
            'room':    room_el,
            'objects': objects_str,
            'cpu_pct': round(cpu, 1),
            'ram_pct': round(ram, 1),
        }
        if batt_pct is not None:
            ctx['battery_pct'] = batt_pct

        self._pub.publish(String(data=json.dumps(ctx, ensure_ascii=False)))


def main():
    rclpy.init()
    node = SituationalAwarenessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
