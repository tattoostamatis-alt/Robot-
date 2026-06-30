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

import numpy as np
import psutil
import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

# kela.yaml map params — used to convert world (x,y) → mask pixel
_MAP_ORIGIN_X  = -3.854
_MAP_ORIGIN_Y  = -7.894
_MAP_RESOLUTION = 0.050


def _load_locations() -> dict:
    try:
        path = os.path.join(
            get_package_share_directory('home_robot'), 'config', 'locations.yaml'
        )
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_room_mask():
    """Load room_mask.png + room_colors.yaml. Returns (mask_arr, color_map) or (None, None)."""
    pkg = get_package_share_directory('home_robot')
    mask_path   = os.path.join(pkg, 'maps', 'room_mask.png')
    colors_path = os.path.join(pkg, 'maps', 'room_colors.yaml')
    try:
        from PIL import Image
        arr = np.array(Image.open(mask_path).convert('RGBA'))
        with open(colors_path) as f:
            color_map = yaml.safe_load(f)  # {room_name: [R, G, B]}
        return arr, color_map
    except Exception:
        return None, None


def _nearest_room(x: float, y: float, locations: dict,
                  mask_arr=None, color_map=None) -> str:
    # ── Pixel lookup in painted mask (preferred) ──────────────────────
    if mask_arr is not None and color_map:
        h, w = mask_arr.shape[:2]
        col = int((x - _MAP_ORIGIN_X) / _MAP_RESOLUTION)
        # mask saved in original PGM orientation: row 0 = top = max y
        row = int((y - _MAP_ORIGIN_Y) / _MAP_RESOLUTION)
        if 0 <= col < w and 0 <= row < h:
            r, g, b, a = mask_arr[row, col]
            if a > 50:
                # Find closest room color
                best_name, best_dist = 'άγνωστο', float('inf')
                for name, rgb in color_map.items():
                    d = (int(r)-rgb[0])**2 + (int(g)-rgb[1])**2 + (int(b)-rgb[2])**2
                    if d < best_dist:
                        best_dist, best_name = d, name
                return best_name

    # ── Fallback: nearest center (no mask or out of bounds) ───────────
    best_name, best_dist = 'άγνωστο', float('inf')
    for name, pose in locations.items():
        lx, ly = pose.get('x', 0.), pose.get('y', 0.)
        if lx == 0. and ly == 0.:
            continue
        d = math.sqrt((x - lx)**2 + (y - ly)**2)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


ROOM_NAMES_EL = {
    'saloni':              'σαλόνι',
    'kouzina':             'κουζίνα',
    'diadromos':           'διάδρομος',
    'toualeta':            'τουαλέτα',
    'domatio tou max':     'δωμάτιο του Μαξ',
    'domatio tou mbamba':  'δωμάτιο του μπαμπά',
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
        self._mask_arr, self._color_map = _load_room_mask()
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
        room_key = _nearest_room(self._odom_x, self._odom_y, self._locations,
                                  self._mask_arr, self._color_map)
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
