#!/usr/bin/env python3
"""
situational_awareness_node.py — Aggregates sensor state into a compact JSON
context that llm_bridge_node prepends to every LLM turn automatically.

Sources:
  - TF map→base_link (fallback /odom) → nearest named room (locations.yaml / room mask)
  - /map        → origin/resolution for the world→mask-pixel conversion
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
import time

import numpy as np
import psutil
import yaml
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
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
                  mask_arr=None, color_map=None, map_info=None) -> str:
    # ── Pixel lookup in painted mask (preferred) ──────────────────────
    if mask_arr is not None and color_map and map_info is not None:
        origin_x, origin_y, resolution = map_info
        h, w = mask_arr.shape[:2]
        col = int((x - origin_x) / resolution)
        # mask saved in image orientation: row 0 = top = max y
        row = h - 1 - int((y - origin_y) / resolution)
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


# Single source of truth, shared with format_location() so the room never gets
# spoken as greeklish in one answer and Greek in another.
from home_robot.status_query import ROOM_NAMES_EL  # noqa: E402


class SituationalAwarenessNode(Node):
    def __init__(self):
        super().__init__('situational_awareness_node')

        self.declare_parameter('update_hz',      1.0)
        self.declare_parameter('max_obj_range',  3.5)   # meters — ignore farther objects
        self.declare_parameter('max_obj_count',  5)     # cap objects listed in context
        # ‼️ OFF, and it must stay off while this robot has no battery pack.
        # The pack was removed 2026-07-26 (it runs off a power bank) and the OI
        # still publishes a figure that is nonsense — 41% at 14.44 V, when
        # 14.44 V is a full pack. status_query.format_status already refuses to
        # quote it, but this context line fed the same junk straight into every
        # LLM prompt, so any battery question phrased outside that module's
        # keyword gate got the fabricated number back with full confidence —
        # the exact failure status_query exists to prevent, entering by the
        # back door. Set true only if a real pack is refitted.
        self.declare_parameter('report_battery', False)

        self._report_battery = self.get_parameter('report_battery').value
        self._max_range  = self.get_parameter('max_obj_range').value
        self._max_obj    = self.get_parameter('max_obj_count').value
        hz               = self.get_parameter('update_hz').value

        self._locations  = _load_locations()
        self._mask_arr, self._color_map = _load_room_mask()
        self._map_info   = None   # (origin_x, origin_y, resolution) of the live map
        self._odom_x     = 0.
        self._odom_y     = 0.
        self._objects    = []
        self._tracked_rx = 0.0    # last tracked_objects arrival (see _on_detected)
        self._battery    = None   # BatteryState

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        map_qos = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map',            self._on_map,     map_qos)
        self.create_subscription(Odometry,     '/odom',           self._on_odom,    10)
        self.create_subscription(BatteryState, 'battery/state',   self._on_battery, 10)
        # Prefer tracked_objects (stable IDs) but fall back to detected_objects
        self.create_subscription(String, 'tracked_objects',  self._on_tracked, 10)
        self.create_subscription(String, 'detected_objects', self._on_detected, 10)

        self._pub = self.create_publisher(String, 'situation_context', 10)
        self.create_timer(1.0 / hz, self._publish)

        self.get_logger().info('Situational awareness node ready')

    # ── Subscriptions ────────────────────────────────────────────────

    def _on_map(self, msg: OccupancyGrid):
        info = msg.info
        self._map_info = (info.origin.position.x, info.origin.position.y,
                          info.resolution)
        if self._mask_arr is not None:
            h, w = self._mask_arr.shape[:2]
            if (w, h) != (info.width, info.height):
                self.get_logger().warning(
                    f'room_mask.png is {w}x{h} but the active map is '
                    f'{info.width}x{info.height} — the mask was painted on an '
                    'older map. Falling back to nearest-location lookup; '
                    'redraw maps/room_mask.png over the current map.')
                self._mask_arr = None

    def _on_odom(self, msg: Odometry):
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y

    def _map_xy(self):
        """Robot (x, y) in the map frame; falls back to raw odom if TF is down."""
        try:
            t = self._tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return self._odom_x, self._odom_y

    # ‼️ tracked_objects is the PREFERRED source; detected_objects is a fallback
    # for when the tracker is not running. Both used to be wired to the same
    # callback with no preference at all despite the comment claiming one, so
    # with perception up every detection cycle was processed TWICE — once
    # tracked, once raw — alternating between the two views of the same scene.
    _TRACKED_TTL = 2.0        # s: consider the tracker live for this long

    def _on_tracked(self, msg):
        self._tracked_rx = time.monotonic()
        self._on_objects(msg)

    def _on_detected(self, msg):
        if time.monotonic() - self._tracked_rx < self._TRACKED_TTL:
            return            # the tracker is live; ignore the raw duplicate
        self._on_objects(msg)

    def _on_battery(self, msg: BatteryState):
        self._battery = msg

    def _on_objects(self, msg: String):
        try:
            self._objects = json.loads(msg.data) or []
        except json.JSONDecodeError:
            pass

    # ── Context assembly ─────────────────────────────────────────────

    def _publish(self):
        # Room — mask and locations.yaml are in the map frame, not odom
        x, y = self._map_xy()
        room_key = _nearest_room(x, y, self._locations,
                                  self._mask_arr, self._color_map, self._map_info)
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

        # Battery — suppressed entirely unless report_battery says a real pack
        # is fitted. The NaN/range filter below is not enough on its own:
        # garbage from the OI lands inside [0, 1] just as easily as outside it.
        batt_pct = None
        if self._report_battery and self._battery is not None:
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
