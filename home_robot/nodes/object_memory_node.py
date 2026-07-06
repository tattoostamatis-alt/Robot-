#!/usr/bin/env python3
"""Semantic object memory — remembers *where* things are in the map frame.

object_detector.py publishes per-frame `detected_objects` with metric positions
in the camera_color_optical_frame; those vanish the moment the object leaves
view. This node folds them into a persistent spatial map: each sighting is
transformed into the `map` frame (via TF) and merged into the nearest known
instance of the same label (see home_robot/object_memory.py for the pure merge
logic). The result survives restarts (JSON on disk) and answers "where did I
last see the X?".

Integration, in ascending order of coupling:
  - `object_memory` (String, JSON list) + `object_memory_markers` (MarkerArray)
    — always published, for dashboards / RViz.
  - `object_memory/query` (String label) → `object_memory/answer` (String JSON,
    the best matching instance or {}) — programmatic lookup.
  - memory/store (String) — when `rag_push` is on, concise natural-language
    facts ("cup last seen in kouzina") are pushed to rag_memory_node so the
    existing voice recall path ("Μαξ, πού είναι το X;") answers spatial
    questions with no new LLM tool (the TOOLS schema is prefill-capped — see
    llm_bridge_node.py — so reusing memory/store is deliberate).

Rooms are named by nearest entry in locations.yaml (no PIL/room-mask
dependency needed for a coarse "which room" label).
"""

import json
import math
import os
import threading

import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform

from home_robot.object_memory import ObjectMemory

CAMERA_FRAME = 'camera_color_optical_frame'


def _load_locations() -> dict:
    try:
        path = os.path.join(get_package_share_directory('home_robot'),
                            'config', 'locations.yaml')
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class ObjectMemoryNode(Node):
    def __init__(self):
        super().__init__('object_memory_node')

        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('merge_distance', 0.6)     # m — same-label sightings within this = one instance
        self.declare_parameter('ema_alpha', 0.35)         # position smoothing
        self.declare_parameter('min_conf', 0.5)
        self.declare_parameter('max_range', 5.0)          # ignore detections farther than this (m, camera z)
        self.declare_parameter('prune_age', 0.0)          # s unseen before forgetting; 0 = never
        self.declare_parameter('ignore_labels', ['person'])  # transient/moving — don't memorise
        self.declare_parameter('persist_path', os.path.expanduser('~/.robot_object_memory.json'))
        self.declare_parameter('publish_period', 2.0)     # s
        self.declare_parameter('save_period', 20.0)       # s
        self.declare_parameter('rag_push', True)
        self.declare_parameter('rag_push_period', 30.0)   # s between fact pushes
        self.declare_parameter('tf_timeout', 0.2)         # s

        self.target_frame = self.get_parameter('target_frame').value
        self.max_range    = self.get_parameter('max_range').value
        self.prune_age    = self.get_parameter('prune_age').value
        self.ignore       = set(self.get_parameter('ignore_labels').value)
        self.persist_path = self.get_parameter('persist_path').value
        self.rag_push     = self.get_parameter('rag_push').value
        self.tf_timeout   = self.get_parameter('tf_timeout').value

        self.mem = ObjectMemory(
            merge_distance=self.get_parameter('merge_distance').value,
            ema_alpha=self.get_parameter('ema_alpha').value,
            min_conf=self.get_parameter('min_conf').value,
            clock=lambda: self.get_clock().now().nanoseconds / 1e9,
        )
        self._lock = threading.Lock()
        self._locations = _load_locations()
        self._last_pushed: dict[int, str] = {}   # instance id → last fact pushed
        self._load()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.mem_pub    = self.create_publisher(String, 'object_memory', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'object_memory_markers', 10)
        self.answer_pub = self.create_publisher(String, 'object_memory/answer', 10)
        self.store_pub  = self.create_publisher(String, 'memory/store', 10)

        self.create_subscription(String, 'detected_objects', self._on_detected, 10)
        self.create_subscription(String, 'object_memory/query', self._on_query, 10)

        self.create_timer(self.get_parameter('publish_period').value, self._publish)
        self.create_timer(self.get_parameter('save_period').value, self._save)
        if self.rag_push:
            self.create_timer(self.get_parameter('rag_push_period').value, self._push_facts)

        self.get_logger().info(
            f'Object memory ready — {len(self.mem.all())} instances loaded, '
            f'frame={self.target_frame}, persist={self.persist_path}')

    # ── ingest ────────────────────────────────────────────────────────────
    def _on_detected(self, msg: String):
        try:
            objects = json.loads(msg.data) or []
        except json.JSONDecodeError:
            return

        with self._lock:
            for obj in objects:
                label = str(obj.get('label', '')).lower()
                if not label or label in self.ignore:
                    continue
                z = obj.get('z', 0.0)
                if z <= 0.1 or z > self.max_range:
                    continue
                pt = self._to_map(obj.get('x', 0.0), obj.get('y', 0.0), z)
                if pt is None:
                    continue
                mx, my, mz = pt
                self.mem.observe(label, mx, my, mz,
                                 conf=float(obj.get('conf', 1.0)),
                                 room=self._nearest_room(mx, my))
            if self.prune_age > 0:
                self.mem.prune(self.prune_age)

    def _to_map(self, x, y, z):
        pt = PointStamped()
        pt.header.frame_id = CAMERA_FRAME
        pt.header.stamp = rclpy.time.Time().to_msg()  # latest available
        pt.point.x, pt.point.y, pt.point.z = x, y, z
        try:
            out = self.tf_buffer.transform(
                pt, self.target_frame,
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout))
            return out.point.x, out.point.y, out.point.z
        except Exception:
            return None

    def _nearest_room(self, x, y):
        best, best_d = None, float('inf')
        for name, loc in self._locations.items():
            try:
                d = math.hypot(loc['x'] - x, loc['y'] - y)
            except (TypeError, KeyError):
                continue
            if d < best_d:
                best, best_d = name, d
        return best

    # ── query ─────────────────────────────────────────────────────────────
    def _on_query(self, msg: String):
        label = msg.data.strip().lower()
        with self._lock:
            inst = self.mem.query(label) if label else None
        self.answer_pub.publish(String(
            data=json.dumps(inst or {}, ensure_ascii=False)))

    # ── outputs ───────────────────────────────────────────────────────────
    def _publish(self):
        with self._lock:
            items = self.mem.all()
        self.mem_pub.publish(String(data=json.dumps(items, ensure_ascii=False)))

        markers = MarkerArray()
        for i, inst in enumerate(items):
            m = Marker()
            m.header.frame_id = self.target_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = 'object_memory', i
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(inst['x'])
            m.pose.position.y = float(inst['y'])
            m.pose.position.z = float(inst['z'])
            m.pose.orientation.w = 1.0
            m.scale.z = 0.12
            m.color.a = 1.0
            m.color.r = m.color.g = m.color.b = 1.0
            room = inst.get('room')
            m.text = f"{inst['label']}" + (f" ({room})" if room else "")
            markers.markers.append(m)
        self.marker_pub.publish(markers)

    def _push_facts(self):
        """Push a concise NL fact per instance to rag_memory, but only when it
        changed (new room / new object) so we don't spam the vector store."""
        with self._lock:
            items = self.mem.all()
        for inst in items:
            room = inst.get('room')
            if not room:
                continue
            fact = f"{inst['label']} last seen in {room}"
            if self._last_pushed.get(inst['id']) == fact:
                continue
            self._last_pushed[inst['id']] = fact
            self.store_pub.publish(String(data=fact))

    # ── persistence ───────────────────────────────────────────────────────
    def _load(self):
        try:
            with open(self.persist_path) as f:
                self.mem.load_list(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        except Exception as e:
            self.get_logger().warn(f'Could not load object memory: {e}')

    def _save(self):
        with self._lock:
            data = self.mem.to_list()
        try:
            tmp = self.persist_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.persist_path)   # atomic
        except Exception as e:
            self.get_logger().warn(f'Could not save object memory: {e}')

    def destroy_node(self):
        self._save()
        super().destroy_node()


def main():
    rclpy.init()
    node = ObjectMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
