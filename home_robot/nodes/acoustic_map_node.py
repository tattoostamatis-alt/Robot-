#!/usr/bin/env python3
"""Acoustic map — the robot remembers WHERE sounds come from.

sound_event_node says what it heard and doa_node says which way it came from,
but a direction is only useful for the half second it is spoken. This node
pins each bearing to the robot's place on the map and keeps them, so over days
the doorbell, the tap and the television acquire positions.

Topics
  in   /sound_events   (String, JSON)  — what fired, and the DoA angle with it
       TF map->base_link                — where the robot was when it heard it
  out  /acoustic_map   (String, JSON)  — hotspots per sound, for the dashboard
       /acoustic_map/markers (MarkerArray) — the same, in RViz

Persists to ~/.ros/home_robot_acoustic.json. Belongs to the MAP: a remap
invalidates every coordinate in it, the same way it invalidates taught rooms.

‼️ A single bearing is a ray, not a place. The robot spends most of its life
parked, and from one spot every ray is equally consistent with a source
anywhere along it — so hotspots stay empty until a sound has been heard from at
least two positions. What IS available from one bearing is the ROOM the ray
passes into, which is why that is computed separately and reported immediately.
"""

import json
import math
import os

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

from home_robot.acoustic_map import SoundSourceMap, bearing_to_world, ray_points
from home_robot.sound_events import EVENTS
from home_robot.status_query import NON_ROOMS, ROOM_LOCATIVE_EL

_STATE_PATH = os.path.expanduser('~/.ros/home_robot_acoustic.json')
_LOCATIONS = os.path.expanduser(
    '~/robot_ws/src/home_robot/config/locations.yaml')


class AcousticMapNode(Node):
    def __init__(self):
        super().__init__('acoustic_map_node')

        self.declare_parameter('cell', 0.25)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('state_path', _STATE_PATH)
        self.declare_parameter('locations_file', _LOCATIONS)

        self.state_path = self.get_parameter('state_path').value
        self.map = self._load()
        self.map.cell = self.get_parameter('cell').value
        self.map.max_range = self.get_parameter('max_range').value

        self.locations = self._load_locations()
        self._last = None            # the most recent attribution

        # ‼️ TF, not /amcl_pose. AMCL only publishes a pose while the robot is
        # MOVING, and a robot that hears the doorbell is almost always parked —
        # subscribing to /amcl_pose meant the pose was stale or absent exactly
        # when a sound arrived. map->base_link is always there once localized.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(String, '/sound_events', self._cb_sound, 10)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(String, 'acoustic_map', latched)
        self._marker_pub = self.create_publisher(
            MarkerArray, 'acoustic_map/markers', latched)

        self.create_timer(120.0, self._save)
        self.create_timer(5.0, self._publish)
        self.get_logger().info(
            f'Acoustic map ready — {len(self.map.kinds())} sounds located')

    # ── state ─────────────────────────────────────────────────────────────────

    def _load(self):
        try:
            with open(self.state_path, encoding='utf-8') as fh:
                return SoundSourceMap.from_dict(json.load(fh))
        except FileNotFoundError:
            return SoundSourceMap()
        except (json.JSONDecodeError, OSError) as exc:
            self.get_logger().warn(f'acoustic map unreadable ({exc}) — empty')
            return SoundSourceMap()

    def _save(self):
        self.map.prune()
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.map.to_dict(), fh)
            os.replace(tmp, self.state_path)
        except OSError as exc:
            self.get_logger().warn(f'could not save acoustic map: {exc}',
                                   throttle_duration_sec=300.0)

    def _load_locations(self):
        try:
            with open(self.get_parameter('locations_file').value,
                      encoding='utf-8') as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError):
            return {}
        return {k: v for k, v in data.items()
                if k not in NON_ROOMS and isinstance(v, dict)}

    # ── inputs ────────────────────────────────────────────────────────────────

    def _robot_pose(self):
        """(x, y, yaw) in the map, or None when not localized."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, 2.0 * math.atan2(q.z, q.w)

    def _cb_sound(self, msg: String):
        try:
            data = json.loads(msg.data) or {}
        except json.JSONDecodeError:
            return
        fired = data.get('fired') or []
        angle = data.get('angle')
        if not fired or angle is None:
            return
        pose = self._robot_pose()
        if pose is None:
            self.get_logger().warn('sound heard but not localized — cannot place it',
                                   throttle_duration_sec=60.0)
            return

        x, y, yaw = pose
        for kind in fired:
            self.map.observe(kind, x, y, yaw, angle)
            room = self._room_along_ray(x, y, yaw, angle)
            self._last = {
                'event': kind,
                'greek': EVENTS.get(kind, (kind,))[0],
                'room': room,
                'room_el': ROOM_LOCATIVE_EL.get(room, room) if room else None,
                'angle': angle,
                'from': [round(x, 2), round(y, 2)],
            }
            self.get_logger().info(
                f'{kind} from {ROOM_LOCATIVE_EL.get(room, room or "?")}')
        self._publish()

    def _room_along_ray(self, x, y, yaw, angle):
        """Which taught room the bearing points into.

        Available from ONE observation, unlike a hotspot, which is why the
        robot can say "νερό στην κουζίνα" the very first time it hears it.
        Approximate by design: the nearest taught room centre to any point on
        the ray, which is right for a flat with well-separated rooms and wrong
        for two rooms sharing a long wall. The hotspots are the precise answer,
        once there is enough evidence for one.
        """
        if not self.locations:
            return None
        world = bearing_to_world(yaw, angle)
        best, best_d = None, float('inf')
        for px, py in ray_points(x, y, world, self.map.max_range, 0.25):
            for name, loc in self.locations.items():
                try:
                    d = math.hypot(loc['x'] - px, loc['y'] - py)
                except (KeyError, TypeError):
                    continue
                if d < best_d:
                    best, best_d = name, d
        # Beyond this the "nearest room" is just the closest label on the map,
        # not a room the ray actually entered.
        return best if best_d < 2.0 else None

    # ── output ────────────────────────────────────────────────────────────────

    def _publish(self):
        sounds = []
        for kind in self.map.kinds():
            spots = self.map.hotspots(kind, top=2)
            sounds.append({
                'event': kind,
                'greek': EVENTS.get(kind, (kind,))[0],
                'heard': self.map.observations(kind),
                'spots': [{'x': round(sx, 2), 'y': round(sy, 2), 'w': w}
                          for sx, sy, w in spots],
                'room': self._nearest_room(spots[0][0], spots[0][1])
                        if spots else None,
            })
        sounds.sort(key=lambda s: -s['heard'])
        self._pub.publish(String(data=json.dumps({
            'sounds': sounds,
            'last': self._last,
            'localized': sum(1 for s in sounds if s['spots']),
        }, ensure_ascii=False)))
        self._publish_markers(sounds)

    def _nearest_room(self, x, y):
        best, best_d = None, float('inf')
        for name, loc in self.locations.items():
            try:
                d = math.hypot(loc['x'] - x, loc['y'] - y)
            except (KeyError, TypeError):
                continue
            if d < best_d:
                best, best_d = name, d
        return best if best_d < 3.0 else None

    def _publish_markers(self, sounds):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.header.stamp = stamp
        clear.ns, clear.id, clear.action = 'acoustic', 0, Marker.DELETEALL
        arr.markers.append(clear)
        i = 1
        for s in sounds:
            for spot in s['spots']:
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = stamp
                m.ns, m.id = 'acoustic', i
                m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
                m.pose.position.x = float(spot['x'])
                m.pose.position.y = float(spot['y'])
                m.pose.position.z = 0.6
                m.pose.orientation.w = 1.0
                m.scale.z = 0.16
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.75, 0.2, 0.95
                m.text = f"🔊 {s['greek']}"
                arr.markers.append(m)
                i += 1
        self._marker_pub.publish(arr)


def main():
    rclpy.init()
    node = AcousticMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
