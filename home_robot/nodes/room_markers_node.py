#!/usr/bin/env python3
"""Publishes room markers for RViz and tracks which room the robot is in.

Topics published:
  /room_markers   (MarkerArray)  — coloured spheres + labels in RViz
  /current_room   (String)       — nearest room, latched; from map->base_link
                                   (polled) or /amcl_pose, whichever arrives
"""

import math
import os

import yaml
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


COLORS = [
    (1.0, 0.27, 0.27),   # red    – saloni
    (0.27, 0.53, 1.0),   # blue   – kouzina
    (0.27, 0.80, 0.40),  # green  – diadromos
    (1.0, 0.80, 0.0),    # yellow – toualeta
    (0.80, 0.27, 1.0),   # purple – domatio max
    (1.0, 0.53, 0.0),    # orange – domatio mbamba
]


class RoomMarkersNode(Node):
    def __init__(self):
        super().__init__('room_markers_node')

        loc_path = os.path.join(
            get_package_share_directory('home_robot'), 'config', 'locations.yaml')
        with open(loc_path) as f:
            self._locs = yaml.safe_load(f) or {}

        # Latched marker publisher so RViz gets them even if it starts late
        latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._marker_pub = self.create_publisher(MarkerArray, 'room_markers', latch)
        # Latched for the same reason, and it matters more here. This is
        # published per /amcl_pose, and AMCL publishes only when the robot
        # moves past its update thresholds — measured 2026-07-31, a stationary
        # robot emits no /amcl_pose at all. So a subscriber that starts while
        # the robot is parked receives nothing until someone drives it. Live
        # that made "σε ποιο δωμάτιο είσαι;" answer "ο εντοπισμός δεν είναι
        # ενεργός" while AMCL was perfectly localized — a false alarm about
        # the one subsystem the user would go and debug. Subscribers must
        # request TRANSIENT_LOCAL too, or they will not get the retained sample.
        self._room_pub   = self.create_publisher(String, '/current_room', latch)

        # TRANSIENT_LOCAL, matching AMCL's own publisher QoS (verified with
        # `ros2 topic info /amcl_pose --verbose`). AMCL publishes a pose only
        # when the robot moves past its update thresholds, so on a parked robot
        # a volatile subscription here receives nothing — this node then never
        # publishes a room, and every downstream answer degrades to "δεν ξέρω
        # πού βρίσκομαι". With TRANSIENT_LOCAL we get AMCL's retained last pose
        # immediately on startup and can answer straight away.
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._on_pose, latch)

        # /amcl_pose is not enough on its own. AMCL emits it only when the robot
        # moves past its update thresholds, so a robot that has been parked
        # since boot never produces one — verified 2026-07-31, the topic had no
        # sample at all while map->base_link was perfectly valid. Localization
        # was working; only the *topic* was silent. TF is the honest source:
        # poll it, and fall back to the subscription for the in-between case
        # where a pose arrives before our first tick.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._current_room: str = ''

        self._publish_markers()
        self.create_timer(5.0, self._publish_markers)
        self.create_timer(2.0, self._poll_tf_pose)

        self.get_logger().info(
            f'Room markers ready — {len(self._locs)} rooms: '
            + ', '.join(self._locs.keys()))

    # ── room detection ─────────────────────────────────────────────

    def _poll_tf_pose(self):
        """Derive the room from map->base_link when AMCL's topic stays quiet."""
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return   # not localized yet — the empty room string is correct
        self._update_room(tf.transform.translation.x,
                          tf.transform.translation.y)

    def _on_pose(self, msg: PoseWithCovarianceStamped):
        self._update_room(msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _update_room(self, rx: float, ry: float):
        best_name = ''
        best_dist = float('inf')
        for name, pose in self._locs.items():
            d = math.hypot(rx - float(pose['x']), ry - float(pose['y']))
            if d < best_dist:
                best_dist = d
                best_name = name

        if best_name != self._current_room:
            self._current_room = best_name
            self.get_logger().info(f'Room: {best_name}  (dist {best_dist:.2f}m)')

        self._room_pub.publish(String(data=best_name))

    # ── RViz markers ───────────────────────────────────────────────

    def _publish_markers(self):
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()
        idx = 0

        for i, (name, pose) in enumerate(self._locs.items()):
            r, g, b = COLORS[i % len(COLORS)]
            x, y = float(pose['x']), float(pose['y'])

            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp    = now
            sphere.ns     = 'rooms'
            sphere.id     = idx; idx += 1
            sphere.type   = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x  = x
            sphere.pose.position.y  = y
            sphere.pose.position.z  = 0.05
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.30
            sphere.color.r = r; sphere.color.g = g; sphere.color.b = b
            sphere.color.a = 0.85
            sphere.lifetime.sec = 6
            markers.markers.append(sphere)

            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp    = now
            label.ns     = 'room_labels'
            label.id     = idx; idx += 1
            label.type   = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x  = x
            label.pose.position.y  = y
            label.pose.position.z  = 0.55
            label.pose.orientation.w = 1.0
            label.scale.z  = 0.28
            label.color.r  = r; label.color.g = g; label.color.b = b
            label.color.a  = 1.0
            label.text     = name
            label.lifetime.sec = 6
            markers.markers.append(label)

        self._marker_pub.publish(markers)


def main():
    rclpy.init()
    node = RoomMarkersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
