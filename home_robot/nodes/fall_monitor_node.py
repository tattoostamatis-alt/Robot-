#!/usr/bin/env python3
"""Fall monitoring — watches pose_detections for someone on the floor.

Consumes what pose_node.py already publishes, so nothing extra runs on the
GPU: this node is a few comparisons per frame. All the judgement lives in
home_robot/fall_detect.py as pure functions, tested without a robot.

Subscribes: pose_detections (String, JSON from pose_node)
Publishes:  fall_alert   (Bool, LATCHED) — someone is down right now
            fall_event   (String, JSON)  — one message per alert, with detail
            speech_response (String)     — spoken warning, once per alert

‼️ Needs `use_pose:=true`. Without pose_node there are no skeletons and this
node sits silent — which is indistinguishable from "nobody has fallen", so it
says so on startup and warns if no poses ever arrive.

This deliberately does NOT try to detect the fall *event* (the fast downward
motion): that needs frame-to-frame identity at a stable frame rate, and the
detector runs at a few Hz with no tracking. "Someone is on the floor and has
not got up" is both more reliable and the thing that matters when the house is
otherwise empty.
"""
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, String

from home_robot.fall_detect import FallMonitor, looks_fallen, torso_angle_deg


class FallMonitorNode(Node):

    def __init__(self):
        super().__init__('fall_monitor_node')

        # Seconds a person must stay down before this is called a fall. Long
        # enough that sitting on the floor or reaching under the sofa never
        # trips it; short enough to matter. See fall_detect.py.
        self.declare_parameter('hold_s', 6.0)
        self.declare_parameter('clear_s', 10.0)
        self.declare_parameter('floor_band', 0.55)
        self.declare_parameter('lying_angle', 55.0)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('speak', True)
        self.declare_parameter('speech', 'Είδα κάποιον στο πάτωμα. Είσαι καλά;')

        self.hold_s = self.get_parameter('hold_s').value
        self.floor_band = self.get_parameter('floor_band').value
        self.lying_angle = self.get_parameter('lying_angle').value
        self.frame_height = self.get_parameter('frame_height').value
        self.speak = self.get_parameter('speak').value
        self.speech = self.get_parameter('speech').value

        self._monitor = FallMonitor(
            hold_s=self.hold_s,
            clear_s=self.get_parameter('clear_s').value)

        # Latched: this is state ("someone is down"), and a dashboard that
        # connects after the alert must learn about it immediately rather than
        # showing all-clear until the next message. Same reasoning as
        # /voice_activity and /emergency_stop.
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self._alert_pub = self.create_publisher(Bool, 'fall_alert', latched)
        self._event_pub = self.create_publisher(String, 'fall_event', 10)
        self._speech_pub = self.create_publisher(String, 'speech_response', 10)

        self.create_subscription(String, 'pose_detections', self._on_poses, 10)

        self._last_pose_at = 0.0
        self._warned_no_poses = False
        self._alert_pub.publish(Bool(data=False))   # seed the latched topic
        self.create_timer(10.0, self._health)
        self.get_logger().info(
            f'Fall monitor ready — alerts after {self.hold_s:.0f}s on the floor '
            '(needs use_pose:=true)')

    def _health(self):
        """A silent detector and a safe house look identical. Say which."""
        if self._last_pose_at:
            return
        if not self._warned_no_poses:
            self._warned_no_poses = True
            self.get_logger().warn(
                'No pose_detections yet — fall monitoring is NOT active. '
                'Start the stack with use_pose:=true (and a camera).')

    def _on_poses(self, msg: String):
        self._last_pose_at = time.monotonic()
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        # pose_node publishes {'persons': [...], 'width', 'height'}; older
        # builds published a bare list. Accept both so a rolling restart (or a
        # stale installed copy) does not crash this node.
        if isinstance(data, list):
            persons, frame_h = data, self.frame_height
        else:
            persons = data.get('persons') or []
            frame_h = data.get('height') or self.frame_height

        worst = None
        for p in persons:
            if looks_fallen(p, frame_h, floor_band=self.floor_band,
                            lying_angle=self.lying_angle):
                worst = p
                break

        fired = self._monitor.update(worst is not None)
        # Publish the level on every transition so the dashboard tracks the
        # recovery too, not just the alert.
        if fired or (worst is None and not self._monitor.alerted):
            self._alert_pub.publish(Bool(data=self._monitor.alerted))

        if not fired:
            return

        angle = torso_angle_deg(worst)
        detail = {
            'at': time.time(),
            'held_s': round(self.hold_s, 1),
            'torso_angle': round(angle, 1) if angle is not None else None,
            'box': [worst.get('x1'), worst.get('y1'),
                    worst.get('x2'), worst.get('y2')],
        }
        self.get_logger().warn(f'FALL detected: {detail}')
        self._event_pub.publish(String(data=json.dumps(detail)))
        if self.speak:
            self._speech_pub.publish(String(data=self.speech))


def main():
    rclpy.init()
    node = FallMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
