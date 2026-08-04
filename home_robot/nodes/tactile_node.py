#!/usr/bin/env python3
"""Touch — the arm feels what it is doing, instead of grabbing blind.

The RoArm-M3 reports a load per joint next to every angle, and nothing read it.
This node watches that stream and turns it into three things a person can use:
did it touch something, was it hard or soft, and was it heavy.

Topics
  in   /arm/raw_feedback  (String, JSON)  — the arm's own telemetry
  out  /arm/touch         (String, JSON)  — contact, hardness, weight
       /speech_response   (String)        — spoken, on contact, if enabled

‼️ PASSIVE. This node never commands the arm. It measures whatever the arm is
already doing, whether that is the joystick, MoveIt or pick_place — so it is
safe to run always, and it cannot make the arm push on anything.

‼️ Everything is in raw servo units, not newtons or grams. Nothing here has
been calibrated against a known mass, so the numbers are comparable to each
other and to nothing else. The labels say "σκληρό" and "βαρύ", never "180 g" —
that would be a fabrication dressed as a measurement.

How the three measurements are taken:

  * **Contact** — a step above the load the arm carries while moving freely.
    The baseline follows the arm's own pose, because holding it out
    horizontally loads the shoulder far more than holding it down.
  * **Hardness** — load against travel while in contact, using the
    end-effector position the arm already reports. A table spikes over a
    millimetre; a cushion barely moves.
  * **Weight** — the extra load on shoulder and elbow when the gripper is
    closed versus the last time it was open at rest.
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

from home_robot.tactile import (
    ContactDetector, describe_hardness, describe_touch, describe_weight,
    hardness, total_load, weight_estimate,
)

# The gripper angle below which we treat the hand as closed on something.
_GRIP_CLOSED = 0.5
# Movement below this counts as "still", for a weight reading.
_STILL_MM = 1.5
_STILL_S = 0.6


class TactileNode(Node):
    def __init__(self):
        super().__init__('tactile_node')

        self.declare_parameter('threshold', 45.0)
        self.declare_parameter('confirm', 3)
        self.declare_parameter('speak', False)

        self.speak = self.get_parameter('speak').value
        self.contact = ContactDetector(
            threshold=self.get_parameter('threshold').value,
            confirm=self.get_parameter('confirm').value)

        self._probe = []             # [(travel_mm, load)] while in contact
        self._origin = None          # end-effector position at first contact
        self._last_pos = None
        self._still_since = None
        self._empty_ref = None       # feedback with the hand open and still
        self._hardness = None
        self._weight = None
        self._touched_at = 0.0

        self.create_subscription(String, '/arm/raw_feedback', self._cb, 20)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(String, 'arm/touch', latched)
        self._say_pub = self.create_publisher(String, '/speech_response', 10)

        self.create_timer(0.5, self._publish)
        self.get_logger().info('Tactile ready — passive, never commands the arm')

    def _cb(self, msg: String):
        try:
            fb = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(fb, dict):
            return

        pos = self._position(fb)
        moved = (math.dist(pos, self._last_pos)
                 if (pos and self._last_pos) else 0.0)
        now = time.time()
        if pos:
            if moved > _STILL_MM:
                self._still_since = None
            elif self._still_since is None:
                self._still_since = now
            self._last_pos = pos

        started = self.contact.update(fb)

        if started:
            self._origin = pos
            self._probe = []
            self._hardness = None
            self._touched_at = now
            self.get_logger().info(f'Contact (load +{self.contact.excess})')

        if self.contact.in_contact and pos and self._origin:
            travel = math.dist(pos, self._origin)
            self._probe.append((travel, total_load(fb)))
            del self._probe[:-60]
            self._hardness = hardness(self._probe)
        elif not self.contact.in_contact:
            # Hand open and holding still: this is the reference the next
            # weight reading is measured against.
            if self._grip(fb) > _GRIP_CLOSED and self._is_still(now):
                self._empty_ref = dict(fb)

        if (self._grip(fb) <= _GRIP_CLOSED and self._empty_ref
                and self._is_still(now)):
            self._weight = weight_estimate(fb, self._empty_ref)

        if started and self.speak:
            self._say_pub.publish(String(data=describe_touch(
                slope=self._hardness, weight=self._weight)))

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _position(fb):
        try:
            return (float(fb['x']), float(fb['y']), float(fb['z']))
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _grip(fb):
        try:
            return abs(float(fb.get('g', 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _is_still(self, now):
        return (self._still_since is not None
                and now - self._still_since >= _STILL_S)

    def _publish(self):
        self._pub.publish(String(data=json.dumps({
            'contact': self.contact.in_contact,
            'excess': self.contact.excess,
            'baseline': (round(self.contact.baseline, 1)
                         if self.contact.baseline is not None else None),
            'hardness': (round(self._hardness, 1)
                         if self._hardness is not None else None),
            'hardness_el': describe_hardness(self._hardness),
            'weight': self._weight,
            'weight_el': describe_weight(self._weight),
            'samples': len(self._probe),
            'have_reference': self._empty_ref is not None,
            'since': round(time.time() - self._touched_at, 1)
                     if self._touched_at else None,
        }, ensure_ascii=False)))


def main():
    rclpy.init()
    node = TactileNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
