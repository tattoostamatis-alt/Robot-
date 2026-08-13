#!/usr/bin/env python3
"""Sweep the arm's wrist/roll at a fixed hover point to find where the
gripper AprilTag ('gripper_tag') stays visible to the camera.

2026-08-10's hand-eye test found the tag went OUT of frame the instant the
arm moved to a hover target with t=0,r=0 (neutral wrist/roll) — never got to
test whether the closed-loop correction converges. This probe holds x/y/z
fixed (a pose already known to be a normal hover — see pick_place_node's
approach_height/max_reach) and only varies wrist(t)/roll(r), the two knobs
that change which way the tag's printed face points relative to the camera.

‼️ arm/raw_cmd (T:104) is a RAW passthrough to the firmware — no collision
or joint-limit clamping happens in software (see arm_driver.py module
docstring). Watch the arm live. Ctrl-C sends the arm back to init (T:100)
before exiting.

Usage:
  python3 hand_eye_probe.py [x] [y] [z]
  defaults: x=0.30 y=0.0 z=0.05 (arm_base frame; z=0.05 puts the wrist
  ~0.22 m off the floor, inside the FOV/reach overlap band measured in
  project_robot_camera_fov_vs_arm_reach, not the floor's dead zone)
"""
import json
import sys
import time

import rclpy
import tf2_ros
from rclpy.node import Node
from std_msgs.msg import String

ARM_FRAME = 'arm_base'
GRIPPER_TAG_FRAME = 'gripper_tag'
MOVE_SETTLE = 2.0     # s — matches pick_place_node's movement_settle_time default
POLL_WINDOW = 1.5     # s — how long to watch for a fresh tag sighting per pose
STALE_AFTER = 0.3     # s — older than this = cached/stale TF, not a live sighting

# Conservative sweep — MECH_LIMITS allow wrist -1.57..1.57, roll -3.14..3.14,
# but wide angles here point the tag away from the robot or into the chassis.
WRIST_VALUES = [-0.6, -0.3, 0.0, 0.3, 0.6]
ROLL_VALUES = [-0.8, -0.4, 0.0, 0.4, 0.8]


class HandEyeProbe(Node):
    def __init__(self, x, y, z):
        super().__init__('hand_eye_probe')
        self.x, self.y, self.z = x, y, z
        self.raw_pub = self.create_publisher(String, 'arm/raw_cmd', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _raw(self, cmd):
        self.raw_pub.publish(String(data=json.dumps(cmd)))

    def _cartesian(self, t, r):
        self._raw({'T': 104, 'x': self.x, 'y': self.y, 'z': self.z,
                    't': t, 'r': r, 'spd': 0})

    def _tag_fresh(self):
        """(seen, age_s) — age is None if no transform is cached at all."""
        try:
            tf = self.tf_buffer.lookup_transform(
                ARM_FRAME, GRIPPER_TAG_FRAME, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
        except tf2_ros.TransformException:
            return False, None
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        age = now - stamp
        return age < STALE_AFTER, age

    def probe_pose(self, t, r):
        self._cartesian(t, r)
        deadline = time.monotonic() + MOVE_SETTLE
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        deadline = time.monotonic() + POLL_WINDOW
        best_age = None
        seen = False
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            ok, age = self._tag_fresh()
            if ok:
                seen = True
                best_age = age if best_age is None else min(best_age, age)
        return seen, best_age

    def go_init(self):
        self._raw({'T': 100})
        deadline = time.monotonic() + MOVE_SETTLE
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
    y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    z = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    rclpy.init()
    node = HandEyeProbe(x, y, z)
    print(f'Hover σταθερό στο arm_base ({x:.2f}, {y:.2f}, {z:.2f}). '
          f'Σάρωση wrist×roll — {len(WRIST_VALUES)}×{len(ROLL_VALUES)} θέσεις.')
    print('ΠΡΟΣΟΧΗ: raw_cmd χωρίς safety clamp — παρακολούθα το ρομπότ ζωντανά.')
    print('Ξεκινάει σε 3s... (Ctrl-C για διακοπή + επιστροφή σε init)\n')
    time.sleep(3.0)

    results = []
    try:
        for t in WRIST_VALUES:
            for r in ROLL_VALUES:
                seen, age = node.probe_pose(t, r)
                mark = f'ΟΡΑΤΟ (age={age*1000:.0f}ms)' if seen else 'εκτός πεδίου'
                print(f'  wrist={t:+.1f} roll={r:+.1f}  ->  {mark}')
                results.append((t, r, seen))
    except KeyboardInterrupt:
        print('\nΔιακόπηκε — επιστροφή σε init...')
    finally:
        node.go_init()
        node.destroy_node()
        rclpy.shutdown()

    hits = [(t, r) for t, r, seen in results if seen]
    print(f'\n{len(hits)}/{len(results)} θέσεις με ορατό tag.')
    if hits:
        print('Ορατές (wrist, roll):', ', '.join(f'({t:+.1f},{r:+.1f})' for t, r in hits))
    else:
        print('Καμία θέση δεν κράτησε το tag ορατό σε αυτό το hover (x,y,z) — '
              'δοκίμασε διαφορετικό x/y/z (π.χ. πιο κοντά στην κάμερα ή πιο χαμηλό z).')


if __name__ == '__main__':
    main()
