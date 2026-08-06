#!/usr/bin/env python3
"""Measure the D435's height and downward tilt by fitting a plane to the floor.

Why this exists: tf_base_camera carried pitch=0 from the day it was written
until 2026-08-06, while the camera is actually tilted ~30 deg down on its
bracket. Nothing complained. A wrong pitch does not break anything loudly — it
just puts every camera detection at the wrong HEIGHT, by more the further away
the object is. On the floor, at arm's length, that is about 0.45 m of error,
which is how «πιάσε την παντόφλα» ended with the gripper closing in mid-air
above the slipper.

A protractor against the bracket is not good enough here: 5 deg is ~7 cm of
height error at 0.8 m. The depth camera can measure its own mounting far more
accurately than you can, so let it.

Usage — point the robot at a clear patch of FLOOR (no furniture, no feet, no
rug edges in the middle of the frame) and run:

    python3 scripts/measure_camera_pitch.py

Then put the printed values into tf_base_camera in launch/bringup.launch.py and
rebuild. Re-run after any change to the camera bracket.
"""

import math
import sys
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'

# Sample a central block only: the frame edges catch walls, furniture legs and
# the robot's own arm, and a handful of those points drags the fit badly.
U_RANGE = (200, 460, 12)
V_RANGE = (150, 470, 12)
Z_RANGE = (0.3, 2.5)          # m — outside this the D435 is noise


def collect(node, bridge, timeout=15.0):
    got = {}
    node.create_subscription(
        Image, DEPTH_TOPIC,
        lambda m: got.setdefault('depth', bridge.imgmsg_to_cv2(m, '16UC1')), 10)
    node.create_subscription(
        CameraInfo, INFO_TOPIC, lambda m: got.setdefault('k', m.k), 10)
    end = time.time() + timeout
    while len(got) < 2 and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.5)
    return got.get('depth'), got.get('k')


def floor_points(depth, k):
    fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    depth = depth.astype(np.float32) / 1000.0
    pts = []
    for v in range(*V_RANGE):
        for u in range(*U_RANGE):
            z = depth[v, u]
            if Z_RANGE[0] < z < Z_RANGE[1]:
                pts.append(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return np.array(pts)


def fit(points):
    """Height and downward pitch of the camera, in the optical frame.

    Optical convention: x right, y DOWN, z forward. So the floor is a plane
    y = a*x + c*z + d, its normal is (a, -1, c) and the tilt of the optical
    axis out of that plane is what we want.
    """
    A = np.c_[points[:, 0], points[:, 2], np.ones(len(points))]
    (a, c, d), *_ = np.linalg.lstsq(A, points[:, 1], rcond=None)
    normal = np.array([a, -1.0, c])
    normal /= np.linalg.norm(normal)
    return abs(d), math.degrees(math.asin(abs(normal[2])))


def main():
    rclpy.init()
    node = Node('measure_camera_pitch')
    depth, k = collect(node, CvBridge())
    if depth is None or k is None:
        print(f'No depth on {DEPTH_TOPIC} — is the camera up '
              f'(use_perception:=true) and publishing aligned depth?')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    points = floor_points(depth, k)
    if len(points) < 100:
        print(f'Only {len(points)} usable depth points — aim at a clear patch '
              f'of floor and try again.')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    height, pitch_deg = fit(points)
    residual = None
    print(f'{len(points)} floor points')
    print(f'  height above floor : {height:.3f} m')
    print(f'  pitch (downwards)  : {pitch_deg:.1f} deg  ({math.radians(pitch_deg):.3f} rad)')
    print()
    print('Put these into tf_base_camera in launch/bringup.launch.py:')
    print(f"    '--z', '{height:.3f}', '--pitch', '{math.radians(pitch_deg):.3f}',")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
