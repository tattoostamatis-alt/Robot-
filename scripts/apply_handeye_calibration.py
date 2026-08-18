#!/usr/bin/env python3
"""Folds an easy_handeye2 result into home_robot's base_link->arm_base static TF.

Run AFTER a calibration has been saved via handeye_calibrate.launch.py (rqt
calibrator -> Compute -> Save). Reads world->camera_color_optical_frame from
~/.ros2/easy_handeye2/calibrations/<name>.calib, looks up the currently-live
base_link->camera_color_optical_frame (from the running bringup stack), and
combines them to print the replacement static_transform_publisher arguments
for bringup.launch.py's "Static TF: base_link -> arm_base" block — 'world' IS
arm_base's intended origin (see arm_scene_obstacles.py), this just replaces
the hand-measured guess with the calibrated one.

Requires bringup (camera) running so base_link->camera_color_optical_frame is
live in tf. Does NOT edit bringup.launch.py itself -- paste the printed
--x/--y/--z/--qx/--qy/--qz/--qw values into the static_transform_publisher
arguments there by hand, so the change is reviewed before it affects a real
pick.
"""
import sys

import rclpy
import tf_transformations
import yaml
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

CALIB_PATH_TMPL = '~/.ros2/easy_handeye2/calibrations/{name}.calib'


def transform_to_matrix(t):
    trans = [t.translation.x, t.translation.y, t.translation.z]
    rot = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
    m = tf_transformations.quaternion_matrix(rot)
    m[0:3, 3] = trans
    return m


def matrix_to_static_tf_args(m):
    trans = m[0:3, 3]
    quat = tf_transformations.quaternion_from_matrix(m)
    return (
        f"'--x', '{trans[0]:.4f}', '--y', '{trans[1]:.4f}', '--z', '{trans[2]:.4f}', "
        f"'--qx', '{quat[0]:.4f}', '--qy', '{quat[1]:.4f}', '--qz', '{quat[2]:.4f}', "
        f"'--qw', '{quat[3]:.4f}',"
    )


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'd435_arm'
    calib_path = __import__('os').path.expanduser(CALIB_PATH_TMPL.format(name=name))

    with open(calib_path) as f:
        calib = yaml.safe_load(f)
    t = calib['transform']
    world_to_camera = transform_to_matrix(
        type('T', (), {
            'translation': type('V', (), t['translation']),
            'rotation': type('V', (), t['rotation']),
        })()
    )

    rclpy.init()
    node = Node('apply_handeye_calibration')
    buf = Buffer()
    listener = TransformListener(buf, node)
    node.get_logger().info('Waiting for live base_link -> camera_color_optical_frame ...')
    tf_msg = None
    while rclpy.ok() and tf_msg is None:
        rclpy.spin_once(node, timeout_sec=0.5)
        if buf.can_transform('base_link', 'camera_color_optical_frame', rclpy.time.Time()):
            tf_msg = buf.lookup_transform('base_link', 'camera_color_optical_frame', rclpy.time.Time())

    base_to_camera = transform_to_matrix(tf_msg.transform)
    # base_link -> world = base_link -> camera  *  inverse(world -> camera)
    base_to_world = base_to_camera @ tf_transformations.inverse_matrix(world_to_camera)

    print('\nReplace the static_transform_publisher arguments for')
    print("'Static TF: base_link -> arm_base' in bringup.launch.py with:\n")
    print(matrix_to_static_tf_args(base_to_world))
    print("'--frame-id', 'base_link', '--child-frame-id', 'arm_base'],")

    rclpy.shutdown()


if __name__ == '__main__':
    main()
