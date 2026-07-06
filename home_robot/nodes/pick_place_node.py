#!/usr/bin/env python3
"""Pick-and-place — bridges object_detector.py detections to arm_driver.py.

NOT YET HW-VERIFIED end-to-end. The RoArm-M3 is now connected (use_arm:=true),
but every motion still goes through arm_driver.py's documented unknowns (T:105
feedback field names, T:210 torque semantics) plus this node's own: the
base_link->arm_base TF (bringup.launch.py's tf_base_arm) is a guessed
placeholder, not a measurement. Re-verify slowly/by hand and calibrate
tf_base_arm before trusting a full grasp.

Flow, triggered by `pick_command` (std_msgs/String, JSON {"label": "cup"}
or {} for "whatever clutter object_detector.py last saw"):
  1. Look up the target in the latest `detected_objects` message
     (object_detector.py — x/y/z are metric, in the
     camera_color_optical_frame convention: x right, y down, z forward).
  2. Transform that point into `arm_base` via tf2 (the TF chain is
     object_detector's camera_color_optical_frame -> camera_link, published
     by the realsense driver itself, -> base_link -> arm_base, the latter
     two both static_transform_publishers from bringup.launch.py).
  3. Open gripper, hover above the target, then **visually servo** the XY
     alignment (see below) before descending, grasping, lifting, moving to
     the drop-off pose, releasing, and returning to init (T:100).

Visual servoing (closed-loop XY correction, param `servo_enabled`):
The D435 is body-mounted, not on the arm (eye-to-hand), so the object stays
in view while the arm moves. Rather than committing to one noisy snapshot,
the node re-reads detected_objects while hovering, re-locks the *same*
object (nearest same-label detection to the current estimate, so it can't
jump to a different cup), and nudges the hover XY toward it until the
estimate settles within `servo_tolerance` or `servo_max_iters` runs out.
This refines out per-frame detection jitter; it does NOT correct a
systematic base_link->arm_base calibration bias (the camera can't see the
gripper) — calibrate tf_base_arm for that. If detections stop mid-servo the
loop keeps the last good estimate and proceeds (graceful open-loop fallback).

arm_driver.py exposes no "motion complete" feedback (T:105 reports
joint/EE pose, not a busy flag), so each step waits a fixed settle time
instead of polling for arrival — generous by design until real timing is
observed.
"""

import json
import math
import threading
import time

import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Float32, String
from tf2_geometry_msgs import do_transform_point


CAMERA_FRAME = 'camera_color_optical_frame'
ARM_FRAME = 'arm_base'


class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')

        self.declare_parameter('approach_height', 0.10)   # m above target before descending
        self.declare_parameter('grasp_z_offset', 0.0)      # m added to target z at grasp (fingertip vs. detected centroid)
        self.declare_parameter('drop_x', 0.15)             # arm_base frame — placeholder, no tray/bin measured yet
        self.declare_parameter('drop_y', -0.15)
        self.declare_parameter('drop_z', 0.10)
        self.declare_parameter('gripper_open', 1.2)        # rad — JOINT_LIMITS['hand'] is 1.08..3.14, lower = more open
        self.declare_parameter('gripper_closed', 3.0)
        self.declare_parameter('arm_speed', 0)             # passed through to T:104 "spd", 0 = firmware default
        self.declare_parameter('movement_settle_time', 2.0)  # s — no completion feedback from arm_driver.py, see module docstring
        self.declare_parameter('tf_timeout', 2.0)
        # Visual servoing (closed-loop XY refinement while hovering)
        self.declare_parameter('servo_enabled', True)
        self.declare_parameter('servo_tolerance', 0.015)   # m — XY estimate settled → descend
        self.declare_parameter('servo_max_iters', 4)       # correction moves before giving up
        self.declare_parameter('servo_settle_time', 0.8)   # s — shorter than a full move; just a nudge
        self.declare_parameter('servo_relock_radius', 0.15)  # m — max jump to still count as the same object

        self.approach_height = self.get_parameter('approach_height').value
        self.grasp_z_offset = self.get_parameter('grasp_z_offset').value
        self.drop_pose = (self.get_parameter('drop_x').value,
                           self.get_parameter('drop_y').value,
                           self.get_parameter('drop_z').value)
        self.gripper_open = self.get_parameter('gripper_open').value
        self.gripper_closed = self.get_parameter('gripper_closed').value
        self.arm_speed = self.get_parameter('arm_speed').value
        self.movement_settle_time = self.get_parameter('movement_settle_time').value
        self.tf_timeout = self.get_parameter('tf_timeout').value
        self.servo_enabled = self.get_parameter('servo_enabled').value
        self.servo_tolerance = self.get_parameter('servo_tolerance').value
        self.servo_max_iters = self.get_parameter('servo_max_iters').value
        self.servo_settle_time = self.get_parameter('servo_settle_time').value
        self.servo_relock_radius = self.get_parameter('servo_relock_radius').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.raw_cmd_pub = self.create_publisher(String, 'arm/raw_cmd', 10)
        self.gripper_pub = self.create_publisher(Float32, 'arm/gripper_cmd', 10)
        self.response_pub = self.create_publisher(String, 'speech_response', 10)
        self.result_pub = self.create_publisher(String, 'pick_result', 10)

        self._latest_objects = None
        self._busy = threading.Lock()

        self.create_subscription(String, 'detected_objects', self._on_detected_objects, 10)
        self.create_subscription(String, 'pick_command', self._on_pick_command, 10)

        self.get_logger().info('Pick-place node started (UNTESTED — no arm hardware yet)')

    def _on_detected_objects(self, msg: String):
        try:
            self._latest_objects = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_pick_command(self, msg: String):
        try:
            data = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            data = {}

        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already executing a pick, ignoring new request')
            self._publish_result('error', 'busy with another pick')
            return
        threading.Thread(target=self._wrapped, args=(data.get('label'),), daemon=True).start()

    def _wrapped(self, label):
        try:
            self._run_pick(label)
        finally:
            self._busy.release()

    def _select_target(self, label):
        if not self._latest_objects:
            return None
        candidates = [o for o in self._latest_objects if o.get('clutter')]
        if label:
            candidates = [o for o in candidates if o.get('label') == label] or candidates
        return candidates[0] if candidates else None

    def _run_pick(self, label):
        target = self._select_target(label)
        if target is None:
            self._say('Δεν βλέπω κάτι να σηκώσω αυτή τη στιγμή.')
            self._publish_result('error', 'no matching object in detected_objects')
            return

        arm_point = self._transform_to_arm_frame(target['x'], target['y'], target['z'])
        if arm_point is None:
            self._publish_result('error', f'TF lookup {CAMERA_FRAME} -> {ARM_FRAME} failed (is tf_base_arm running?)')
            return

        ax, ay, az = arm_point
        az += self.grasp_z_offset
        hover_z = az + self.approach_height

        self._say(f'Πάω να σηκώσω: {target["label"]}.')
        self.get_logger().info(
            f'Pick target "{target["label"]}" at arm_base ({ax:.3f}, {ay:.3f}, {az:.3f})')

        self._gripper(self.gripper_open)
        self._cartesian(ax, ay, hover_z)

        # Closed-loop XY refinement while hovering (see module docstring).
        if self.servo_enabled:
            ax, ay, az = self._servo(target['label'], ax, ay, az, hover_z)

        self._cartesian(ax, ay, az)
        self._gripper(self.gripper_closed)
        self._cartesian(ax, ay, hover_z)

        dx, dy, dz = self.drop_pose
        self._cartesian(dx, dy, dz)
        self._gripper(self.gripper_open)
        self._raw({'T': 100})  # back to init pose
        time.sleep(self.movement_settle_time)

        self._say(f'Τακτοποίησα: {target["label"]}.')
        self._publish_result('ok', target['label'])

    def _servo(self, label, ax, ay, az, hover_z):
        """Refine (ax, ay, az) toward the live detection while hovering. Returns
        the settled arm-frame target. Keeps the current estimate if detections
        or TF drop out mid-loop (graceful open-loop fallback)."""
        for i in range(self.servo_max_iters):
            locked = self._relock(label, ax, ay)
            if locked is None:
                self.get_logger().info('Servo: lost sight of target, using last estimate')
                break
            nx, ny, nz = locked
            dxy = math.hypot(nx - ax, ny - ay)
            ax, ay, az = nx, ny, nz + self.grasp_z_offset
            if dxy <= self.servo_tolerance:
                self.get_logger().info(f'Servo converged (Δ={dxy*1000:.0f}mm) after {i} nudges')
                break
            self.get_logger().info(f'Servo nudge {i+1}: Δ={dxy*1000:.0f}mm → ({ax:.3f}, {ay:.3f})')
            self._cartesian(ax, ay, hover_z, settle=self.servo_settle_time)
        return ax, ay, az

    def _relock(self, label, ref_ax, ref_ay):
        """Among the latest detections of `label`, return the arm-frame (x,y,z)
        of the one nearest the current estimate (so we track the *same* object,
        not jump to another). None if none within servo_relock_radius."""
        best, best_d = None, self.servo_relock_radius
        for o in (self._latest_objects or []):
            if o.get('label') != label:
                continue
            p = self._transform_to_arm_frame(o['x'], o['y'], o['z'])
            if p is None:
                continue
            d = math.hypot(p[0] - ref_ax, p[1] - ref_ay)
            if d <= best_d:
                best, best_d = p, d
        return best

    def _transform_to_arm_frame(self, x, y, z):
        point = PointStamped()
        point.header.frame_id = CAMERA_FRAME
        point.header.stamp = rclpy.time.Time().to_msg()  # latest available transform
        point.point.x, point.point.y, point.point.z = x, y, z
        try:
            transform = self.tf_buffer.lookup_transform(
                ARM_FRAME, CAMERA_FRAME, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout))
        except tf2_ros.TransformException as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return None
        out = do_transform_point(point, transform)
        return out.point.x, out.point.y, out.point.z

    def _cartesian(self, x, y, z, settle=None):
        self._raw({'T': 104, 'x': x, 'y': y, 'z': z, 't': 0, 'r': 0, 'spd': self.arm_speed})
        time.sleep(self.movement_settle_time if settle is None else settle)

    def _gripper(self, pos):
        self.gripper_pub.publish(Float32(data=pos))
        time.sleep(self.movement_settle_time)

    def _raw(self, cmd: dict):
        self.raw_cmd_pub.publish(String(data=json.dumps(cmd)))

    def _say(self, text):
        self.get_logger().info(f'Pick-place: {text}')
        self.response_pub.publish(String(data=text))

    def _publish_result(self, status, detail):
        self.result_pub.publish(String(data=json.dumps({'status': status, 'detail': detail})))


def main():
    rclpy.init()
    node = PickPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
