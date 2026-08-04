#!/usr/bin/env python3
"""Pointing gestures — "πήγαινε εκεί" with a finger instead of a room name.

Reads the skeletons pose_node already publishes, lifts the pointing arm into 3D
with the aligned depth image, and extends the shoulder->wrist line until it
meets the floor. That floor point is where the person is pointing.

Topics
  in   /pose_detections                                    (String, JSON)
       /camera/camera/aligned_depth_to_color/image_raw     (Image, 16UC1)
       /camera/camera/color/camera_info                    (CameraInfo)
  out  /gesture_point    (PointStamped, map)  — confirmed target, latched
       /gesture_status   (String, JSON)       — live state for the dashboard
       /gesture_markers  (MarkerArray)        — the ray + target ring in RViz

‼️ This node does NOT drive by default. `auto_goal` is false, so a confirmed
gesture is published and drawn, and something else decides to act on it — the
dashboard button or the voice tool `goto_pointed`. Pointing is far too easy to
do by accident (reaching for a cup crosses the same geometry) to wire straight
into the navigator. Set auto_goal:=true only for demos, with a clear floor.

The depth image is used unsynchronised: pose_node runs its own decimation, so
exact pairing would drop most frames. The latest depth frame is at most ~100 ms
older than the skeleton, and an arm does not move far in 100 ms — but this is
also why a gesture needs several agreeing frames before it counts.
"""

import json
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Empty, String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform

from home_robot.gesture import (
    GestureStabilizer, arm_straightness, depths_are_consistent,
    floor_intersection, is_pointing,
)
from home_robot.gesture_bindings import ACTION_EL, GestureBindings
from home_robot.gesture_vocab import GESTURE_EL, GestureClassifier

CAMERA_FRAME = 'camera_color_optical_frame'
_BINDINGS_PATH = os.path.expanduser('~/.ros/home_robot_gesture_bindings.json')

# Depth is noisy on a hand, so sample a small patch and take the median rather
# than the single pixel under the keypoint.
_PATCH = 5


class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')

        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('floor_z', 0.0)
        self.declare_parameter('confirm_hits', 5)
        self.declare_parameter('cluster_radius', 0.6)   # m
        self.declare_parameter('max_distance', 8.0)     # m from the hand
        self.declare_parameter('min_keypoint_conf', 0.5)
        self.declare_parameter('auto_goal', False)
        self.declare_parameter('tf_timeout', 0.2)
        self.declare_parameter('vocab_hold_frames', 6)
        self.declare_parameter('motion_gestures', False)

        self.target_frame = self.get_parameter('target_frame').value
        self.floor_z      = self.get_parameter('floor_z').value
        self.max_distance = self.get_parameter('max_distance').value
        self.min_kp_conf  = self.get_parameter('min_keypoint_conf').value
        self.auto_goal    = self.get_parameter('auto_goal').value
        self.tf_timeout   = self.get_parameter('tf_timeout').value

        self.bridge = CvBridge()
        self._depth = None
        self._intrinsics = None            # (fx, fy, cx, cy)
        self._last_point = None            # last confirmed (x, y) in map

        # Body-pose vocabulary on top of the pointing geometry. Same skeletons,
        # no extra camera work.
        self.bindings = self._load_bindings()
        self.vocab = GestureClassifier(
            hold_frames=self.get_parameter('vocab_hold_frames').value)
        self._last_gesture = None
        self._gesture_at = 0.0

        self._stab = GestureStabilizer(
            confirm_hits=self.get_parameter('confirm_hits').value,
            radius=self.get_parameter('cluster_radius').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(String, '/pose_detections', self._cb_poses, 5)
        self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw',
            self._cb_depth, 5)
        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._cb_info, 5)
        # How the dashboard button and the `goto_pointed` voice tool ask for the
        # last gesture to be acted on. Kept as an explicit trigger so the act of
        # driving is always someone's decision, never a side effect of pointing.
        self.create_subscription(Empty, '/gesture_go', self._cb_go, 5)

        # Latched: the dashboard and the voice tool both want the last gesture
        # after the arm has already dropped, not only while it is being held.
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._point_pub  = self.create_publisher(PointStamped, 'gesture_point', latched)
        self._status_pub = self.create_publisher(String, 'gesture_status', 10)
        self._marker_pub = self.create_publisher(MarkerArray, 'gesture_markers', 5)
        self._goal_pub   = self.create_publisher(PoseStamped, '/goal_pose', 10)
        # One topic per command family, matching what already exists on the
        # graph — no new command protocol invented for gestures.
        self._stop_pub   = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self._estop_pub  = self.create_publisher(Bool, '/emergency_stop', latched)
        self._dock_pub   = self.create_publisher(Bool, '/dock', 10)
        self._follow_pub = self.create_publisher(Bool, '/follow_command', 10)
        self._say_pub    = self.create_publisher(String, '/speech_response', 10)
        self._cancel_pub = self.create_publisher(String, '/mission/cancel', 10)
        self._patrol_pub = self.create_publisher(String, '/patrol_command', 10)
        self._event_pub  = self.create_publisher(String, 'gesture_event', 10)
        self.create_subscription(String, '/gesture_bindings/set',
                                 self._cb_set_bindings, 5)

        self.get_logger().info(
            f'Gesture node ready (auto_goal={self.auto_goal}) — point at the '
            f'floor and hold for {self._stab.confirm_hits} frames')

    # ── inputs ────────────────────────────────────────────────────────────────

    def _cb_info(self, msg: CameraInfo):
        self._intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _cb_depth(self, msg: Image):
        try:
            self._depth = self.bridge.imgmsg_to_cv2(
                msg, '16UC1').astype(np.float32) / 1000.0
        except Exception as exc:                       # malformed frame
            self.get_logger().warn(f'depth conversion failed: {exc}',
                                   throttle_duration_sec=10.0)

    def _cb_poses(self, msg: String):
        if self._depth is None or self._intrinsics is None:
            return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        # ‼️ pose_node publishes {"persons": [...], "width": W, "height": H},
        # not a bare list. Iterating the dict yielded its KEYS — plain strings —
        # and the node died on its first frame with 'str' has no attribute 'get'.
        persons = payload.get('persons', []) if isinstance(payload, dict) else payload
        if not isinstance(persons, list):
            return

        # The vocabulary reads the FIRST person only: a command has one author,
        # and two people gesturing at once must not race each other.
        if persons:
            first = {k['name']: k for k in persons[0].get('keypoints', [])}
            fired = self.vocab.update(first)
            if fired:
                self._on_gesture(fired)

        best = None            # (straightness, floor_point, debug)
        for person in persons:
            kps = {k['name']: k for k in person.get('keypoints', [])}
            for side in ('right', 'left'):
                found = self._arm_floor_point(kps, side)
                if found is None:
                    continue
                straight, point, debug = found
                if best is None or straight > best[0]:
                    best = (straight, point, debug)

        point = best[1] if best else None
        confirmed = self._stab.update(point)

        if confirmed is not None:
            self._on_confirmed(confirmed)

        self._publish_status(point, best[2] if best else None, confirmed)
        self._publish_markers(point, confirmed)

    # ── geometry ──────────────────────────────────────────────────────────────

    def _arm_floor_point(self, kps, side):
        """Floor point for one arm, or None if it is not pointing at the floor.

        Returns (straightness_deg, (x, y), debug_dict) so the caller can prefer
        the straighter of two arms when someone points with both.
        """
        try:
            shoulder = kps[f'{side}_shoulder']
            elbow    = kps[f'{side}_elbow']
            wrist    = kps[f'{side}_wrist']
        except KeyError:
            return None

        if min(shoulder['v'], elbow['v'], wrist['v']) < self.min_kp_conf:
            return None

        s2d = (shoulder['x'], shoulder['y'])
        e2d = (elbow['x'], elbow['y'])
        w2d = (wrist['x'], wrist['y'])
        if not is_pointing(s2d, e2d, w2d):
            return None

        s_depth = self._depth_at(*s2d)
        w_depth = self._depth_at(*w2d)
        if s_depth is None or w_depth is None:
            return None
        if not depths_are_consistent(s_depth, w_depth):
            return None

        s_cam = self._pixel_to_camera(s2d, s_depth)
        w_cam = self._pixel_to_camera(w2d, w_depth)

        s_map = self._to_map(s_cam)
        w_map = self._to_map(w_cam)
        if s_map is None or w_map is None:
            return None

        hit = floor_intersection(s_map, w_map, floor_z=self.floor_z,
                                 max_distance=self.max_distance)
        if hit is None:
            return None

        straight = arm_straightness(s2d, e2d, w2d)
        debug = {'side': side, 'straightness': round(straight, 1),
                 'wrist_depth': round(w_depth, 2)}
        return straight, hit, debug

    def _depth_at(self, u, v):
        """Median depth over a small patch, ignoring the zeros RealSense writes
        where it has no reading. None when the patch is entirely invalid."""
        h, w = self._depth.shape[:2]
        u0, u1 = max(0, u - _PATCH), min(w, u + _PATCH + 1)
        v0, v1 = max(0, v - _PATCH), min(h, v + _PATCH + 1)
        if u0 >= u1 or v0 >= v1:
            return None
        patch = self._depth[v0:v1, u0:u1]
        valid = patch[patch > 0.0]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _pixel_to_camera(self, pixel, depth_m):
        fx, fy, cx, cy = self._intrinsics
        u, v = pixel
        return ((u - cx) * depth_m / fx,
                (v - cy) * depth_m / fy,
                depth_m)

    def _to_map(self, xyz):
        pt = PointStamped()
        pt.header.frame_id = CAMERA_FRAME
        pt.header.stamp = rclpy.time.Time().to_msg()   # latest available
        pt.point.x, pt.point.y, pt.point.z = xyz
        try:
            out = self.tf_buffer.transform(
                pt, self.target_frame,
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout))
            return out.point.x, out.point.y, out.point.z
        except Exception:
            return None

    # ── outputs ───────────────────────────────────────────────────────────────

    def _on_confirmed(self, xy):
        x, y = xy
        self._last_point = (x, y)

        msg = PointStamped()
        msg.header.frame_id = self.target_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = x, y, self.floor_z
        self._point_pub.publish(msg)
        self.get_logger().info(f'Pointing confirmed → ({x:.2f}, {y:.2f})')

        if self.auto_goal:
            self.send_goal()

    # ── the vocabulary ────────────────────────────────────────────────────────

    def _load_bindings(self):
        try:
            with open(_BINDINGS_PATH, encoding='utf-8') as fh:
                b = GestureBindings.from_dict(json.load(fh))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            b = GestureBindings()
        # The launch flag is the floor: a stored mapping cannot switch motion
        # gestures on when the operator started the robot with them off.
        if not self.get_parameter('motion_gestures').value:
            b.motion_enabled = False
        return b

    def _save_bindings(self):
        try:
            os.makedirs(os.path.dirname(_BINDINGS_PATH), exist_ok=True)
            tmp = _BINDINGS_PATH + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.bindings.to_dict(), fh, ensure_ascii=False)
            os.replace(tmp, _BINDINGS_PATH)
        except OSError as exc:
            self.get_logger().warn(f'could not save bindings: {exc}')

    def _cb_set_bindings(self, msg: String):
        try:
            data = json.loads(msg.data) or {}
        except json.JSONDecodeError:
            return
        for gesture, action in (data.get('bindings') or {}).items():
            if not self.bindings.set(gesture, action):
                self.get_logger().warn(f'rejected binding {gesture} -> {action}')
        if 'motion_enabled' in data:
            if self.get_parameter('motion_gestures').value:
                self.bindings.motion_enabled = bool(data['motion_enabled'])
            else:
                self.get_logger().warn(
                    'motion gestures are disabled by launch (motion_gestures:=false)')
        self._save_bindings()
        self.get_logger().info('Gesture bindings updated')

    def _on_gesture(self, gesture):
        """A body pose was held long enough. Run whatever it is bound to."""
        import time
        self._last_gesture, self._gesture_at = gesture, time.time()
        action = self.bindings.action_for(gesture)
        greek = GESTURE_EL.get(gesture, gesture)
        self._event_pub.publish(String(data=json.dumps(
            {'gesture': gesture, 'greek': greek, 'action': action,
             'ts': self._gesture_at}, ensure_ascii=False)))
        if action is None:
            self.get_logger().info(f'Gesture «{greek}» — not bound')
            return
        self.get_logger().info(f'Gesture «{greek}» -> {action}')
        self._dispatch(action)

    def _dispatch(self, action):
        """Send the command. Stop-type actions go out immediately and
        repeatedly; a single Twist can be lost in a DDS hiccup."""
        if action == 'stop':
            for _ in range(3):
                self._stop_pub.publish(Twist())
        elif action == 'estop':
            self._estop_pub.publish(Bool(data=True))
        elif action == 'goto_pointed':
            if not self.send_goal():
                self._say_pub.publish(String(
                    data='Δεν έχω σημείο — δείξε πρώτα στο πάτωμα.'))
        elif action == 'follow':
            self._follow_pub.publish(Bool(data=True))
        elif action == 'stop_follow':
            self._follow_pub.publish(Bool(data=False))
        elif action == 'dock':
            self._dock_pub.publish(Bool(data=True))
        elif action == 'patrol':
            self._patrol_pub.publish(String(data='{}'))
        elif action == 'cancel':
            self._cancel_pub.publish(String(data='{}'))
        elif action == 'say_hello':
            self._say_pub.publish(String(data='Γεια σου!'))
        elif action == 'listen':
            self._say_pub.publish(String(data='Σε ακούω.'))

    def _cb_go(self, _msg: Empty):
        if not self.send_goal():
            self.get_logger().warn('No gesture to act on yet — point at the '
                                   'floor and hold until the ring turns green')

    def send_goal(self):
        """Navigate to the last confirmed gesture. Called by auto_goal, and by
        whatever else decides the gesture should be acted on."""
        if self._last_point is None:
            return False
        x, y = self._last_point
        goal = PoseStamped()
        goal.header.frame_id = self.target_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = x, y
        # No meaningful heading in a pointed floor spot — face the way the robot
        # will already be travelling rather than inventing a rotation on arrival.
        goal.pose.orientation.w = 1.0
        self._goal_pub.publish(goal)
        self.get_logger().info(f'Gesture goal sent → ({x:.2f}, {y:.2f})')
        return True

    def _publish_status(self, live_point, debug, confirmed):
        cand = self._stab.candidate
        status = {
            'pointing':  live_point is not None,
            'progress':  round(self._stab.progress, 2),
            'candidate': ({'x': round(cand.x, 2), 'y': round(cand.y, 2),
                           'hits': cand.hits} if cand else None),
            'confirmed': ({'x': round(confirmed[0], 2),
                           'y': round(confirmed[1], 2)} if confirmed else None),
            'last':      ({'x': round(self._last_point[0], 2),
                           'y': round(self._last_point[1], 2)}
                          if self._last_point else None),
            'auto_goal': self.auto_goal,
            'vocab': {
                'holding': self.vocab.holding,
                'holding_el': GESTURE_EL.get(self.vocab.holding),
                'progress': round(self.vocab.progress, 2),
                'last': self._last_gesture,
                'last_el': GESTURE_EL.get(self._last_gesture),
                'motion_enabled': self.bindings.motion_enabled,
                'bindings': self.bindings.as_dict(),
                'action_labels': ACTION_EL,
                # The whole table, so the dashboard can name every row and not
                # only the gesture currently being held.
                'gesture_labels': GESTURE_EL,
            },
        }
        if debug:
            status.update(debug)
        self._status_pub.publish(String(data=json.dumps(status)))

    def _publish_markers(self, live_point, confirmed):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        target = confirmed or live_point
        if target is None:
            # Clear the previous ring rather than leaving it hanging in RViz.
            m = Marker()
            m.header.frame_id = self.target_frame
            m.header.stamp = stamp
            m.ns, m.id = 'gesture', 0
            m.action = Marker.DELETEALL
            arr.markers.append(m)
            self._marker_pub.publish(arr)
            return

        ring = Marker()
        ring.header.frame_id = self.target_frame
        ring.header.stamp = stamp
        ring.ns, ring.id = 'gesture', 0
        ring.type = Marker.CYLINDER
        ring.action = Marker.ADD
        ring.pose.position.x, ring.pose.position.y = target
        ring.pose.position.z = self.floor_z + 0.01
        ring.pose.orientation.w = 1.0
        ring.scale.x = ring.scale.y = 0.5
        ring.scale.z = 0.02
        # Amber while gathering evidence, green once the gesture is confirmed.
        ring.color.r = 0.1 if confirmed else 1.0
        ring.color.g = 0.9 if confirmed else 0.7
        ring.color.b = 0.1
        ring.color.a = 0.85 if confirmed else 0.45
        arr.markers.append(ring)

        self._marker_pub.publish(arr)


def main():
    rclpy.init()
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
