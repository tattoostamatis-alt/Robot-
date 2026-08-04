#!/usr/bin/env python3
"""People — one identity per person, from face, voice and body.

Face recognition and speaker diarization each already exist and each already
knows a set of names. Neither knows about the other, so the robot could tell
you who was talking OR who was visible, never who was in the room. This node
joins them, adds a body measurement, and gives the dashboard one place to
enrol somebody across all three.

Topics
  in   /face_identities   (String, JSON)  — who is visible
       /current_speaker   (String)        — who is talking
       /pose_detections   (String, JSON)  — skeletons, for the body measurement
       /camera/camera/aligned_depth_to_color/image_raw
       /people/add        (String)        — a name to create
       /people/remove     (String)
       /people/enrol      (String, JSON)  — {"name":..., "what":"face|voice"}
  out  /people            (String, JSON)  — the roster and who is here now
       /faces/enrol       (String)        — relayed to face_identity_node
       /diarization/register (String)     — relayed to diarization_node

‼️ Height is measured against the MAP frame, where z = 0 is the floor. The head
keypoint's height above the floor is therefore just its map-frame z — no need
to see the feet, which are behind furniture or out of frame most of the time.
That does mean the measurement needs localization; without a map->camera
transform there is no floor to measure from and the node reports no height
rather than a wrong one.
"""

import json
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform

from home_robot.people import HEAD_TO_CROWN, PersonRegistry

_STATE_PATH = os.path.expanduser('~/.ros/home_robot_people.json')
CAMERA_FRAME = 'camera_color_optical_frame'

# Head keypoints in preference order: the nose is the best anchor, the eyes and
# ears are the fallbacks when a head is turned away.
_HEAD_KPS = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear')
_PATCH = 5
_SPEAKER_TTL = 8.0          # s a voice match still counts as "now"


class PeopleNode(Node):
    def __init__(self):
        super().__init__('people_node')

        self.declare_parameter('min_keypoint_conf', 0.5)
        self.declare_parameter('measure_every_n', 3)
        self.declare_parameter('tf_timeout', 0.2)

        self.min_conf = self.get_parameter('min_keypoint_conf').value
        self.every_n = self.get_parameter('measure_every_n').value
        self.tf_timeout = self.get_parameter('tf_timeout').value

        self.bridge = CvBridge()
        self.registry = self._load()
        self._depth = None
        self._intrinsics = None
        self._faces = []
        self._speaker = None
        self._speaker_at = 0.0
        self._count = 0
        self._last_height = None
        self._here = None
        self._reason = ''
        self._confidence = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(String, '/face_identities', self._cb_faces, 5)
        self.create_subscription(String, '/current_speaker', self._cb_speaker, 10)
        self.create_subscription(String, '/pose_detections', self._cb_poses, 5)
        self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw',
            self._cb_depth, 3)
        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._cb_info, 5)
        self.create_subscription(String, '/people/add', self._cb_add, 5)
        self.create_subscription(String, '/people/remove', self._cb_remove, 5)
        self.create_subscription(String, '/people/enrol', self._cb_enrol, 5)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(String, 'people', latched)
        self._face_enrol_pub = self.create_publisher(String, '/faces/enrol', 10)
        self._voice_enrol_pub = self.create_publisher(
            String, '/diarization/register', 10)

        self.create_timer(60.0, self._save)
        self.create_timer(2.0, self._publish)
        self.get_logger().info(
            f'People ready — {len(self.registry)} known '
            f'({", ".join(self.registry.names()) or "nobody"})')

    # ── state ─────────────────────────────────────────────────────────────────

    def _load(self):
        try:
            with open(_STATE_PATH, encoding='utf-8') as fh:
                return PersonRegistry.from_dict(json.load(fh))
        except FileNotFoundError:
            return PersonRegistry()
        except (json.JSONDecodeError, OSError) as exc:
            self.get_logger().warn(f'people file unreadable ({exc}) — empty')
            return PersonRegistry()

    def _save(self):
        try:
            os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
            tmp = _STATE_PATH + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.registry.to_dict(), fh, ensure_ascii=False)
            os.replace(tmp, _STATE_PATH)
        except OSError as exc:
            self.get_logger().warn(f'could not save people: {exc}',
                                   throttle_duration_sec=300.0)

    # ── enrolment ─────────────────────────────────────────────────────────────

    def _cb_add(self, msg: String):
        name = (msg.data or '').strip()
        if self.registry.add(name):
            self.get_logger().info(f'Added "{name}"')
            self._save()
            self._publish()

    def _cb_remove(self, msg: String):
        name = (msg.data or '').strip()
        if self.registry.remove(name):
            self.get_logger().info(f'Removed "{name}"')
            self._save()
            self._publish()

    def _cb_enrol(self, msg: String):
        """Start enrolling one modality. The specialist node does the work; we
        only remember that the person now has it."""
        try:
            req = json.loads(msg.data) or {}
        except json.JSONDecodeError:
            return
        name = str(req.get('name', '')).strip()
        what = str(req.get('what', '')).strip()
        if not name:
            return
        self.registry.add(name)
        if what == 'face':
            self._face_enrol_pub.publish(String(data=name))
            self.registry.mark_face(name)
            self.get_logger().info(f'Enrolling face for "{name}"')
        elif what == 'voice':
            self._voice_enrol_pub.publish(String(data=name))
            self.registry.mark_voice(name)
            self.get_logger().info(f'Enrolling voice for "{name}" — speak now')
        self._save()
        self._publish()

    # ── inputs ────────────────────────────────────────────────────────────────

    def _cb_info(self, msg: CameraInfo):
        self._intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _cb_depth(self, msg: Image):
        try:
            self._depth = self.bridge.imgmsg_to_cv2(
                msg, '16UC1').astype(np.float32) / 1000.0
        except Exception:
            pass

    def _cb_faces(self, msg: String):
        try:
            data = json.loads(msg.data) or {}
        except json.JSONDecodeError:
            return
        self._faces = [f.get('name') for f in (data.get('faces') or [])
                       if f.get('name') and f['name'] != 'unknown']
        # Whatever face_identity has enrolled is, by definition, known here.
        for name in data.get('enrolled') or []:
            self.registry.mark_face(name)

    def _cb_speaker(self, msg: String):
        name = (msg.data or '').strip()
        if name and name.lower() != 'unknown':
            self._speaker = name
            self._speaker_at = time.time()
            self.registry.mark_voice(name)

    def _cb_poses(self, msg: String):
        self._count += 1
        if self._count % self.every_n != 0:
            return
        if self._depth is None or self._intrinsics is None:
            return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        persons = payload.get('persons', []) if isinstance(payload, dict) else payload
        if not persons:
            self._last_height = None
            return
        kps = {k['name']: k for k in persons[0].get('keypoints', [])}
        self._last_height = self._measure_height(kps)

        # Fuse whatever is available right now.
        voice = (self._speaker
                 if time.time() - self._speaker_at < _SPEAKER_TTL else None)
        face = self._faces[0] if self._faces else None
        name, conf, why = self.registry.identify(face, voice, self._last_height)
        self._here, self._confidence, self._reason = name, conf, why
        if name:
            self.registry.observe(name, height=self._last_height,
                                  now=time.time())

    def _measure_height(self, kps):
        """Height above the floor of the top of the head, in metres.

        None whenever anything is missing — a wrong height is worse than no
        height, because it feeds a veto that can hide a correct match.
        """
        head = None
        for name in _HEAD_KPS:
            kp = kps.get(name)
            if kp and kp.get('v', 0.0) >= self.min_conf:
                head = kp
                break
        if head is None:
            return None
        depth = self._depth_at(int(head['x']), int(head['y']))
        if depth is None:
            return None
        fx, fy, cx, cy = self._intrinsics
        pt = PointStamped()
        pt.header.frame_id = CAMERA_FRAME
        pt.header.stamp = rclpy.time.Time().to_msg()
        pt.point.x = (head['x'] - cx) * depth / fx
        pt.point.y = (head['y'] - cy) * depth / fy
        pt.point.z = depth
        try:
            out = self.tf_buffer.transform(
                pt, 'map',
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout))
        except Exception:
            return None                      # not localized: no floor to use
        return float(out.point.z) + HEAD_TO_CROWN

    def _depth_at(self, u, v):
        h, w = self._depth.shape[:2]
        u0, u1 = max(0, u - _PATCH), min(w, u + _PATCH + 1)
        v0, v1 = max(0, v - _PATCH), min(h, v + _PATCH + 1)
        if u0 >= u1 or v0 >= v1:
            return None
        patch = self._depth[v0:v1, u0:u1]
        valid = patch[patch > 0.0]
        if valid.size == 0:
            return None
        d = float(np.median(valid))
        return d if 0.3 < d <= 6.0 else None

    # ── output ────────────────────────────────────────────────────────────────

    def _publish(self):
        voice = (self._speaker
                 if time.time() - self._speaker_at < _SPEAKER_TTL else None)
        self._pub.publish(String(data=json.dumps({
            'people': self.registry.summary(),
            'here': self._here,
            'confidence': round(self._confidence, 2),
            'reason': self._reason,
            'seeing': self._faces,
            'hearing': voice,
            'measured_height': (round(self._last_height, 2)
                                if self._last_height else None),
        }, ensure_ascii=False)))


def main():
    rclpy.init()
    node = PeopleNode()
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
