#!/usr/bin/env python3
"""Finger gestures — 21 hand landmarks, cropped from the wrist we already know.

The body skeleton stops at the wrist, so thumbs-up and a fist are invisible to
it. This node adds the hand landmark model on top.

‼️ It does NOT run palm detection. The usual MediaPipe pipeline is palm
detector -> landmark model, but pose_node already publishes both wrists for
every person, so the crop is free: take a square around the wrist, sized from
the shoulder width, and hand it straight to the landmark model. That skips an
entire network per frame and an SSD anchor decode, and it cannot lose a hand
that the skeleton can see.

‼️ MediaPipe the PYTHON PACKAGE must not be installed here. It pulls numpy>=2
and OpenCV 5; ROS's cv_bridge is built against numpy 1.26 and OpenCV 4 and dies
with `KeyError: 16` on every image, taking all of perception with it. This uses
the ONNX export of the same model through onnxruntime, which has neither
dependency. (Learned the hard way, 2026-08-04.)

Topics
  in   /pose_detections   (String, JSON)  — wrists, from pose_node
       /camera/camera/color/image_raw
  out  /hand_gesture      (String, JSON)  — what the hand is doing, for the UI
       plus whatever the binding fires, via the same topics gesture_node uses.
"""

import json
import os
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from home_robot.gesture_bindings import ACTION_EL, GestureBindings
from home_robot.hand_gestures import HAND_GESTURE_EL, HandGestureClassifier

_HF_REPO = 'opencv/handpose_estimation_mediapipe'
_HF_FILE = 'handpose_estimation_mediapipe_2023feb.onnx'
_BINDINGS_PATH = os.path.expanduser('~/.ros/home_robot_gesture_bindings.json')

_INPUT = 224
# The crop is this many shoulder-widths across, centred on the wrist. Big
# enough to hold an open hand at arm's length, small enough that the hand fills
# the frame — the landmark model expects a hand, not a hand in a room.
_CROP_SCALE = 1.1
_MIN_CROP_PX = 48


class HandGestureNode(Node):
    def __init__(self):
        super().__init__('hand_gesture_node')

        self.declare_parameter('hold_frames', 6)
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('process_every_n', 2)
        self.declare_parameter('num_threads', 2)
        self.declare_parameter('motion_gestures', False)

        self.min_conf = self.get_parameter('min_confidence').value
        self.every_n = self.get_parameter('process_every_n').value

        self.bridge = CvBridge()
        self._session = None
        self._input_name = None
        self._frame = None
        self._lock = threading.Lock()
        self._count = 0
        self._last = None
        self._last_at = 0.0
        self._landmarks = None

        self.bindings = self._load_bindings()
        self.classifier = HandGestureClassifier(
            hold_frames=self.get_parameter('hold_frames').value)

        threading.Thread(target=self._load_model, daemon=True).start()

        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._cb_image, 3)
        self.create_subscription(String, '/pose_detections', self._cb_poses, 5)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(String, 'hand_gesture', latched)
        self._stop_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self._estop_pub = self.create_publisher(Bool, '/emergency_stop', latched)
        self._dock_pub = self.create_publisher(Bool, '/dock', 10)
        self._follow_pub = self.create_publisher(Bool, '/follow_command', 10)
        self._say_pub = self.create_publisher(String, '/speech_response', 10)
        self._cancel_pub = self.create_publisher(String, '/mission/cancel', 10)
        self._patrol_pub = self.create_publisher(String, '/patrol_command', 10)
        self._go_pub = self.create_publisher(String, '/gesture_go_request', 10)

        self.get_logger().info('Hand gesture node starting — loading landmarks…')

    # ── model ─────────────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            self.get_logger().error(f'onnxruntime/huggingface_hub missing: {exc}')
            return
        try:
            path = hf_hub_download(_HF_REPO, _HF_FILE)
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().error(f'could not fetch hand landmarks ({exc})')
            return
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.get_parameter('num_threads').value
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(path, opts,
                                             providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name
        self.get_logger().info('Hand landmarks ready (21 points, CPU)')

    def _load_bindings(self):
        try:
            with open(_BINDINGS_PATH, encoding='utf-8') as fh:
                b = GestureBindings.from_dict(json.load(fh))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            b = GestureBindings()
        if not self.get_parameter('motion_gestures').value:
            b.motion_enabled = False
        return b

    # ── inputs ────────────────────────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return
        with self._lock:
            self._frame = frame

    def _cb_poses(self, msg: String):
        if self._session is None:
            return
        self._count += 1
        if self._count % self.every_n != 0:
            return
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        persons = payload.get('persons', []) if isinstance(payload, dict) else payload
        if not persons:
            self._update(None)
            return
        with self._lock:
            frame = None if self._frame is None else self._frame
        if frame is None:
            return

        kps = {k['name']: k for k in persons[0].get('keypoints', [])}
        landmarks = None
        for side in ('right', 'left'):
            crop, origin, size = self._crop_hand(frame, kps, side)
            if crop is None:
                continue
            landmarks = self._landmarks_from(crop, origin, size)
            if landmarks:
                break
        self._update(landmarks)

    def _crop_hand(self, frame, kps, side):
        """Square crop centred on a wrist, sized from the shoulder width."""
        wrist = kps.get(f'{side}_wrist')
        ls, rs = kps.get('left_shoulder'), kps.get('right_shoulder')
        if not wrist or wrist.get('v', 0) < self.min_conf or not (ls and rs):
            return None, None, None
        width = float(np.hypot(ls['x'] - rs['x'], ls['y'] - rs['y']))
        size = int(width * _CROP_SCALE)
        if size < _MIN_CROP_PX:
            return None, None, None
        cx, cy = int(wrist['x']), int(wrist['y'])
        h, w = frame.shape[:2]
        x0, y0 = max(0, cx - size // 2), max(0, cy - size // 2)
        x1, y1 = min(w, x0 + size), min(h, y0 + size)
        if x1 - x0 < _MIN_CROP_PX or y1 - y0 < _MIN_CROP_PX:
            return None, None, None
        return frame[y0:y1, x0:x1], (x0, y0), (x1 - x0, y1 - y0)

    def _landmarks_from(self, crop, origin, size):
        """Run the landmark model and map the 21 points back to image pixels."""
        blob = cv2.resize(crop, (_INPUT, _INPUT)).astype(np.float32) / 255.0
        blob = blob[np.newaxis]                      # NHWC, as the graph wants
        try:
            out = self._session.run(None, {self._input_name: blob})
        except Exception as exc:                     # noqa: BLE001
            self.get_logger().error(f'hand inference failed: {exc}',
                                    throttle_duration_sec=30.0)
            return None
        conf = float(np.asarray(out[1]).reshape(-1)[0])
        if conf < self.min_conf:
            return None
        pts = np.asarray(out[0]).reshape(-1, 3)
        if pts.shape[0] < 21:
            return None
        sx, sy = size[0] / _INPUT, size[1] / _INPUT
        return [[float(p[0] * sx + origin[0]),
                 float(p[1] * sy + origin[1]),
                 float(p[2])] for p in pts[:21]]

    # ── output ────────────────────────────────────────────────────────────────

    def _update(self, landmarks):
        self._landmarks = landmarks
        fired = self.classifier.update(landmarks)
        if fired:
            self._last, self._last_at = fired, time.time()
            action = self.bindings.action_for(fired)
            greek = HAND_GESTURE_EL.get(fired, fired)
            self.get_logger().info(
                f'Hand «{greek}» -> {action or "not bound"}')
            if action:
                self._dispatch(action)
        self._publish(fired)

    def _dispatch(self, action):
        if action == 'stop':
            for _ in range(3):
                self._stop_pub.publish(Twist())
        elif action == 'estop':
            self._estop_pub.publish(Bool(data=True))
        elif action == 'goto_pointed':
            # gesture_node owns the pointed target; ask it rather than
            # duplicating the geometry here.
            self._go_pub.publish(String(data='hand'))
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

    def _publish(self, fired):
        holding = self.classifier.holding
        self._pub.publish(String(data=json.dumps({
            'ready': self._session is not None,
            'holding': holding,
            'holding_el': HAND_GESTURE_EL.get(holding),
            'progress': round(self.classifier.progress, 2),
            'fired': fired,
            'last': self._last,
            'last_el': HAND_GESTURE_EL.get(self._last),
            'hand_visible': self._landmarks is not None,
            'motion_enabled': self.bindings.motion_enabled,
            'bindings': self.bindings.as_dict(),
            'action_labels': ACTION_EL,
            'gesture_labels': HAND_GESTURE_EL,
        }, ensure_ascii=False)))


def main():
    rclpy.init()
    node = HandGestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
