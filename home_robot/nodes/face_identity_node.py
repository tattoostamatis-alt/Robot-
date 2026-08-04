#!/usr/bin/env python3
"""Who is in the room — SFace identities on top of YuNet's faces.

face_detection_node already finds faces and publishes five landmarks each.
This node aligns those crops and embeds them with SFace, so the robot knows
WHO is present without anyone having to speak. diarization only knows a person
while they are talking; people come home long before they say anything.

Topics
  in   /face_detections   (String, JSON)  — boxes + 5 landmarks, from YuNet
       /camera/camera/color/image_raw
       /faces/enrol       (String)        — a name; the next clear face is
                                            enrolled as that person
       /faces/forget      (String)        — a name to remove
  out  /face_identities   (String, JSON)  — who is on screen, for the dashboard
       /face_arrivals     (String)        — a name, once, when someone arrives

Gallery persists in ~/.ros/home_robot_faces.json.

‼️ Enrolment is EXPLICIT. diarization auto-enrols and its gallery fills with
Speaker_2, Speaker_7… which name nobody. A face gallery does the same, except
the entries also look like people. So a name has to be asked for.

‼️ Below the similarity threshold the answer is 'unknown', never the argmax.
A robot that confidently calls one child by the other's name is worse than one
that says nothing.

The model (~37 MB ONNX) comes from HuggingFace on first run and is cached, the
same way YAMNet and the whisper models are. Nothing is committed to the repo.
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
from sensor_msgs.msg import Image
from std_msgs.msg import String

from home_robot.face_identity import (
    REFERENCE_LANDMARKS, FaceGallery, PresenceTracker,
)

_HF_REPO = 'opencv/face_recognition_sface'
_HF_FILE = 'face_recognition_sface_2021dec.onnx'
_GALLERY_PATH = os.path.expanduser('~/.ros/home_robot_faces.json')

# Faces smaller than this are too few pixels to embed reliably; SFace wants a
# 112x112 crop and upscaling a 30px face invents detail rather than finding it.
_MIN_FACE_PX = 60


class FaceIdentityNode(Node):
    def __init__(self):
        super().__init__('face_identity_node')

        self.declare_parameter('threshold', 0.363)   # OpenCV Zoo's SFace value
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('forget_after', 25.0)
        self.declare_parameter('min_face_px', _MIN_FACE_PX)
        self.declare_parameter('enrol_shots', 5)
        self.declare_parameter('num_threads', 2)

        self.min_face_px = self.get_parameter('min_face_px').value
        self.enrol_shots = self.get_parameter('enrol_shots').value

        self.bridge = CvBridge()
        self._session = None
        self._input_name = None
        self._frame = None
        self._lock = threading.Lock()
        self._enrolling = None       # (name, shots_remaining)
        self._last = []              # last frame's identities, for the UI

        self.gallery = self._load_gallery()
        self.gallery.threshold = self.get_parameter('threshold').value
        self.presence = PresenceTracker(
            confirm=self.get_parameter('confirm_frames').value,
            forget_after=self.get_parameter('forget_after').value)

        threading.Thread(target=self._load_model, daemon=True).start()

        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._cb_image, 3)
        self.create_subscription(String, '/face_detections', self._cb_faces, 5)
        self.create_subscription(String, '/faces/enrol', self._cb_enrol, 5)
        self.create_subscription(String, '/faces/forget', self._cb_forget, 5)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._id_pub = self.create_publisher(String, 'face_identities', latched)
        self._arrival_pub = self.create_publisher(String, 'face_arrivals', 10)

        self.create_timer(60.0, self._save_gallery)
        self.get_logger().info(
            f'Face identity starting — {len(self.gallery)} people enrolled '
            f'({", ".join(self.gallery.names()) or "none"})')

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
            self.get_logger().error(
                f'could not fetch SFace ({exc}) — face identity disabled. '
                f'It downloads once and caches; check the network.')
            return
        opts = ort.SessionOptions()
        # Capped for the same reason as everything else here: the default is
        # every core, which measured SLOWER and starved the rest of the stack.
        opts.intra_op_num_threads = self.get_parameter('num_threads').value
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(path, opts,
                                             providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name
        self.get_logger().info('SFace ready (~15 ms per face, CPU)')

    # ── state ─────────────────────────────────────────────────────────────────

    def _load_gallery(self):
        try:
            with open(_GALLERY_PATH, encoding='utf-8') as fh:
                return FaceGallery.from_dict(json.load(fh))
        except FileNotFoundError:
            return FaceGallery()
        except (json.JSONDecodeError, OSError) as exc:
            self.get_logger().warn(f'face gallery unreadable ({exc}) — empty')
            return FaceGallery()

    def _save_gallery(self):
        try:
            os.makedirs(os.path.dirname(_GALLERY_PATH), exist_ok=True)
            tmp = _GALLERY_PATH + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.gallery.to_dict(), fh)
            os.replace(tmp, _GALLERY_PATH)
        except OSError as exc:
            self.get_logger().warn(f'could not save face gallery: {exc}',
                                   throttle_duration_sec=300.0)

    # ── inputs ────────────────────────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return
        with self._lock:
            self._frame = frame

    def _cb_enrol(self, msg: String):
        name = (msg.data or '').strip()
        if not name:
            return
        self._enrolling = (name, self.enrol_shots)
        self.get_logger().info(
            f'Enrolling "{name}" — hold still, {self.enrol_shots} shots')

    def _cb_forget(self, msg: String):
        name = (msg.data or '').strip()
        if self.gallery.forget(name):
            self.presence.reset()
            self._save_gallery()
            self.get_logger().info(f'Forgot "{name}"')

    # ── the work ──────────────────────────────────────────────────────────────

    def _cb_faces(self, msg: String):
        if self._session is None:
            return
        try:
            faces = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(faces, list):
            return
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            return

        now = time.time()
        identities, names = [], []
        for face in faces:
            crop = self._align(frame, face)
            if crop is None:
                continue
            embedding = self._embed(crop)
            if embedding is None:
                continue

            if self._enrolling:
                enrol_name, left = self._enrolling
                self.gallery.enrol(enrol_name, embedding)
                left -= 1
                self._enrolling = (enrol_name, left) if left > 0 else None
                if self._enrolling is None:
                    self._save_gallery()
                    self.get_logger().info(f'Enrolled "{enrol_name}"')
                identities.append({'name': enrol_name, 'score': 1.0,
                                   'enrolling': True,
                                   'x1': face.get('x1'), 'y1': face.get('y1'),
                                   'x2': face.get('x2'), 'y2': face.get('y2')})
                continue

            name, score = self.gallery.match(embedding)
            identities.append({'name': name or 'unknown',
                               'score': round(float(score), 3),
                               'x1': face.get('x1'), 'y1': face.get('y1'),
                               'x2': face.get('x2'), 'y2': face.get('y2')})
            if name:
                names.append(name)

        for arrived in self.presence.update(names, now):
            self._arrival_pub.publish(String(data=arrived))
            self.get_logger().info(f'{arrived} is here')

        self._last = identities
        self._publish(now)

    def _align(self, frame, face):
        """Warp a detected face to the 112x112 SFace template.

        ‼️ Uses YuNet's five landmarks, not the bounding box. SFace was trained
        on crops aligned this way; a plain box crop costs a large part of the
        accuracy, and the failure is silent — embeddings still come out, they
        are just worse at telling people apart.
        """
        landmarks = face.get('landmarks') or []
        if len(landmarks) != 5:
            return None
        try:
            x1, y1 = int(face['x1']), int(face['y1'])
            x2, y2 = int(face['x2']), int(face['y2'])
        except (KeyError, TypeError, ValueError):
            return None
        if min(x2 - x1, y2 - y1) < self.min_face_px:
            return None

        src = np.array([[p['x'], p['y']] for p in landmarks], dtype=np.float32)
        dst = np.array(REFERENCE_LANDMARKS, dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if matrix is None:
            return None
        return cv2.warpAffine(frame, matrix, (112, 112),
                              borderValue=(0, 0, 0))

    def _embed(self, crop):
        blob = crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        try:
            return self._session.run(None, {self._input_name: blob})[0][0].tolist()
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().error(f'SFace inference failed: {exc}',
                                    throttle_duration_sec=30.0)
            return None

    def _publish(self, now):
        enrolling = self._enrolling
        self._id_pub.publish(String(data=json.dumps({
            'ready': self._session is not None,
            'enrolled': self.gallery.names(),
            'present': self.presence.present(now),
            'faces': self._last,
            'enrolling': ({'name': enrolling[0], 'left': enrolling[1]}
                          if enrolling else None),
        }, ensure_ascii=False)))


def main():
    rclpy.init()
    node = FaceIdentityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save_gallery()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
