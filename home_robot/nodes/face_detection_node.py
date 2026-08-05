#!/usr/bin/env python3
"""Face detection — YuNet through OpenCV's own FaceDetectorYN.

## ‼️ It never found a face. Not once.

Until 2026-08-05 this node ran a locally quantized `yunet_int8.onnx` and
published `[]` on every single frame, at 10 Hz, for a full core of CPU. Not
"sometimes missed a face" — the `obj` output of that graph was **exactly
0.000** at every anchor and every stride, and the score is `cls * obj`, so no
detection could ever clear any threshold. Confirmed on the live robot: 10 s of
`/face_detections` is 10 s of `data: '[]'`.

The cause is in scripts/quantize_npu_models.sh, which calibrates with

    np.random.randn(*shape)          # values around 0, sigma 1

while real frames are 0-255. INT8 calibration derives its activation scales
from that data, so every scale came out roughly two orders of magnitude too
small, activations saturated, and the objectness branch collapsed to zero. The
`cls` branch survived as noise — 0.45 to 0.87 at EVERY anchor, which is its own
tell that the graph is not doing anything.

So everything downstream was dead too, quietly: face_identity had nobody to
identify, and situational_awareness never had a face to report.

## What it does now, and what that costs

`cv2.FaceDetectorYN` with the stock opencv_zoo model. Decode and NMS happen in
C++ instead of the ~65 lines of Python that used to do it here, and the model
is float — no quantization to get wrong.

Measured on this box, per frame, at 10 Hz (see project_robot_perception_load):

    int8 640x640, 2 threads       64% of a core, 0 faces ever
    fp32 640x480, 1 thread        43% of a core, faces down to 70 px
    fp32 480x360, 1 thread        23% of a core, faces down to 90 px
    fp32 320x240, 1 thread         9% of a core, faces down to 160 px

‼️ MORE THREADS COST MORE CPU HERE, they only buy latency: 640x480 is 43 ms
wall / 43 ms CPU on one thread, and 25 ms wall / 51 ms CPU on four. One thread
at 5 Hz leaves the frame budget nine tenths empty, so there is nothing to buy.

640x480 is the default because of the last column. A 24 cm face at 480 px of
42.5 deg vertical FOV is ~148/d pixels tall, so 70 px is a person at ~2.1 m and
160 px is one at ~0.9 m — closer than anyone stands when talking to the robot.
Drop to 480x360 if the CPU matters more than the last half-metre.

## ‼️ OpenCV 4.6 cannot load this model

The system OpenCV is 4.6.0 and its FaceDetectorYN throws "Layer with requested
id=-1 not found" on the first detect() with the 2023mar model. The venv on the
path below has 4.11, which works — that import order is load-bearing, not a
leftover from the NPU experiment it was added for.
"""

import os
import sys

# ‼️ Order matters: this venv supplies OpenCV 4.11. See the module docstring —
# the system 4.6 cannot run this model at all.
_VENV_SITE = '/home/dimi/ryzenai_venv/lib/python3.12/site-packages'
if os.path.isdir(_VENV_SITE):
    sys.path.insert(0, _VENV_SITE)

import json
import threading
import urllib.request

import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yunet_face_2023mar.onnx')
# 232 KB. *.onnx is gitignored (weights are not committed), so a fresh checkout
# has no model — fetch it rather than come up healthy and blind, which is
# exactly the failure this rewrite exists to end.
_MODEL_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/'
              'face_detection_yunet/face_detection_yunet_2023mar.onnx')

_SCORE_THR = 0.6
_NMS_THR = 0.3
_TOP_K = 5000


def ensure_model(path: str = _MODEL_PATH, url: str = _MODEL_URL) -> bool:
    """Download the model if it is missing. True if it is there afterwards."""
    if os.path.isfile(path):
        return True
    tmp = path + '.part'
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
        return True
    except Exception:                                          # noqa: BLE001
        for leftover in (tmp,):
            try:
                os.remove(leftover)
            except OSError:
                pass
        return False


def to_json(faces, scale_x: float = 1.0, scale_y: float = 1.0) -> list:
    """FaceDetectorYN rows -> the JSON face_identity and the dashboard expect.

    A row is [x, y, w, h, then five landmark x/y pairs, then score]. The
    landmark ORDER is right eye, left eye, nose, right mouth corner, left mouth
    corner — the same order SFace's alignment template is built from, so it is
    passed through untouched. Reordering it would still produce embeddings; it
    would just quietly stop telling people apart (face_identity_node._align).

    Scales map back to full-frame coordinates when detection ran on a resized
    copy, since everything downstream indexes the original image.
    """
    out = []
    for row in faces if faces is not None else []:
        x, y, w, h = (float(v) for v in row[:4])
        out.append({
            'x1': int(x * scale_x),
            'y1': int(y * scale_y),
            'x2': int((x + w) * scale_x),
            'y2': int((y + h) * scale_y),
            'score': round(float(row[-1]), 2),
            'landmarks': [{'x': int(float(row[4 + 2 * i]) * scale_x),
                           'y': int(float(row[5 + 2 * i]) * scale_y)}
                          for i in range(5)],
        })
    return out


class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        # 6 = 5 Hz off a 30 Hz camera. Was 3 (10 Hz), which is more often than
        # a face moves and twice the cost. Recognition and "who is talking"
        # both work off a face that is there for seconds.
        self.declare_parameter('process_every_n', 6)
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 480)
        # See the docstring: extra threads buy latency this node does not need
        # and cost CPU it cannot spare.
        self.declare_parameter('threads', 1)

        self.process_every_n = int(self.get_parameter('process_every_n').value)
        self.input_size = (int(self.get_parameter('input_width').value),
                           int(self.get_parameter('input_height').value))
        self.threads = int(self.get_parameter('threads').value)

        self.bridge = CvBridge()
        self._detector = None
        self._frame_count = 0
        self._scale = (1.0, 1.0)
        self._warned_size = False

        threading.Thread(target=self._load_model, daemon=True).start()

        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._cb, 5)
        self.faces_pub = self.create_publisher(String, 'face_detections', 10)
        self.get_logger().info('Face detection node ready')

    def _load_model(self):
        if not ensure_model():
            self.get_logger().error(
                f'Face model missing and could not be downloaded: {_MODEL_PATH}. '
                'Face detection, face identity and "who is talking" are all '
                f'dead until it is there. Fetch it by hand from {_MODEL_URL}')
            return
        # cv2.setNumThreads() at runtime breaks FaceDetectorYN on 4.6; on 4.11
        # it is honoured, and it must be set before the detector is built.
        cv2.setNumThreads(self.threads)
        self._detector = cv2.FaceDetectorYN.create(
            _MODEL_PATH, '', self.input_size, _SCORE_THR, _NMS_THR, _TOP_K)
        self.get_logger().info(
            f'YuNet loaded — {self.input_size[0]}x{self.input_size[1]}, '
            f'{self.threads} thread(s), '
            f'{30.0 / max(1, self.process_every_n):.1f} Hz')

    def _cb(self, color_msg: Image):
        if self._detector is None:
            return
        self._frame_count += 1
        if self._frame_count % self.process_every_n != 0:
            return

        bgr = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        img_h, img_w = bgr.shape[:2]
        want_w, want_h = self.input_size
        if (img_w, img_h) != self.input_size:
            # ‼️ The detector's input size is fixed at construction: handing it
            # a differently sized frame is undefined, not automatic. Resize and
            # scale the boxes back, so the coordinates downstream stay in the
            # original frame.
            if not self._warned_size:
                self._warned_size = True
                self.get_logger().info(
                    f'Camera is {img_w}x{img_h}, detecting at {want_w}x{want_h}')
            bgr = cv2.resize(bgr, self.input_size)
            self._scale = (img_w / want_w, img_h / want_h)

        _n, faces = self._detector.detect(bgr)
        self.faces_pub.publish(String(data=json.dumps(
            to_json(faces, *self._scale))))


def main():
    rclpy.init()
    node = FaceDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
