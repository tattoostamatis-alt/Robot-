#!/usr/bin/env python3
"""Face detection — YuNet (quantized) on NPU → bounding boxes + landmarks."""

import os
import sys

_VENV_SITE = '/home/dimi/ryzenai_venv/lib/python3.12/site-packages'
if os.path.isdir(_VENV_SITE):
    sys.path.insert(0, _VENV_SITE)
os.environ.setdefault('XILINX_XRT', '/opt/xilinx/xrt')
os.environ.setdefault('RYZEN_AI_INSTALLATION_PATH', '/home/dimi/ryzenai_venv')

import cv2
import json
import threading

import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

_VAIP_CONFIG = '/home/dimi/ryzenai_venv/voe-4.0-linux_x86_64/vaip_config.json'
_MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'yunet_int8.onnx')
# ‼️ Must match the ONNX graph, which is [1, 3, 640, 640] NCHW. These said
# 192x320 and every inference died with "Got invalid dimensions for input:
# index 3 Got: 320 Expected: 640" — the node crashed on its first frame, on
# every launch, so use_face_detection:=true silently brought up nothing.
# Verified against the model with onnx.load(); re-check if the model is swapped.
_INPUT_H     = 640
_INPUT_W     = 640
_SCORE_THR   = 0.6
_NMS_THR     = 0.3


def _build_session():
    # YuNet is 158 KB, but the graph is 640x640 float32 NCHW, so a pass is ~30ms,
    # not the <3ms the 320x320 model would cost. No benefit mapping it to the NPU.
    #
    # ‼️ Cap the threads. onnxruntime defaults intra_op to every core, and on this
    # 16-core box that is a hard loss: measured 136ms/pass at the default against
    # 55ms on one thread and 29ms on four — the tiny per-op tensors spend all their
    # time in barrier sync. It also pinned ~9.5 cores at 10fps and drove load past
    # 60, which starved the rest of the stack. Four threads is the knee of the
    # curve; two is within 30% of it for a quarter of the cores, and at
    # process_every_n=3 that leaves the 100ms frame budget with room to spare.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        _MODEL_PATH, opts, providers=['CPUExecutionProvider']), 'CPU (2 threads)'


def _decode_multiscale(outputs, img_w, img_h, score_thr, nms_thr):
    """YuNet's real output: 12 tensors, 3 strides x (cls, obj, bbox, kps).

    ‼️ The previous decoder only handled a single fused (1, N, 15) tensor and
    returned [] for anything else. This model emits the multi-scale form, so
    face detection produced ZERO faces on every frame — silently, with the node
    alive and no error. Discovered 2026-08-03 after fixing the input size that
    had been crashing it outright.

    Layout, confirmed against the shapes for a 640x640 input
    (80x80=6400, 40x40=1600, 20x20=400):
        [cls_8, cls_16, cls_32, obj_8, obj_16, obj_32,
         bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
    Score is cls * obj. Boxes are (cx, cy, w, h) in anchor units: centre as an
    offset from the anchor in stride units, size in log space.
    """
    strides = (8, 16, 32)
    n = len(strides)
    cls_o, obj_o = outputs[0:n], outputs[n:2 * n]
    box_o, kps_o = outputs[2 * n:3 * n], outputs[3 * n:4 * n]

    boxes, scores, landmarks = [], [], []
    for i, stride in enumerate(strides):
        cols = _INPUT_W // stride
        rows = _INPUT_H // stride
        cls = np.array(cls_o[i]).reshape(-1)
        obj = np.array(obj_o[i]).reshape(-1)
        box = np.array(box_o[i]).reshape(-1, 4)
        kps = np.array(kps_o[i]).reshape(-1, 10)
        conf = np.clip(cls, 0, 1) * np.clip(obj, 0, 1)

        keep = np.nonzero(conf >= score_thr)[0]
        if keep.size == 0:
            continue
        ax = (keep % cols) * stride
        ay = (keep // cols) * stride

        cx = ax + box[keep, 0] * stride
        cy = ay + box[keep, 1] * stride
        w = np.exp(box[keep, 2]) * stride
        h = np.exp(box[keep, 3]) * stride

        sx, sy = img_w / _INPUT_W, img_h / _INPUT_H
        for j, k in enumerate(keep):
            boxes.append([float((cx[j] - w[j] / 2) * sx),
                          float((cy[j] - h[j] / 2) * sy),
                          float(w[j] * sx), float(h[j] * sy)])
            scores.append(float(conf[k]))
            landmarks.append([
                {'x': int((ax[j] + kps[k, 2 * m] * stride) * sx),
                 'y': int((ay[j] + kps[k, 2 * m + 1] * stride) * sy)}
                for m in range(5)])

    if not boxes:
        return []
    idx = cv2.dnn.NMSBoxes(boxes, scores, score_thr, nms_thr)
    faces = []
    for i in np.array(idx).flatten() if len(idx) else []:
        x, y, w, h = boxes[i]
        faces.append({'x1': int(x), 'y1': int(y),
                      'x2': int(x + w), 'y2': int(y + h),
                      'score': round(scores[i], 2),
                      'landmarks': landmarks[i]})
    return faces




class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        self.declare_parameter('process_every_n', 3)

        self.process_every_n = self.get_parameter('process_every_n').value
        self.bridge          = CvBridge()
        self._session        = None
        self._frame_count    = 0

        threading.Thread(target=self._load_model, daemon=True).start()

        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._cb, 5)
        self.faces_pub = self.create_publisher(String, 'face_detections', 10)
        self.get_logger().info('Face detection node ready')

    def _load_model(self):
        if not os.path.isfile(_MODEL_PATH):
            self.get_logger().warn(
                f'Face model not found: {_MODEL_PATH} — run scripts/quantize_npu_models.sh first')
            return
        sess, backend = _build_session()
        self._session = sess
        self.get_logger().info(f'YuNet loaded on {backend}')

    def _cb(self, color_msg: Image):
        if self._session is None:
            return
        self._frame_count += 1
        if self._frame_count % self.process_every_n != 0:
            return

        bgr = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        img_h, img_w = bgr.shape[:2]

        resized = cv2.resize(bgr, (_INPUT_W, _INPUT_H))
        # NCHW first: that is what this graph declares. The NHWC attempt below
        # stays as a fallback for a differently-exported model.
        inp = np.transpose(resized, (2, 0, 1))[np.newaxis].astype(np.float32)

        try:
            outputs = self._session.run(None, {'input': inp})
        except Exception:
            # Try NCHW format
            inp_t = resized.astype(np.float32)[np.newaxis]   # (1, H, W, 3)
            outputs = self._session.run(None, {self._session.get_inputs()[0].name: inp_t})

        faces = _decode_multiscale(outputs, img_w, img_h, _SCORE_THR, _NMS_THR)
        self.faces_pub.publish(String(data=json.dumps(faces)))


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
