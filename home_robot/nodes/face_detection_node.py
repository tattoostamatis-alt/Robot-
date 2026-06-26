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
_INPUT_H     = 192
_INPUT_W     = 320
_SCORE_THR   = 0.6
_NMS_THR     = 0.3


def _build_session():
    providers = ort.get_available_providers()
    if 'VitisAIExecutionProvider' in providers and os.path.isfile(_VAIP_CONFIG):
        return ort.InferenceSession(
            _MODEL_PATH,
            providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
            provider_options=[{'config_file': _VAIP_CONFIG}, {}]), 'NPU'
    return ort.InferenceSession(_MODEL_PATH, providers=['CPUExecutionProvider']), 'CPU'


def _decode_yunet(outputs, img_w, img_h, score_thr, nms_thr):
    """
    YuNet post-processing.
    Outputs: cls_scores (N,1), bbox_preds (N,4), kps_preds (N,10)
    Anchor strides: 8,16,32 over (320,192) input.
    """
    faces = []
    # YuNet ONNX typically returns one fused output: (1, N, 15)
    # columns: score, x1, y1, w, h, lm0x, lm0y, ..., lm4x, lm4y
    if len(outputs) == 1:
        dets = outputs[0][0]  # (N, 15)
        scores = dets[:, 0]
        mask   = scores >= score_thr
        dets   = dets[mask]
        if not len(dets):
            return []

        scores = dets[:, 0]
        bboxes = dets[:, 1:5].copy()  # x, y, w, h in input coords
        # Scale to original image
        sx = img_w / _INPUT_W
        sy = img_h / _INPUT_H
        bboxes[:, 0] *= sx; bboxes[:, 2] *= sx
        bboxes[:, 1] *= sy; bboxes[:, 3] *= sy
        x1 = bboxes[:,0]; y1 = bboxes[:,1]
        x2 = x1+bboxes[:,2]; y2 = y1+bboxes[:,3]

        boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)
        indices = cv2.dnn.NMSBoxes(
            bboxes[:,:4].tolist(), scores.tolist(), score_thr, nms_thr)
        if len(indices) == 0:
            return []

        for idx in np.array(indices).flatten():
            landmarks = []
            for lm in range(5):
                lx = float(dets[idx, 5 + lm*2])   * sx
                ly = float(dets[idx, 5 + lm*2+1]) * sy
                landmarks.append({'x': int(lx), 'y': int(ly)})
            faces.append({
                'x1': int(x1[idx]), 'y1': int(y1[idx]),
                'x2': int(x2[idx]), 'y2': int(y2[idx]),
                'score': round(float(scores[idx]), 2),
                'landmarks': landmarks,
            })
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
        inp = resized.astype(np.float32)[np.newaxis]  # (1, H, W, 3)

        try:
            outputs = self._session.run(None, {'input': inp})
        except Exception:
            # Try NCHW format
            inp_t = np.transpose(resized, (2,0,1))[np.newaxis].astype(np.float32)
            outputs = self._session.run(None, {self._session.get_inputs()[0].name: inp_t})

        faces = _decode_yunet(outputs, img_w, img_h, _SCORE_THR, _NMS_THR)
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
