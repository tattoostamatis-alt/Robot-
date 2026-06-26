#!/usr/bin/env python3
"""Pose estimation — YOLO11n-pose on NPU (VitisAI EP) → 17 COCO keypoints per person."""

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
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

_VAIP_CONFIG = '/home/dimi/ryzenai_venv/voe-4.0-linux_x86_64/vaip_config.json'
_MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'yolo11n_pose_int8.onnx')
_INPUT_SIZE  = 640

COCO_KP_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

COCO_KP_EDGES = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]


def _letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = (size - nh) // 2, (size - nw) // 2
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h+nh, pad_w:pad_w+nw] = img_r
    return padded, scale, pad_w, pad_h


def _preprocess(bgr):
    padded, scale, pad_w, pad_h = _letterbox(bgr)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    t = rgb.astype(np.float32) / 255.0
    return np.transpose(t, (2,0,1))[np.newaxis], scale, pad_w, pad_h


def _nms(boxes, scores, iou_thr=0.45):
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order=order[1:][iou<=iou_thr]
    return keep


def _postprocess(output, scale, pad_w, pad_h, conf_thr, img_w, img_h):
    """Decode YOLO11n-pose output (1,56,8400) → list of person dicts."""
    pred = output[0].T  # (8400, 56)
    # 4 box + 1 conf + 51 keypoints (17×3)
    cx, cy, w, h = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
    confs = pred[:,4]
    kps   = pred[:,5:]  # (8400, 51)

    mask = confs >= conf_thr
    if not mask.any():
        return []

    cx,cy,w,h = cx[mask],cy[mask],w[mask],h[mask]
    confs = confs[mask]
    kps   = kps[mask]

    x1 = np.clip(((cx-w/2)-pad_w)/scale, 0, img_w-1).astype(int)
    y1 = np.clip(((cy-h/2)-pad_h)/scale, 0, img_h-1).astype(int)
    x2 = np.clip(((cx+w/2)-pad_w)/scale, 0, img_w-1).astype(int)
    y2 = np.clip(((cy+h/2)-pad_h)/scale, 0, img_h-1).astype(int)

    boxes = np.stack([x1,y1,x2,y2], axis=1).astype(float)
    keep  = _nms(boxes, confs)

    results = []
    for k in keep:
        raw_kp = kps[k].reshape(17, 3)
        keypoints = []
        for j, (kx, ky, kv) in enumerate(raw_kp):
            px = int(np.clip((kx - pad_w) / scale, 0, img_w-1))
            py = int(np.clip((ky - pad_h) / scale, 0, img_h-1))
            keypoints.append({'name': COCO_KP_NAMES[j], 'x': px, 'y': py, 'v': float(kv)})
        results.append({
            'x1': int(x1[k]), 'y1': int(y1[k]), 'x2': int(x2[k]), 'y2': int(y2[k]),
            'conf': round(float(confs[k]), 2),
            'keypoints': keypoints,
        })
    return results


def _build_session():
    providers = ort.get_available_providers()
    if 'VitisAIExecutionProvider' in providers and os.path.isfile(_VAIP_CONFIG):
        return ort.InferenceSession(
            _MODEL_PATH,
            providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
            provider_options=[{'config_file': _VAIP_CONFIG}, {}]), 'NPU'
    return ort.InferenceSession(_MODEL_PATH, providers=['CPUExecutionProvider']), 'CPU'


class PoseNode(Node):
    def __init__(self):
        super().__init__('pose_node')
        self.declare_parameter('confidence',      0.5)
        self.declare_parameter('process_every_n', 2)

        self.conf            = self.get_parameter('confidence').value
        self.process_every_n = self.get_parameter('process_every_n').value
        self.bridge          = CvBridge()
        self._session        = None
        self._frame_count    = 0

        threading.Thread(target=self._load_model, daemon=True).start()

        color_sub = Subscriber(self, Image,      '/camera/camera/color/image_raw')
        info_sub  = Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')
        self._sync = ApproximateTimeSynchronizer(
            [color_sub, info_sub], queue_size=5, slop=0.05)
        self._sync.registerCallback(self._cb)

        self.poses_pub  = self.create_publisher(String,      'pose_detections', 10)
        self.markers_pub = self.create_publisher(MarkerArray, 'pose_markers',   10)
        self.get_logger().info('Pose node ready')

    def _load_model(self):
        if not os.path.isfile(_MODEL_PATH):
            self.get_logger().warn(
                f'Pose model not found: {_MODEL_PATH} — run scripts/quantize_npu_models.sh first')
            return
        sess, backend = _build_session()
        self._session = sess
        self.get_logger().info(f'YOLO11n-pose loaded on {backend}')

    def _cb(self, color_msg: Image, info_msg: CameraInfo):
        if self._session is None:
            return
        self._frame_count += 1
        if self._frame_count % self.process_every_n != 0:
            return

        color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        img_h, img_w = color_img.shape[:2]

        tensor, scale, pad_w, pad_h = _preprocess(color_img)
        raw = self._session.run(None, {'images': tensor})
        persons = _postprocess(raw[0], scale, pad_w, pad_h, self.conf, img_w, img_h)

        markers = MarkerArray()
        for i, p in enumerate(persons):
            for j, kp in enumerate(p['keypoints']):
                if kp['v'] < 0.5:
                    continue
                m = Marker()
                m.header.frame_id = 'camera_color_optical_frame'
                m.header.stamp    = self.get_clock().now().to_msg()
                m.ns, m.id        = 'keypoints', i * 17 + j
                m.type            = Marker.SPHERE
                m.action          = Marker.ADD
                m.pose.position.x = float(kp['x']) / img_w
                m.pose.position.y = float(kp['y']) / img_h
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.02
                m.color.a = 1.0; m.color.g = 1.0
                markers.markers.append(m)

        self.poses_pub.publish(String(data=json.dumps(persons)))
        self.markers_pub.publish(markers)


def main():
    rclpy.init()
    node = PoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
