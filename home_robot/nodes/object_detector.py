#!/usr/bin/env python3
"""Object detector — RealSense → YOLO11n (CPU) → 3D positions."""

import os
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


COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush',
]

CLUTTER_CLASSES = {
    'cup', 'bottle', 'book', 'remote', 'cell phone',
    'scissors', 'toothbrush', 'backpack', 'handbag',
    'tie', 'suitcase', 'umbrella', 'shoe',
}

_MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'yolo11n_int8.onnx')
_INPUT_SIZE   = 640


def _letterbox(img, size=640):
    """Resize + pad to square, return (padded_img, scale, pad_w, pad_h)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = (size - nh) // 2
    pad_w = (size - nw) // 2
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + nh, pad_w:pad_w + nw] = img_r
    return padded, scale, pad_w, pad_h


def _preprocess(bgr):
    padded, scale, pad_w, pad_h = _letterbox(bgr, _INPUT_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]  # NCHW
    return tensor, scale, pad_w, pad_h


def _nms(boxes, scores, iou_thr=0.45):
    """Simple NMS; boxes = (N,4) x1y1x2y2, scores = (N,)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def _postprocess(output, scale, pad_w, pad_h, conf_thr, img_w, img_h):
    """Decode YOLO11 output (1,84,8400) → list of (x1,y1,x2,y2,conf,cls_id)."""
    pred = output[0].T  # (8400, 84)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    class_scores = pred[:, 4:]              # (8400, 80)
    cls_ids = class_scores.argmax(axis=1)
    confs   = class_scores[np.arange(len(cls_ids)), cls_ids]

    mask = confs >= conf_thr
    if not mask.any():
        return []

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    confs   = confs[mask]
    cls_ids = cls_ids[mask]

    # cx/cy/w/h are in 640×640 letterbox space → original pixels
    x1 = ((cx - w / 2) - pad_w) / scale
    y1 = ((cy - h / 2) - pad_h) / scale
    x2 = ((cx + w / 2) - pad_w) / scale
    y2 = ((cy + h / 2) - pad_h) / scale

    x1 = np.clip(x1, 0, img_w - 1)
    y1 = np.clip(y1, 0, img_h - 1)
    x2 = np.clip(x2, 0, img_w - 1)
    y2 = np.clip(y2, 0, img_h - 1)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    results = []
    for cls in np.unique(cls_ids):
        m = cls_ids == cls
        keep = _nms(boxes[m], confs[m])
        for k in keep:
            idx = np.where(m)[0][k]
            results.append((
                int(boxes[idx, 0]), int(boxes[idx, 1]),
                int(boxes[idx, 2]), int(boxes[idx, 3]),
                float(confs[idx]), int(cls_ids[idx]),
            ))
    return results


def _build_session():
    # CPU-only: keeps NPU free for Qwen3 (XRT context-switch adds ~6s to LLM latency)
    return ort.InferenceSession(
        _MODEL_PATH,
        providers=['CPUExecutionProvider'],
    ), 'CPU'


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        self.declare_parameter('confidence',     0.5)
        self.declare_parameter('process_every_n', 6)

        self.conf            = self.get_parameter('confidence').value
        self.process_every_n = self.get_parameter('process_every_n').value
        self.bridge          = CvBridge()
        self._session        = None
        self._backend        = 'loading'
        self._fx = self._fy = self._cx = self._cy = None
        self._frame_count    = 0

        threading.Thread(target=self._load_model, daemon=True).start()

        color_sub = Subscriber(self, Image,      '/camera/camera/color/image_raw')
        depth_sub = Subscriber(self, Image,      '/camera/camera/aligned_depth_to_color/image_raw')
        info_sub  = Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')

        self._sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._detect_cb)

        self.objects_pub = self.create_publisher(String,      'detected_objects', 10)
        self.markers_pub = self.create_publisher(MarkerArray, 'object_markers',   10)

        self.get_logger().info('Object detector ready — waiting for camera topics...')

    def _load_model(self):
        sess, backend = _build_session()
        self._session = sess
        self._backend = backend
        self.get_logger().info(f'YOLO11n loaded on {backend} ({_MODEL_PATH})')

    def _detect_cb(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        if self._session is None:
            return
        self._frame_count += 1
        if self._frame_count % self.process_every_n != 0:
            return

        self._fx = info_msg.k[0]
        self._fy = info_msg.k[4]
        self._cx = info_msg.k[2]
        self._cy = info_msg.k[5]

        color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32) / 1000.0
        img_h, img_w = depth_img.shape[:2]

        tensor, scale, pad_w, pad_h = _preprocess(color_img)
        raw = self._session.run(None, {'images': tensor})
        detections = _postprocess(raw[0], scale, pad_w, pad_h, self.conf, img_w, img_h)

        detected = []
        markers  = MarkerArray()

        for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(detections):
            label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else str(cls_id)

            bx = (x1 + x2) // 2
            by = (y1 + y2) // 2
            depth_m     = float(depth_img[by, bx])
            box_distance = self._box_distance(depth_img, x1, y1, x2, y2, img_w, img_h)

            if depth_m <= 0.1 or depth_m > 5.0:
                continue

            px = (bx - self._cx) * depth_m / self._fx
            py = (by - self._cy) * depth_m / self._fy
            pz = depth_m

            obj = {
                'label':   label,
                'conf':    round(conf, 2),
                'x':       round(px, 3),
                'y':       round(py, 3),
                'z':       round(pz, 3),
                'clutter': label in CLUTTER_CLASSES,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'img_w': img_w, 'img_h': img_h,
                'box_distance': round(box_distance, 3) if box_distance is not None else None,
            }
            detected.append(obj)

            m = Marker()
            m.header.frame_id = 'camera_color_optical_frame'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns, m.id        = 'objects', i
            m.type            = Marker.TEXT_VIEW_FACING
            m.action          = Marker.ADD
            m.pose.position.x = pz
            m.pose.position.y = -px
            m.pose.position.z = -py
            m.pose.orientation.w = 1.0
            m.scale.z = 0.1
            m.color.a = 1.0
            m.color.r, m.color.g = (1.0, 0.0) if obj['clutter'] else (0.0, 1.0)
            m.text = f"{label} {conf:.0%}"
            markers.markers.append(m)

        self.objects_pub.publish(String(data=json.dumps(detected)))
        self.markers_pub.publish(markers)

    @staticmethod
    def _box_distance(depth_m_img, x1, y1, x2, y2, width, height):
        ix1, iy1 = max(int(x1), 0), max(int(y1), 0)
        ix2, iy2 = min(int(x2), width), min(int(y2), height)
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        region = depth_m_img[iy1:iy2, ix1:ix2]
        valid  = region[region > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid))


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
