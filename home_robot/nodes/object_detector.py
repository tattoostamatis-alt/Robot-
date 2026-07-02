#!/usr/bin/env python3
"""Object detector — RealSense → YOLO11n (iGPU/ROCm) → 3D positions.

Inference runs on the AMD Radeon 860M iGPU via Ultralytics + torch-ROCm
(device='cuda:0'), which keeps the NPU 100% free for Qwen3 and offloads the
CPU (freeing it for the faster-whisper STT). ~93 FPS on the iGPU vs ~46 on CPU.
fp16 is NOT used — it aborts with HSA_STATUS_ERROR_INVALID_ISA on this arch.
Falls back to CPU automatically if the iGPU is unavailable.
"""

# Must be set BEFORE torch is imported (via ultralytics) so ROCm accepts the
# gfx1150 iGPU. setdefault: a launch-file env override still wins.
import os
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import json
import threading

import numpy as np

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

_MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'yolo11n.pt')
_INPUT_SIZE   = 640


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        self.declare_parameter('confidence',     0.5)
        self.declare_parameter('process_every_n', 6)
        # 'cuda:0' = Radeon 860M iGPU via ROCm; 'cpu' to force CPU fallback.
        self.declare_parameter('device',          'cuda:0')

        self.conf            = self.get_parameter('confidence').value
        self.process_every_n = self.get_parameter('process_every_n').value
        self._device_req     = self.get_parameter('device').value
        self.bridge          = CvBridge()
        self._model          = None
        self._device         = None
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
        # Import here (background thread) so torch/ROCm init never blocks node startup.
        import torch
        from ultralytics import YOLO

        device = self._device_req
        if device.startswith('cuda') and not torch.cuda.is_available():
            self.get_logger().warn(
                f'iGPU/ROCm not available (HSA_OVERRIDE_GFX_VERSION='
                f'{os.environ.get("HSA_OVERRIDE_GFX_VERSION")}); falling back to CPU.')
            device = 'cpu'

        model = YOLO(_MODEL_PATH)
        model.to(device)
        # Warm up off the callback path: pays the first-call JIT/alloc cost now.
        warm = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
        model.predict(warm, device=device, imgsz=_INPUT_SIZE,
                      half=False, verbose=False)

        self._model   = model
        self._device  = device
        self._backend = ('iGPU/ROCm' if device.startswith('cuda') else 'CPU')
        self.get_logger().info(f'YOLO11n loaded on {self._backend} ({_MODEL_PATH})')

    def _detect_cb(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        if self._model is None:
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

        # Ultralytics accepts a BGR ndarray and returns boxes already scaled back
        # to original-image pixels (it handles letterbox + NMS internally).
        res = self._model.predict(color_img, device=self._device, imgsz=_INPUT_SIZE,
                                  conf=self.conf, half=False, verbose=False)[0]
        b = res.boxes
        detections = []
        if b is not None and len(b):
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            clss  = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clss):
                detections.append((
                    int(max(0, min(x1, img_w - 1))), int(max(0, min(y1, img_h - 1))),
                    int(max(0, min(x2, img_w - 1))), int(max(0, min(y2, img_h - 1))),
                    float(cf), int(cl),
                ))

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
