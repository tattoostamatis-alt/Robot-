#!/usr/bin/env python3
"""Pose estimation — YOLO11n-pose on the iGPU (ROCm) → 17 COCO keypoints per person.

Runs on the AMD Radeon 860M via Ultralytics + torch-ROCm (device='cuda:0'),
keeping the NPU free for Qwen3 and offloading the CPU. fp16 is NOT used (aborts
with HSA_STATUS_ERROR_INVALID_ISA on this arch). Falls back to CPU automatically.
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

_MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'yolo11n-pose.pt')
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


class PoseNode(Node):
    def __init__(self):
        super().__init__('pose_node')
        self.declare_parameter('confidence',      0.5)
        self.declare_parameter('process_every_n', 2)
        # 'cuda:0' = Radeon 860M iGPU via ROCm; 'cpu' to force CPU fallback.
        self.declare_parameter('device',          'cuda:0')

        self.conf            = self.get_parameter('confidence').value
        self.process_every_n = self.get_parameter('process_every_n').value
        self._device_req     = self.get_parameter('device').value
        self.bridge          = CvBridge()
        self._model          = None
        self._device         = None
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
            self.get_logger().warn(f'Pose model not found: {_MODEL_PATH}')
            return
        # Import here (background thread) so torch/ROCm init never blocks startup.
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
        warm = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
        model.predict(warm, device=device, imgsz=_INPUT_SIZE, half=False, verbose=False)

        self._model  = model
        self._device = device
        backend = 'iGPU/ROCm' if device.startswith('cuda') else 'CPU'
        self.get_logger().info(f'YOLO11n-pose loaded on {backend} ({_MODEL_PATH})')

    def _cb(self, color_msg: Image, info_msg: CameraInfo):
        if self._model is None:
            return
        self._frame_count += 1
        if self._frame_count % self.process_every_n != 0:
            return

        color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        img_h, img_w = color_img.shape[:2]

        # Ultralytics returns boxes + keypoints already scaled to original pixels.
        res = self._model.predict(color_img, device=self._device, imgsz=_INPUT_SIZE,
                                  conf=self.conf, half=False, verbose=False)[0]
        persons = []
        if res.boxes is not None and len(res.boxes) and res.keypoints is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            kpts  = res.keypoints.data.cpu().numpy()  # (N, 17, 3): x, y, v
            for (x1, y1, x2, y2), cf, kp in zip(xyxy, confs, kpts):
                keypoints = [
                    {'name': COCO_KP_NAMES[j],
                     'x': int(max(0, min(kx, img_w - 1))),
                     'y': int(max(0, min(ky, img_h - 1))),
                     'v': float(kv)}
                    for j, (kx, ky, kv) in enumerate(kp)
                ]
                persons.append({
                    'x1': int(max(0, min(x1, img_w - 1))), 'y1': int(max(0, min(y1, img_h - 1))),
                    'x2': int(max(0, min(x2, img_w - 1))), 'y2': int(max(0, min(y2, img_h - 1))),
                    'conf': round(float(cf), 2),
                    'keypoints': keypoints,
                })

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
