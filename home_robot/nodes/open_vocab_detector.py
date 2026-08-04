#!/usr/bin/env python3
"""Open-vocabulary detection — find things COCO never heard of.

object_detector knows 80 fixed classes. "Φέρε μου τα κλειδιά" was therefore
impossible: keys, glasses, chargers, wallets and remotes are simply not in
COCO, so `fetch`/`pick`/`goto_object` could only ever work on cups and books.
YOLO-World takes the class list as a PROMPT, so the vocabulary becomes whatever
was just asked for.

Topics
  in   /vocab/set        (String)  — JSON list or comma-separated names, Greek
                                     or English. Empty string clears and idles.
       /camera/camera/color/image_raw
       /camera/camera/aligned_depth_to_color/image_raw
       /camera/camera/color/camera_info
  out  /open_vocab_detections (String, JSON) — SAME schema as /detected_objects,
                                     so object_memory and pick_place consume it
                                     without knowing which detector found it
       /vocab/state      (String, JSON) — what it is hunting for, for the UI
       /open_vocab_markers (MarkerArray)

‼️ Idle by default, and it returns to idle `idle_timeout` seconds after the last
request. This is the whole reason it is a separate node: a second detector
running continuously on the same iGPU is exactly the load problem that already
bit this project (see the face_detection thread-count fix). It costs nothing
until someone asks for something COCO cannot find, and `needs_open_vocab`
declines even then if COCO already covers it.
"""

# Must be set BEFORE torch is imported (via ultralytics) so ROCm accepts the
# gfx1150 iGPU. setdefault: a launch-file env override still wins.
import os
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import json
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from home_robot.vocabulary import greek_for, normalize_vocabulary

_MODEL = 'yolov8s-worldv2.pt'


class OpenVocabDetector(Node):
    def __init__(self):
        super().__init__('open_vocab_detector')

        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('conf', 0.05)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('process_every_n', 3)
        self.declare_parameter('idle_timeout', 90.0)     # s
        self.declare_parameter('max_classes', 16)

        self._device_req  = self.get_parameter('device').value
        self.conf         = self.get_parameter('conf').value
        self.imgsz        = self.get_parameter('imgsz').value
        self.every_n      = self.get_parameter('process_every_n').value
        self.idle_timeout = self.get_parameter('idle_timeout').value
        self.max_classes  = self.get_parameter('max_classes').value

        self.bridge = CvBridge()
        self._model = None
        self._device = None
        self._lock = threading.Lock()
        self._vocab = []                 # active prompts, [] == idle
        self._vocab_set_at = 0.0
        self._frame_count = 0
        self._depth = None
        self._intrinsics = None
        self._last_hits = []
        # Counts completed inferences, NOT frames received. A caller waiting on
        # a result has to tell "no detections yet, still starting up" from "ran
        # and found nothing" — both publish hits == [], and only this
        # distinguishes them.
        self._inferences = 0

        threading.Thread(target=self._load_model, daemon=True).start()

        self.create_subscription(String, '/vocab/set', self._cb_vocab, 5)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._cb_color, 3)
        self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw',
            self._cb_depth, 3)
        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._cb_info, 5)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._det_pub    = self.create_publisher(String, 'open_vocab_detections', 10)
        self._state_pub  = self.create_publisher(String, 'vocab/state', latched)
        self._marker_pub = self.create_publisher(MarkerArray, 'open_vocab_markers', 5)

        self.create_timer(5.0, self._tick_idle)
        self._publish_state()
        self.get_logger().info('Open-vocabulary detector ready (idle)')

    # ── model ─────────────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            import torch
            from ultralytics import YOLOWorld
        except ImportError as exc:
            self.get_logger().error(f'ultralytics/torch missing: {exc}')
            return

        device = self._device_req
        if device.startswith('cuda') and not torch.cuda.is_available():
            self.get_logger().warn('ROCm/CUDA unavailable — open-vocab on CPU '
                                   '(slow; expect seconds per frame)')
            device = 'cpu'
        try:
            model = YOLOWorld(_MODEL)
            model.to(device)
        except Exception as exc:                          # noqa: BLE001
            # Most likely the weights could not be downloaded on first run.
            self.get_logger().error(f'could not load {_MODEL}: {exc}')
            return

        # ‼️ Warm CLIP up HERE, not on the first request. set_classes encodes
        # the prompts with CLIP ViT-B/32, and the very first call downloads it
        # (354 MB) — on the live run that blocked the executor for ~2.5 minutes
        # while holding _lock, so the node looked frozen and every /vocab/set
        # went unanswered. Doing it during load keeps the wait inside the
        # "loading" state, where it belongs, and later requests are instant.
        try:
            model.set_classes(['object'])
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().error(
                f'CLIP text encoder unavailable ({exc}) — open vocabulary needs '
                f'it. Install with: pip install --user clip-anytorch')
            return

        self._device = device
        self._model = model
        self.get_logger().info(f'YOLO-World + CLIP ready on {device}')
        self._publish_state()

    # ── vocabulary ────────────────────────────────────────────────────────────

    def _cb_vocab(self, msg: String):
        raw = (msg.data or '').strip()
        if not raw:
            self._set_vocab([])
            return
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                items = [str(items)]
        except json.JSONDecodeError:
            items = [p for p in raw.split(',') if p.strip()]

        prompts = normalize_vocabulary(items, limit=self.max_classes)
        if not prompts:
            self.get_logger().warn(f'nothing recognisable in vocabulary {raw!r}')
            self._publish_state(error=f'άγνωστο: {raw}')
            return
        self._set_vocab(prompts)

    def _set_vocab(self, prompts):
        with self._lock:
            self._vocab = prompts
            self._vocab_set_at = time.time()
            self._last_hits = []
            if prompts and self._model is not None:
                # set_classes re-encodes the text prompts; it is the expensive
                # part, so do it on change and never per frame.
                try:
                    self._model.set_classes(prompts)
                except Exception as exc:                  # noqa: BLE001
                    self.get_logger().error(f'set_classes failed: {exc}')
                    self._vocab = []
        if prompts:
            self.get_logger().info(f'Hunting for: {", ".join(prompts)}')
        else:
            self.get_logger().info('Vocabulary cleared — idle')
        self._publish_state()

    def _tick_idle(self):
        with self._lock:
            active = bool(self._vocab)
            age = time.time() - self._vocab_set_at
        if active and self.idle_timeout > 0 and age > self.idle_timeout:
            self.get_logger().info(
                f'No request for {self.idle_timeout:.0f}s — releasing the GPU')
            self._set_vocab([])

    # ── camera ────────────────────────────────────────────────────────────────

    def _cb_info(self, msg: CameraInfo):
        self._intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _cb_depth(self, msg: Image):
        try:
            self._depth = self.bridge.imgmsg_to_cv2(
                msg, '16UC1').astype(np.float32) / 1000.0
        except Exception:                                  # malformed frame
            pass

    def _cb_color(self, msg: Image):
        with self._lock:
            vocab = list(self._vocab)
        if not vocab or self._model is None:
            return
        self._frame_count += 1
        if self._frame_count % self.every_n != 0:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return
        img_h, img_w = bgr.shape[:2]

        try:
            res = self._model.predict(bgr, device=self._device, imgsz=self.imgsz,
                                      conf=self.conf, verbose=False)[0]
        except Exception as exc:                           # noqa: BLE001
            self.get_logger().error(f'inference failed: {exc}',
                                    throttle_duration_sec=30.0)
            return

        objects = []
        if res.boxes is not None and len(res.boxes):
            xyxy  = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss  = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clss):
                label = vocab[ci] if ci < len(vocab) else str(ci)
                obj = {
                    'label': label,
                    'conf': round(float(cf), 2),
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                    'img_w': img_w, 'img_h': img_h,
                    # Marked so downstream can tell a prompted find from a COCO
                    # detection — the open-vocab confidences are not comparable.
                    'open_vocab': True,
                    'clutter': False,
                }
                obj.update(self._locate(x1, y1, x2, y2))
                objects.append(obj)

        self._inferences += 1
        self._last_hits = objects
        self._det_pub.publish(String(data=json.dumps(objects, ensure_ascii=False)))
        self._publish_markers(objects)
        self._publish_state()

    def _locate(self, x1, y1, x2, y2):
        """Camera-frame 3D for a box centre, matching object_detector's schema.

        Returns x/y/z as None when depth is unavailable rather than 0.0 — a
        zero would read as "at the camera" and put a phantom object on the map.
        """
        blank = {'x': None, 'y': None, 'z': None, 'box_distance': None}
        if self._depth is None or self._intrinsics is None:
            return blank
        fx, fy, cx, cy = self._intrinsics
        h, w = self._depth.shape[:2]
        u = int(np.clip((x1 + x2) / 2, 0, w - 1))
        v = int(np.clip((y1 + y2) / 2, 0, h - 1))
        patch = self._depth[max(0, v - 4):v + 5, max(0, u - 4):u + 5]
        valid = patch[patch > 0.0]
        if valid.size == 0:
            return blank
        z = float(np.median(valid))
        if not (0.1 < z <= 6.0):
            return blank
        return {'x': round((u - cx) * z / fx, 3),
                'y': round((v - cy) * z / fy, 3),
                'z': round(z, 3),
                'box_distance': round(z, 3)}

    # ── outputs ───────────────────────────────────────────────────────────────

    def _publish_state(self, error=None):
        with self._lock:
            vocab = list(self._vocab)
        self._state_pub.publish(String(data=json.dumps({
            'active': bool(vocab),
            'vocabulary': vocab,
            'greek': [greek_for(p) for p in vocab],
            'ready': self._model is not None,
            'device': self._device,
            'inferences': self._inferences,
            'hits': [{'label': o['label'], 'conf': o['conf'], 'z': o.get('z')}
                     for o in self._last_hits],
            'error': error,
        }, ensure_ascii=False)))

    def _publish_markers(self, objects):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        clear = Marker()
        clear.header.frame_id = 'camera_color_optical_frame'
        clear.header.stamp = stamp
        clear.ns, clear.id, clear.action = 'open_vocab', 0, Marker.DELETEALL
        arr.markers.append(clear)
        for i, obj in enumerate(objects):
            if obj.get('z') is None:
                continue
            m = Marker()
            m.header.frame_id = 'camera_color_optical_frame'
            m.header.stamp = stamp
            m.ns, m.id = 'open_vocab', i + 1
            m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            m.pose.position.x = float(obj['x'])
            m.pose.position.y = float(obj['y'])
            m.pose.position.z = float(obj['z'])
            m.pose.orientation.w = 1.0
            m.scale.z = 0.09
            m.color.r, m.color.g, m.color.b, m.color.a = 0.4, 0.9, 1.0, 0.95
            m.text = f"{obj['label']} {obj['conf']:.2f}"
            arr.markers.append(m)
        self._marker_pub.publish(arr)


def main():
    rclpy.init()
    node = OpenVocabDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
