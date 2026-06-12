#!/usr/bin/env python3
"""Object detector — subscribes to RealSense ROS2 topics + YOLOv8 → detected objects with 3D positions."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import numpy as np
import json
import threading

_yolo_model = None


def get_yolo(model_name='yolo11n.pt'):
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(model_name)
    return _yolo_model


CLUTTER_CLASSES = {
    'cup', 'bottle', 'book', 'remote', 'cell phone',
    'scissors', 'toothbrush', 'backpack', 'handbag',
    'tie', 'suitcase', 'umbrella', 'shoe',
}


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('model', 'yolo11n.pt')

        self.conf   = self.get_parameter('confidence').value
        model_name  = self.get_parameter('model').value
        self.bridge = CvBridge()
        self.model  = None
        self._fx = self._fy = self._cx = self._cy = None

        threading.Thread(target=self._load_model, args=(model_name,), daemon=True).start()

        # Subscribe to realsense2_camera ROS2 topics
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

    def _load_model(self, model_name):
        self.model = get_yolo(model_name)
        self.get_logger().info(f'YOLO model loaded: {model_name}')

    def _detect_cb(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        if self.model is None:
            return

        self._fx = info_msg.k[0]
        self._fy = info_msg.k[4]
        self._cx = info_msg.k[2]
        self._cy = info_msg.k[5]

        color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32) / 1000.0

        results  = self.model(color_img, conf=self.conf, verbose=False)[0]
        detected = []
        markers  = MarkerArray()

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            label  = self.model.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            depth_m = float(depth_img[cy, cx])

            if depth_m <= 0.1 or depth_m > 5.0:
                continue

            # Deproject pixel → 3D point
            px = (cx - self._cx) * depth_m / self._fx
            py = (cy - self._cy) * depth_m / self._fy
            pz = depth_m

            obj = {
                'label':   label,
                'conf':    round(conf, 2),
                'x':       round(px, 3),
                'y':       round(py, 3),
                'z':       round(pz, 3),
                'clutter': label in CLUTTER_CLASSES,
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
