#!/usr/bin/env python3
"""Record a NeRF dataset while the robot drives — images plus MEASURED poses.

The expensive, fragile half of any NeRF pipeline is normally structure-from-
motion: COLMAP guessing where each photo was taken. It is slow, and it fails
exactly where a flat is hardest — corridors of blank wall with no features to
match. This robot does not need it. AMCL localises against a map it built, and
TF gives map -> camera_color_optical_frame for every single frame. The poses are
measured, not estimated.

Output is the nerfstudio `transforms.json` layout, so the same folder feeds
scripts/train_nerf.py here and any other tool later without conversion.

    ros2 topic pub -1 /nerf/capture std_msgs/msg/Bool '{data: true}'   # start
    ros2 topic pub -1 /nerf/capture std_msgs/msg/Bool '{data: false}'  # stop

The dashboard's NeRF tab drives the same topic.

‼️ Frames are only kept when the camera has actually MOVED. Standing still for a
minute at 30 Hz would otherwise write 1800 near-identical views, which teach the
model nothing, take a gigabyte, and bias training toward wherever the robot
happened to pause.
"""
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String


def _quat_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class NerfCaptureNode(Node):
    def __init__(self):
        super().__init__('nerf_capture_node')
        self.declare_parameter('out_dir', '~/.home_robot/nerf/house')
        # Thresholds for "the view changed enough to be worth a frame". 12 cm /
        # 8 degrees gives good coverage of a room at walking pace without
        # flooding the set.
        self.declare_parameter('min_translation', 0.12)   # m
        self.declare_parameter('min_rotation', 0.14)      # rad (~8 deg)
        self.declare_parameter('max_frames', 600)
        self.declare_parameter('image_width', 640)        # downscaled on save
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')

        self.out_dir = os.path.expanduser(self.get_parameter('out_dir').value)
        self.min_t = self.get_parameter('min_translation').value
        self.min_r = self.get_parameter('min_rotation').value
        self.max_frames = self.get_parameter('max_frames').value
        self.img_w = self.get_parameter('image_width').value
        self.cam_frame = self.get_parameter('camera_frame').value

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._active = False
        self._frames = []          # accumulated transforms.json entries
        self._last_pose = None     # (t, R) of the last kept frame
        self._last_xy = None       # (x, y) of the last kept frame, for the live map
        self._K = None
        self._info = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(CameraInfo,
                                 '/camera/camera/color/camera_info',
                                 self._cb_info, 5)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._cb_image, 5)
        self.create_subscription(Bool, '/nerf/capture', self._cb_capture, 10)
        self._status_pub = self.create_publisher(String, '/nerf/status', 10)
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(f'NeRF capture ready → {self.out_dir}')

    # ── control ──────────────────────────────────────────────────────────
    def _archive_existing(self):
        """Move a previously-saved capture out of the way before starting a new
        one. Without this, the new session's image indices restart at 00000 and
        collide with the old files, and _write_transforms() overwrites the old
        transforms.json with the new (shorter) frame list — silently discarding
        the poses for every old photo the new session doesn't reach again.

        This is exactly what happened on 2026-08-04: an 11s accidental restart
        after a full 599-frame salon capture wiped that capture's pose data,
        leaving 599 orphaned .jpg files with no transforms.json entry.
        """
        marker = os.path.join(self.out_dir, 'transforms.json')
        if not os.path.exists(marker):
            return
        backup = f'{self.out_dir}.bak-{time.strftime("%Y%m%d_%H%M%S")}'
        try:
            os.rename(self.out_dir, backup)
            self.get_logger().info(f'archived previous capture → {backup}')
        except OSError as e:
            self.get_logger().error(f'could not archive previous capture: {e}')

    def _cb_capture(self, msg: Bool):
        with self._lock:
            if msg.data and not self._active:
                self._archive_existing()
                self._frames, self._last_pose, self._last_xy = [], None, None
                os.makedirs(os.path.join(self.out_dir, 'images'), exist_ok=True)
                self._active = True
                self.get_logger().info('capture STARTED')
            elif not msg.data and self._active:
                self._active = False
                self._write_transforms()
                self.get_logger().info(
                    f'capture STOPPED — {len(self._frames)} frames in {self.out_dir}')

    def _cb_info(self, msg: CameraInfo):
        if self._K is None:
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self._info = (msg.width, msg.height)

    # ── frame intake ─────────────────────────────────────────────────────
    def _cb_image(self, msg: Image):
        with self._lock:
            if not self._active or self._K is None:
                return
            if len(self._frames) >= self.max_frames:
                return
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', self.cam_frame, rclpy.time.Time())
        except Exception:
            return                        # not localised yet; nothing to anchor to

        tr = tf.transform.translation
        rot = tf.transform.rotation
        t = np.array([tr.x, tr.y, tr.z])
        R = _quat_to_R(rot.x, rot.y, rot.z, rot.w)

        with self._lock:
            if self._last_pose is not None:
                dt = float(np.linalg.norm(t - self._last_pose[0]))
                # Rotation angle between the two orientations, from the trace of
                # R_prev^T R — cheaper and steadier than comparing quaternions.
                cos = (np.trace(self._last_pose[1].T @ R) - 1.0) / 2.0
                dr = float(math.acos(max(-1.0, min(1.0, cos))))
                if dt < self.min_t and dr < self.min_r:
                    return                # the view has not changed enough
            idx = len(self._frames)

        try:
            img = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return
        h, w = img.shape[:2]
        scale = 1.0
        if w > self.img_w:
            scale = self.img_w / float(w)
            img = cv2.resize(img, (self.img_w, int(round(h * scale))))

        name = f'images/{idx:05d}.jpg'
        cv2.imwrite(os.path.join(self.out_dir, name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, t
        with self._lock:
            self._frames.append({
                'file_path': name,
                'transform_matrix': c2w.tolist(),
            })
            self._last_pose = (t, R)
            self._last_xy = (float(t[0]), float(t[1]))
            if len(self._frames) >= self.max_frames:
                self._active = False
                self._write_transforms()
                self.get_logger().warn(f'max_frames ({self.max_frames}) reached')

    # ── output ───────────────────────────────────────────────────────────
    def _write_transforms(self):
        """nerfstudio-format transforms.json. Caller holds the lock."""
        if not self._frames or self._K is None:
            return
        src_w, src_h = self._info
        s = self.img_w / float(src_w) if src_w > self.img_w else 1.0
        data = {
            # Intrinsics scale with the images, or every ray is wrong by the
            # same factor the pictures were shrunk by.
            'fl_x': self._K[0, 0] * s, 'fl_y': self._K[1, 1] * s,
            'cx': self._K[0, 2] * s,   'cy': self._K[1, 2] * s,
            'w': int(round(src_w * s)), 'h': int(round(src_h * s)),
            'camera_model': 'OPENCV',
            # The poses are in the ROS optical convention (x right, y down,
            # z forward) and in the MAP frame — the same coordinates as the 2D
            # map, the taught rooms and the RTAB-Map cloud.
            'convention': 'opencv',
            'frames': self._frames,
        }
        path = os.path.join(self.out_dir, 'transforms.json')
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)          # atomic: a half-written set is unusable

    def _publish_status(self):
        with self._lock:
            self._status_pub.publish(String(data=json.dumps({
                'active': self._active,
                'frames': len(self._frames),
                'max_frames': self.max_frames,
                'dir': self.out_dir,
                'last_xy': list(self._last_xy) if self._last_xy else None,
            })))


def main():
    rclpy.init()
    node = NerfCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
