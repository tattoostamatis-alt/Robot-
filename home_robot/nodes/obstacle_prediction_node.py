#!/usr/bin/env python3
"""
obstacle_prediction_node.py — Projects moving-object trajectories forward and
publishes predicted positions as PointCloud2 for Nav2's local costmap voxel layer.

Subscribes: tracked_objects (std_msgs/String, JSON from tracker_node.py)
            /camera/camera/color/camera_info (CameraInfo — for fx/fy/cx/cy)
Publishes:  predicted_obstacles (sensor_msgs/PointCloud2, base_link frame)

Add to nav2_params.yaml local_costmap voxel_layer observation_sources:
  observation_sources: scan depth_camera predicted_obstacles
  predicted_obstacles:
    topic: /predicted_obstacles
    clearing: false
    marking: true
    data_type: "PointCloud2"
    obstacle_max_range: 5.0
    raytrace_max_range: 5.0

Only objects with 3D speed > `min_speed` (m/s) are projected. Prediction
horizon is `horizon` seconds at `steps` steps — each step adds a point to
the published cloud so Nav2 inflates cost ahead of the object's path.
"""

import json
import math
import struct
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from std_msgs.msg import String

import tf2_ros
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs   # noqa: F401 — registers transform methods


def _make_pc2(points_xyz: list, frame_id: str, stamp) -> PointCloud2:
    """Build a minimal XYZ PointCloud2 from a list of (x, y, z) tuples."""
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    data = b''.join(struct.pack('fff', *p) for p in points_xyz)
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp    = stamp
    msg.height = 1
    msg.width  = max(len(points_xyz), 1)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = 12 * msg.width
    msg.data         = data
    msg.is_dense     = True
    return msg


class ObstaclePredictionNode(Node):
    def __init__(self):
        super().__init__('obstacle_prediction_node')

        self.declare_parameter('min_speed',   0.15)  # m/s — below this, don't predict
        self.declare_parameter('horizon',     2.0)   # seconds ahead
        self.declare_parameter('steps',       5)     # number of predicted positions per object
        self.declare_parameter('target_frame', 'base_link')

        self._min_speed    = self.get_parameter('min_speed').value
        self._horizon      = self.get_parameter('horizon').value
        self._steps        = self.get_parameter('steps').value
        self._target_frame = self.get_parameter('target_frame').value

        # Camera intrinsics (populated from /camera_info)
        self._fx = self._fy = self._cx = self._cy = None
        self._cam_lock = threading.Lock()

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(String,     'tracked_objects',                   self._on_tracked, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',  self._on_cam_info, 10)

        self._pub = self.create_publisher(PointCloud2, 'predicted_obstacles', 10)

        self.get_logger().info(
            f'Obstacle prediction ready — min_speed={self._min_speed}m/s '
            f'horizon={self._horizon}s steps={self._steps}'
        )

    def _on_cam_info(self, msg: CameraInfo):
        with self._cam_lock:
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]

    def _on_tracked(self, msg: String):
        try:
            tracks = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        with self._cam_lock:
            fx, fy, cx, cy = self._fx, self._fy, self._cx, self._cy

        if fx is None:
            return  # haven't received camera_info yet

        stamp = self.get_clock().now().to_msg()
        predicted_points: list[tuple] = []

        for t in tracks:
            # 3D position in camera_color_optical_frame (from object_detector)
            x_cam = t.get('x', 0.)
            y_cam = t.get('y', 0.)
            z_cam = t.get('z', 0.)

            if z_cam <= 0.1:
                continue

            # Convert pixel velocity → camera-frame 3D velocity (m/frame)
            # Tracker gives pixels/detection-cycle; we treat it as relative motion.
            vel_px = t.get('vel_x', 0.)
            vel_py = t.get('vel_y', 0.)
            vx_cam = vel_px * z_cam / fx   # m per detection cycle
            vy_cam = vel_py * z_cam / fy

            speed = math.sqrt(vx_cam**2 + vy_cam**2)
            if speed < self._min_speed * 0.1:
                # Too slow to be worth predicting (skip)
                continue

            # Project N future positions (constant-velocity, constant depth approx)
            for step in range(1, self._steps + 1):
                t_frac = step / self._steps
                xp = x_cam + vx_cam * self._horizon * t_frac
                yp = y_cam + vy_cam * self._horizon * t_frac
                zp = z_cam  # assume constant depth for simplicity

                # Transform from camera_color_optical_frame to target_frame
                pt_in = PointStamped()
                pt_in.header.frame_id = 'camera_color_optical_frame'
                pt_in.header.stamp    = stamp
                pt_in.point.x = float(xp)
                pt_in.point.y = float(yp)
                pt_in.point.z = float(zp)

                try:
                    pt_out = self._tf_buffer.transform(
                        pt_in, self._target_frame, timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    predicted_points.append(
                        (pt_out.point.x, pt_out.point.y, pt_out.point.z)
                    )
                except Exception:
                    # TF not ready yet or stale — skip this point
                    pass

        # Always publish (empty cloud clears old marks if clearing=true,
        # but we use clearing=false so publish only when there's something)
        if predicted_points:
            self._pub.publish(_make_pc2(predicted_points, self._target_frame, stamp))


def main():
    rclpy.init()
    node = ObstaclePredictionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
