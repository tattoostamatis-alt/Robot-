#!/usr/bin/env python3
"""
semantic_costmap_node.py — Inflates specific object categories in Nav2's costmap
by publishing a wider PointCloud2 footprint around them on /semantic_obstacles.

"Avoid" categories (person, dog, cat, baby, child) get a cylindrical expansion
of radius `person_radius` meters so Nav2 keeps a larger margin around people.
Other moving objects get `object_radius`. Static ignored categories are skipped.

Add to nav2_params.yaml local_costmap voxel_layer:
  observation_sources: scan depth_camera predicted_obstacles semantic_obstacles
  semantic_obstacles:
    topic: /semantic_obstacles
    clearing: false
    marking: true
    data_type: "PointCloud2"
    obstacle_max_range: 5.0
    raytrace_max_range: 5.0

Subscribes: detected_objects OR tracked_objects (JSON)
Publishes:  /semantic_obstacles (PointCloud2, base_link frame)
"""

import json
import math
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from std_msgs.msg import String
from nav2_msgs.srv import ClearEntireCostmap

import tf2_ros
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs  # noqa: F401


# Classes that deserve extra personal-space inflation
PERSON_CLASSES  = {'person', 'dog', 'cat', 'horse', 'sheep', 'cow', 'elephant',
                   'bear', 'zebra', 'giraffe'}
OBJECT_CLASSES  = {'bicycle', 'motorcycle', 'car', 'truck', 'bus'}
# Classes to skip entirely (e.g. things mounted on walls/ceiling)
IGNORE_CLASSES  = {'tv', 'laptop', 'remote', 'clock', 'book', 'mouse', 'keyboard'}


def _cylinder_points(cx: float, cy: float, cz: float,
                     radius: float, height: float = 1.8,
                     rings: int = 3, sectors: int = 8) -> list:
    """Return ~rings*sectors points forming a cylinder around (cx,cy,cz)."""
    pts = []
    for r in range(rings):
        z = cz - 0.1 + (height / (rings - 1)) * r if rings > 1 else cz
        for s in range(sectors):
            angle = 2 * math.pi * s / sectors
            pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), z))
    return pts


def _make_pc2(points_xyz: list, frame_id: str, stamp) -> PointCloud2:
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


class SemanticCostmapNode(Node):
    def __init__(self):
        super().__init__('semantic_costmap_node')

        self.declare_parameter('person_radius', 0.6)   # meters around a person
        self.declare_parameter('object_radius', 0.35)  # meters around a vehicle/bike
        self.declare_parameter('max_range',     5.0)   # ignore detections farther than this
        self.declare_parameter('target_frame',  'base_link')

        self._person_r     = self.get_parameter('person_radius').value
        self._object_r     = self.get_parameter('object_radius').value
        self._max_range    = self.get_parameter('max_range').value
        self._target_frame = self.get_parameter('target_frame').value

        self._cam_lock = threading.Lock()
        self._fx = self._fy = self._cx = self._cy = None
        self._tracked_rx = 0.0    # last tracked_objects arrival (see _on_detected)

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Use camera_info for depth-from-bbox fallback when z not available
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self._on_cam_info, 10)
        # tracked_objects preferred (has IDs+velocity); fallback to detected_objects
        self.create_subscription(String, 'tracked_objects',  self._on_tracked, 10)
        self.create_subscription(String, 'detected_objects', self._on_detected, 10)

        self._pub = self.create_publisher(PointCloud2, '/semantic_obstacles', 10)
        self._clear_local = self.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')
        self._clear_global = self.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap')
        # Start true so a node restart while detections are already empty also
        # removes semantic marks left in Nav2 by the previous process.
        self._had_semantic_points = True

        self.get_logger().info(
            f'Semantic costmap node ready — person_r={self._person_r}m '
            f'object_r={self._object_r}m'
        )

    # ‼️ tracked_objects is the PREFERRED source; detected_objects is a fallback
    # for when the tracker is not running. Both used to be wired to the same
    # callback with no preference at all despite the comment claiming one, so
    # with perception up every detection cycle was processed TWICE — once
    # tracked, once raw — alternating between the two views of the same scene.
    _TRACKED_TTL = 2.0        # s: consider the tracker live for this long

    def _on_tracked(self, msg):
        self._tracked_rx = time.monotonic()
        self._on_objects(msg)

    def _on_detected(self, msg):
        if time.monotonic() - self._tracked_rx < self._TRACKED_TTL:
            return            # the tracker is live; ignore the raw duplicate
        self._on_objects(msg)

    def _on_cam_info(self, msg: CameraInfo):
        with self._cam_lock:
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]

    def _on_objects(self, msg: String):
        try:
            objects = json.loads(msg.data) or []
        except json.JSONDecodeError:
            return

        stamp = self.get_clock().now().to_msg()
        all_pts: list[tuple] = []

        for obj in objects:
            label = obj.get('label', '').lower()
            if label in IGNORE_CLASSES:
                continue

            z_cam = obj.get('z', 0.)
            if z_cam <= 0.05 or z_cam > self._max_range:
                continue

            x_cam = obj.get('x', 0.)
            y_cam = obj.get('y', 0.)

            # Choose inflation radius by category
            if label in PERSON_CLASSES:
                radius = self._person_r
                height = 1.8
            elif label in OBJECT_CLASSES:
                radius = self._object_r
                height = 1.2
            else:
                radius = 0.25
                height = 1.0

            # Build cylinder in camera frame then transform each ring point
            ring_pts = _cylinder_points(x_cam, y_cam, z_cam, radius, height)

            for (rx, ry, rz) in ring_pts:
                pt_in = PointStamped()
                pt_in.header.frame_id = 'camera_color_optical_frame'
                pt_in.header.stamp    = stamp
                pt_in.point.x = float(rx)
                pt_in.point.y = float(ry)
                pt_in.point.z = float(rz)
                try:
                    pt_out = self._tf_buffer.transform(
                        pt_in, self._target_frame,
                        timeout=rclpy.duration.Duration(seconds=0.05)
                    )
                    all_pts.append((pt_out.point.x, pt_out.point.y, pt_out.point.z))
                except Exception:
                    pass

        if all_pts:
            self._had_semantic_points = True
            self._pub.publish(_make_pc2(all_pts, self._target_frame, stamp))
        elif self._had_semantic_points:
            # semantic_obstacles is deliberately marking-only: raytracing a
            # synthetic cylinder would erase real scan obstacles between the
            # camera and the object. Consequently Nav2 cannot clear a vanished
            # semantic object by itself. Clear once on the nonempty->empty
            # edge; the live scan immediately repopulates genuine obstacles.
            self._had_semantic_points = False
            request = ClearEntireCostmap.Request()
            for client in (self._clear_local, self._clear_global):
                if client.service_is_ready():
                    client.call_async(request)
            self.get_logger().info(
                'Semantic detections cleared — removed stale costmap marks')


def main():
    rclpy.init()
    node = SemanticCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
