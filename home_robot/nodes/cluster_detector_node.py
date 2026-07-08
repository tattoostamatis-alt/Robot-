#!/usr/bin/env python3
"""Publish geometric object clusters from the vendored pointcloud pipeline.

Periodically calls `/<filter_ns>/get_cluster_positions` (served by the
`pointcloud_pipeline` node), transforms the detected object centroids into a
reference frame (default `arm_base`), and republishes them as:
  * ``~/clusters``         (std_msgs/String, JSON) - easy to eyeball / consume
  * ``~/cluster_markers``  (visualization_msgs/MarkerArray) - drop into RViz

This is the observable, arm-free way to verify the perception pipeline works on
real D435 data before wiring it into a grasp. It picks NO object and moves
nothing -- it only reports what's on the surface in front of the robot.

Uses async service calls from a timer (never spin_until_future_complete in a
callback) per project convention.

Launch the pipeline + this node together via
``ros2 launch home_robot perception_clusters.launch.py``.
"""
import json

from geometry_msgs.msg import Point
from home_robot.pcl_clusters import transform_and_sort
from interbotix_perception_msgs.srv import ClusterInfoArray
import rclpy
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
import tf2_ros
# registers PointStamped with tf2 (imported for its side effect)
import tf2_geometry_msgs  # noqa: F401
from visualization_msgs.msg import Marker, MarkerArray


class ClusterDetector(Node):
    def __init__(self):
        super().__init__('cluster_detector')
        self.declare_parameter('filter_ns', 'pc_filter')
        self.declare_parameter('ref_frame', 'arm_base')
        self.declare_parameter('sort_axis', 'y')
        self.declare_parameter('reverse', True)
        self.declare_parameter('is_parallel', True)
        self.declare_parameter('period', 1.0)

        p = self.get_parameter
        filter_ns = p('filter_ns').value
        self.ref_frame = p('ref_frame').value
        self.sort_axis = p('sort_axis').value
        self.reverse = p('reverse').value
        self.is_parallel = p('is_parallel').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_json = self.create_publisher(String, '~/clusters', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '~/cluster_markers', 10)

        self.cli = self.create_client(
            ClusterInfoArray, f'/{filter_ns}/get_cluster_positions')
        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().warning(
                f"'/{filter_ns}/get_cluster_positions' not up yet; will keep retrying.")

        self._pending = None
        self.create_timer(p('period').value, self._tick)
        self.get_logger().info(
            f"cluster_detector polling '/{filter_ns}/get_cluster_positions', "
            f"reporting in frame '{self.ref_frame}'.")

    def _tick(self):
        if self._pending is not None and not self._pending.done():
            return  # don't stack requests if the pipeline is slow
        if not self.cli.service_is_ready():
            return
        self._pending = self.cli.call_async(ClusterInfoArray.Request())
        self._pending.add_done_callback(self._on_clusters)

    def _on_clusters(self, future):
        try:
            raw = future.result().clusters
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'cluster service call failed: {exc}')
            return
        clusters = transform_and_sort(
            self, self.tf_buffer, raw, self.ref_frame,
            sort_axis=self.sort_axis, reverse=self.reverse,
            is_parallel=self.is_parallel)

        msg = String()
        msg.data = json.dumps({'frame': self.ref_frame, 'clusters': clusters})
        self.pub_json.publish(msg)
        self.pub_markers.publish(self._to_markers(clusters))
        if clusters:
            self.get_logger().info(
                f'{len(clusters)} cluster(s): ' +
                ', '.join(f"{c['name']}={[round(v, 3) for v in c['position']]}"
                          for c in clusters))

    def _to_markers(self, clusters) -> MarkerArray:
        arr = MarkerArray()
        # clear previous markers so stale spheres don't linger
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        now = self.get_clock().now().to_msg()
        for i, c in enumerate(clusters):
            m = Marker()
            m.header.frame_id = self.ref_frame
            m.header.stamp = now
            m.ns = 'clusters'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=c['position'][0], y=c['position'][1],
                                    z=c['position'][2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.04
            r, g, b = (v / 255.0 for v in c['color'])
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            arr.markers.append(m)
        return arr


def main():
    rclpy.init()
    node = ClusterDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
