"""Slim client-side helpers for the vendored pointcloud clustering pipeline.

Wraps the geometric object-clustering node `pointcloud_pipeline` (BSD-3-Clause,
Trossen Robotics; vendored under robot_ws/src/interbotix_perception_pipelines).
That node runs a PCL pipeline on the D435 cloud
(CropBox -> VoxelGrid -> RANSAC plane removal -> Euclidean clustering ->
radius-outlier removal) and answers the `/<filter_ns>/get_cluster_positions`
service with the centroid of every object sitting on a support surface -- with
NO class label required. This complements pick_place_node.py, which needs a
YOLO label to pick a *named* object; this path picks *whatever clutter* is
there.

Deliberately dependency-light: uses tf2 + tf2_geometry_msgs directly instead of
interbotix_common_modules, so it drops into home_robot cleanly. The blocking
service call is left to the caller (do it async from a timer callback to avoid
the spin_until_future_complete-in-worker-thread trap, per project convention);
this module only provides the pure transform/sort step and a params helper.
"""
from __future__ import annotations

from typing import List

from geometry_msgs.msg import PointStamped
from interbotix_perception_msgs.srv import FilterParams
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform


def filter_params_request(params: dict) -> FilterParams.Request:
    """Build a FilterParams service request from a plain dict (see config yaml)."""
    req = FilterParams.Request()
    for key, val in params.items():
        if hasattr(req, key):
            setattr(req, key, val)
    return req


def transform_and_sort(
    node: Node,
    tf_buffer: tf2_ros.Buffer,
    clusters,
    ref_frame: str,
    sort_axis: str = 'y',
    reverse: bool = False,
    is_parallel: bool = True,
) -> List[dict]:
    """Transform raw ClusterInfo[] (camera frame) into ref_frame and sort them.

    Mirrors interbotix_perception_modules' get_cluster_positions post-processing
    without its heavy deps.

    :param clusters: the ClusterInfo[] straight from the service response
    :param ref_frame: TF frame to express positions in (e.g. 'arm_base')
    :param sort_axis: 'x' | 'y' | 'z' of ref_frame to sort the returned list by
    :param reverse: descending sort when True
    :param is_parallel: if True (ref_frame roughly parallel to the surface), the
        returned z is taken from the cluster's top point (min_z_point) rather
        than its centroid -- the point you actually want to grasp toward
    :return: list of dicts {'name','position':[x,y,z],'color':[r,g,b],'num_points'},
        empty if no clusters or the TF lookup failed
    """
    log = node.get_logger()
    if not clusters:
        return []

    cluster_frame = clusters[0].frame_id

    def _tf_point(px, py, pz):
        ps = PointStamped()
        ps.header.frame_id = cluster_frame
        ps.point.x, ps.point.y, ps.point.z = px, py, pz
        try:
            out = tf_buffer.transform(ps, ref_frame, timeout=tf2_ros.Duration(seconds=1.0))
        except tf2_ros.TransformException as exc:  # Lookup/Connectivity/Extrapolation
            log.error(f"TF '{cluster_frame}' -> '{ref_frame}' failed: {exc}")
            return None
        return out.point

    out_clusters: List[dict] = []
    for i, c in enumerate(clusters):
        p = _tf_point(c.position.x, c.position.y, c.position.z)
        if p is None:
            return []  # if TF is unavailable, nothing downstream is trustworthy
        z = p.z
        if is_parallel:
            top = _tf_point(c.min_z_point.x, c.min_z_point.y, c.min_z_point.z)
            if top is not None:
                z = top.z
        out_clusters.append({
            'name': f'cluster_{i + 1}',
            'position': [p.x, p.y, z],
            'color': [c.color.r, c.color.g, c.color.b],
            'num_points': c.num_points,
        })

    axis = {'x': 0, 'y': 1, 'z': 2}.get(sort_axis, 1)
    if sort_axis not in ('x', 'y', 'z'):
        log.warning(f"Invalid sort_axis '{sort_axis}', defaulting to 'y'.")
    out_clusters.sort(key=lambda c: c['position'][axis], reverse=reverse)
    return out_clusters
