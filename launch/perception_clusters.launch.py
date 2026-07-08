"""Geometric object-clustering perception, adapted from Interbotix LoCoBot.

Starts:
  * ``pointcloud_pipeline`` (vendored interbotix_perception_pipelines, BSD-3):
    PCL CropBox -> Voxel -> RANSAC plane removal -> Euclidean clustering on the
    D435 cloud, exposing ``/<filter_ns>/get_cluster_positions``.
  * ``cluster_detector`` (home_robot): polls that service and republishes the
    object centroids as JSON + RViz markers in the ``ref_frame``.

Needs the D435 pointcloud running (bringup with pointcloud.enable:=true) and
the base_link->arm_base TF present. Picks/moves nothing -- verification only.

  ros2 launch home_robot perception_clusters.launch.py
  ros2 launch home_robot perception_clusters.launch.py enable_pipeline:=true  # continuous (for tuning)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    filter_ns = LaunchConfiguration('filter_ns')
    cloud_topic = LaunchConfiguration('cloud_topic')
    enable_pipeline = LaunchConfiguration('enable_pipeline')
    ref_frame = LaunchConfiguration('ref_frame')
    filter_params = LaunchConfiguration('filter_params')

    args = [
        DeclareLaunchArgument('filter_ns', default_value='pc_filter'),
        DeclareLaunchArgument(
            'cloud_topic',
            default_value='/camera/camera/depth/color/points',
            description='D435 organized pointcloud topic to cluster.'),
        DeclareLaunchArgument(
            'enable_pipeline', default_value='false',
            description=('run PCL continuously (true, for tuning/RViz) vs only '
                         'when get_cluster_positions is called (false).')),
        DeclareLaunchArgument(
            'ref_frame', default_value='arm_base',
            description='frame the cluster positions are reported in.'),
        DeclareLaunchArgument(
            'filter_params',
            default_value=PathJoinSubstitution([
                FindPackageShare('home_robot'), 'config', 'pcl_filter_params.yaml']),
            description='YAML of PCL crop/voxel/cluster tuning params.'),
    ]

    pointcloud_pipeline = Node(
        package='interbotix_perception_pipelines',
        executable='pointcloud_pipeline',
        name='pointcloud_pipeline',
        namespace=filter_ns,
        output='screen',
        parameters=[
            filter_params,
            {'cloud_topic': cloud_topic, 'enable_pipeline': enable_pipeline},
        ],
    )

    cluster_detector = Node(
        package='home_robot',
        executable='cluster_detector_node.py',
        name='cluster_detector',
        output='screen',
        parameters=[{'filter_ns': filter_ns, 'ref_frame': ref_frame}],
    )

    return LaunchDescription(args + [pointcloud_pipeline, cluster_detector])
