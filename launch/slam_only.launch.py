"""Minimal launch — LIDAR + SLAM only (για πρώτες δοκιμές χαρτογράφησης)."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('home_robot')

    # SLAMTEC C1 LIDAR is provided by ros-sllidar-c1.service (systemd,
    # always running on /dev/sllidar = same physical port as /dev/lidar).
    # Do NOT start another sllidar_node here — it would fail to open the
    # serial port (SL_RESULT_OPERATION_TIMEOUT) since it's already in use.

    # odom -> base_link is published dynamically by roomba_driver.py
    # (run separately). Do NOT add a static fallback here — a static
    # odom->base_link identity TF alongside the dynamic one makes
    # slam_toolbox think the robot never moves, so the map never grows
    # past the first scan.

    tf_base_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_laser',
        arguments=['0.12', '0', '0.026', '0', '0', '0', 'base_link', 'laser'],
    )

    slam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('slam_toolbox'), '/launch/online_async_launch.py'
        ]),
        launch_arguments={
            'slam_params_file': PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml']),
            'use_sim_time':     'false',
        }.items(),
    )

    return LaunchDescription([
        tf_base_laser,
        slam_node,
    ])
