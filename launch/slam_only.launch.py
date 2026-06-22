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
    roomba_port = LaunchConfiguration('roomba_port', default='/dev/roomba')

    # SLAMTEC C1 LIDAR is provided by ros-sllidar-c1.service (systemd,
    # always running on /dev/sllidar = same physical port as /dev/lidar).
    # Do NOT start another sllidar_node here — it would fail to open the
    # serial port (SL_RESULT_OPERATION_TIMEOUT) since it's already in use.

    # No obstacle_safety_node in this minimal launch, so listen on cmd_vel
    # directly (bringup.launch.py remaps to cmd_vel_safe instead).
    roomba_node = Node(
        package='home_robot',
        executable='roomba_driver.py',
        name='roomba_driver',
        parameters=[{'port': roomba_port}],
        output='screen',
    )

    # odom -> base_link is published dynamically by ekf_node (below), which
    # fuses roomba_driver.py's wheel velocity with the IMU's absolute yaw —
    # roomba_driver.py itself no longer broadcasts this TF. Do NOT add a
    # static fallback here — a static odom->base_link identity TF alongside
    # the dynamic one makes slam_toolbox think the robot never moves, so the
    # map never grows past the first scan.

    tf_base_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_laser',
        # Lidar remounted 2026-06-21: 120mm forward of the wheel axle,
        # 220mm above it. yaw=pi: previous mount (2026-06-17) had it
        # flipped front/back — see bringup.launch.py's tf_base_laser
        # comment for the live symptom that confirmed this (2026-06-18).
        arguments=['--x', '0.12', '--y', '0', '--z', '0.22',
                   '--roll', '0', '--pitch', '0', '--yaw', '3.14159265',
                   '--frame-id', 'base_link', '--child-frame-id', 'laser'],
    )

    # TODO: measure once the IMU is permanently mounted on the chassis
    # (currently breadboard-prototyped) — identity assumes it sits flat and
    # aligned with base_link (x forward, y left, z up).
    tf_base_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_imu',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'imu_link'],
    )

    imu_node = Node(
        package='home_robot',
        executable='imu_node.py',
        name='imu_node',
        output='screen',
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[PathJoinSubstitution([pkg, 'config', 'ekf.yaml'])],
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
        DeclareLaunchArgument('roomba_port', default_value='/dev/roomba'),
        roomba_node,
        tf_base_laser,
        tf_base_imu,
        imu_node,
        ekf_node,
        slam_node,
    ])
