"""Minimal launch — LIDAR + SLAM only (για πρώτες δοκιμές χαρτογράφησης)."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('home_robot')
    roomba_port = LaunchConfiguration('roomba_port', default='/dev/roomba')
    use_joy = LaunchConfiguration('use_joy', default='true')
    # DualSense is /dev/input/js0 by default (confirmed live 2026-06-22) —
    # js1 is its motion-sensor sub-device, not the gamepad itself.
    joy_device_id = LaunchConfiguration('joy_device_id', default='0')

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
        # Remeasured 2026-07-01: 150mm forward of the wheel axle, 220mm above
        # it, centered left/right. yaw=pi: the unit is mounted rotated 180
        # degrees in the horizontal plane (connector facing back), so the laser
        # frame is rotated 180 degrees about Z to align scans with base_link.
        arguments=['--x', '0.15', '--y', '0', '--z', '0.22',
                   '--roll', '0', '--pitch', '0', '--yaw', '3.14159265',
                   '--frame-id', 'base_link', '--child-frame-id', 'laser'],
    )

    # IMU mounting orientation, measured 2026-07-01 from the BNO085's own
    # gravity-referenced reading while the robot sat level: the board is
    # mounted UPSIDE-DOWN (roll ~= -177.3 deg) with a slight -2.8 deg pitch.
    # These rotations bring imu_link into base_link so the fused orientation
    # reads level (verified: base_link roll/pitch -> 0.0 deg over 448 samples).
    # Yaw is left 0 -- the game rotation vector yaw is arbitrary each boot and
    # AMCL absorbs it via map->odom.
    # TODO: translation still 0,0,0 -- measure the IMU's x/y/z on the chassis.
    tf_base_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_imu',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '-3.0952', '--pitch', '-0.0490', '--yaw', '0',
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

    # ── PS5 (DualSense) teleop for manual mapping runs ────────────
    # Publishes straight to cmd_vel — roomba_node above listens on cmd_vel
    # directly in this minimal launch (no obstacle_safety_node here to
    # remap through), same as teleop_twist_keyboard would.
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'device_id': joy_device_id, 'deadzone': 0.05}],
        output='screen',
        condition=IfCondition(use_joy),
    )

    teleop_twist_joy_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        parameters=[PathJoinSubstitution([pkg, 'config', 'teleop_twist_joy_ps5.yaml'])],
        output='screen',
        condition=IfCondition(use_joy),
    )

    return LaunchDescription([
        DeclareLaunchArgument('roomba_port', default_value='/dev/roomba'),
        DeclareLaunchArgument('use_joy',     default_value='true'),
        DeclareLaunchArgument('joy_device_id', default_value='0'),
        roomba_node,
        tf_base_laser,
        tf_base_imu,
        imu_node,
        ekf_node,
        slam_node,
        joy_node,
        teleop_twist_joy_node,
    ])
