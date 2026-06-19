"""Full robot bringup — LIDAR + SLAM + Nav2 + RealSense + Roomba driver + RViz2."""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, GroupAction,
                            EmitEvent, RegisterEventHandler)
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace, LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
from lifecycle_msgs.msg import Transition
import os


def generate_launch_description():
    pkg = FindPackageShare('home_robot')
    nav2_pkg = FindPackageShare('nav2_bringup')

    use_slam      = LaunchConfiguration('use_slam',      default='true')
    use_camera    = LaunchConfiguration('use_camera',    default='true')
    use_voice     = LaunchConfiguration('use_voice',     default='false')
    use_arm       = LaunchConfiguration('use_arm',       default='false')
    use_wake_word = LaunchConfiguration('use_wake_word', default='false')
    use_stt       = LaunchConfiguration('use_stt',       default='false')
    use_llm       = LaunchConfiguration('use_llm',       default='false')
    use_planner   = LaunchConfiguration('use_planner',   default='false')
    use_vision    = LaunchConfiguration('use_vision',    default='false')
    use_tts       = LaunchConfiguration('use_tts',       default='false')
    use_rtabmap   = LaunchConfiguration('use_rtabmap',   default='false')
    use_explore   = LaunchConfiguration('use_explore',   default='false')
    use_rviz      = LaunchConfiguration('use_rviz',      default='true')
    roomba_port   = LaunchConfiguration('roomba_port',   default='/dev/roomba')
    arm_port      = LaunchConfiguration('arm_port',      default='/dev/arm')
    imu_port      = LaunchConfiguration('imu_port',      default='/dev/imu')
    # 'mapping' (default, lifelong pose-graph growth) or 'localization'
    # (relocalize against a fixed previously-saved map — much more stable
    # position once a good map already exists, see save_map.sh). Requires
    # slam_map_file when 'localization'.
    slam_mode     = LaunchConfiguration('slam_mode',     default='mapping')
    slam_map_file = LaunchConfiguration('slam_map_file', default='')
    use_keepout   = LaunchConfiguration('use_keepout',   default='false')

    # ── SLAMTEC C1 LIDAR ─────────────────────────────────────────
    # Provided by ros-sllidar-c1.service (systemd, always running on
    # /dev/sllidar = same physical port as /dev/lidar). Do NOT start
    # another sllidar_node here — it would fail to open the serial
    # port (SL_RESULT_OPERATION_TIMEOUT) since it's already in use.

    # ── Roomba driver ─────────────────────────────────────────────
    roomba_node = Node(
        package='home_robot',
        executable='roomba_driver.py',
        name='roomba_driver',
        parameters=[{'port': roomba_port}],
        output='screen',
    )

    # odom -> base_link is published dynamically by ekf_node (below, only
    # when use_slam:=true), which fuses roomba_node's wheel velocity with
    # the IMU's absolute yaw — roomba_node itself no longer broadcasts this
    # TF. When use_rtabmap:=true instead, rgbd_odometry_node provides it.
    # Do NOT add a static fallback here — a static odom->base_link identity
    # TF alongside the dynamic one makes slam_toolbox think the robot never
    # moves, so the map never grows past the first scan.

    # ── IMU (ESP32 + MPU9250) ───────────────────────────────────────
    imu_node = Node(
        package='home_robot',
        executable='imu_node.py',
        name='imu_node',
        parameters=[{'port': imu_port}],
        output='screen',
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

    # Fuses roomba_node's wheel velocity (vx) with the IMU's absolute yaw
    # (config/ekf.yaml) — only meaningful alongside slam_toolbox's wheel-
    # odometry-based SLAM, not rtabmap's vision-based odometry, hence the
    # use_slam condition (same as slam_toolbox_node below).
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[PathJoinSubstitution([pkg, 'config', 'ekf.yaml'])],
        condition=IfCondition(use_slam),
    )

    # ── Static TF: base_link → laser ──────────────────────────────
    # base_link = midpoint of the wheel axle (the point the odometry
    # math in roomba_driver.py is computed about), at ground height.
    # All values measured by the user:
    #   - x: 0 — lidar remounted directly over the wheel axle (moved
    #     2026-06-17, was 120mm forward).
    #   - y: 0 — lidar is centered left/right.
    #   - z: lidar is 22mm above the wheel axle (remeasured 2026-06-17
    #     after the remount, was 26mm).
    #   - roll/pitch: 0 — lidar mount confirmed level with a spirit level.
    # Previously-noted "blocked rear FOV" turned out NOT to be a fixed
    # mechanical blockage — two consecutive scans taken with the robot
    # stationary showed almost entirely different missing-return angles
    # (16 of ~95 indices in common), which points to a lidar signal/
    # hardware issue rather than a mounting geometry problem. Moving the
    # mount to be centered over the axle is a mechanical improvement but
    # is not expected to fix that scatter — still pending a physical
    # inspection (lens/cable).
    tf_base_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_laser',
        # yaw=pi: lidar mount was physically flipped front/back during the
        # 2026-06-17 remount (centering it over the wheel axle) — without
        # this, a wall actually in front of the robot gets drawn behind it
        # in the map (confirmed live 2026-06-18: robot position/heading
        # moves correctly forward, but the scanned walls appear to recede
        # backward instead of approaching).
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.022',
                   '--roll', '0', '--pitch', '0', '--yaw', '3.14159265',
                   '--frame-id', 'base_link', '--child-frame-id', 'laser'],
    )

    # ── Static TF: base_link → camera_link ───────────────────────
    # Measured: x=0.10m forward, z=0.021m above the wheel axle (same
    # mount/bracket as laser, centered left/right, level, facing
    # straight ahead — so y=0, roll=pitch=yaw=0, same as
    # tf_base_laser). Matters for RTAB-Map RGBD odometry
    # (use_rtabmap:=true) and for object_detector's detections being
    # placed correctly in the map/costmap.
    tf_base_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_camera',
        arguments=['--x', '0.10', '--y', '0.0', '--z', '0.021',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link'],
    )

    # ── SLAM Toolbox ──────────────────────────────────────────────
    # Built directly (instead of including slam_toolbox's
    # online_async_launch.py) so slam_mode/slam_map_file can override
    # nav2_params.yaml's mode/map_file_name at launch time — lets the same
    # bringup switch between lifelong mapping (default) and relocalizing
    # against a fixed saved map (slam_mode:=localization
    # slam_map_file:=/path/to/map) without editing the yaml. Mirrors
    # slam_toolbox's own online_async_launch.py lifecycle handling
    # (configure then auto-activate) exactly.
    slam_toolbox_node = LifecycleNode(
        parameters=[
            PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml']),
            {
                'use_sim_time': False,
                'mode': slam_mode,
                'map_file_name': slam_map_file,
            },
        ],
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
        condition=IfCondition(use_slam),
    )

    slam_configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(use_slam),
    )

    slam_activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_toolbox_node,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_toolbox_node),
                    transition_id=Transition.TRANSITION_ACTIVATE,
                )),
            ],
        ),
        condition=IfCondition(use_slam),
    )

    # ── RTAB-Map (RGBD visual odometry + 3D SLAM) ────────────────
    # Alternative to slam_toolbox that doesn't need wheel odometry —
    # rgbd_odometry computes odom->base_link from the D435 RGB+depth
    # streams alone (using the existing static base_link->camera_link
    # TF above), bypassing the roomba get_sensors()/encoder issue.
    # rtabmap also subscribes to /scan for a more accurate 2D
    # occupancy grid (published on /map for Nav2's costmaps) and for
    # extra loop-closure constraints.
    #
    # Also fuses the IMU (visual-inertial odometry) — gives the vision
    # pipeline an orientation prior, which helps it recover faster in
    # feature-poor/dark scenes than RGBD alone.
    # wait_imu_to_init blocks the first odometry update until an IMU
    # message has been received, so initial orientation isn't guessed.
    #
    # NOTE: use_rtabmap and use_slam both provide odom->base_link and
    # map->odom TF + /map — do not enable both at once. Pass
    # use_slam:=false when using use_rtabmap:=true.
    rgbd_odometry_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'publish_tf': True,
            'approx_sync': True,
            'Reg/Force3DoF': 'true',
            'wait_imu_to_init': True,
        }],
        remappings=[
            ('rgb/image', '/camera/camera/color/image_raw'),
            ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('rgb/camera_info', '/camera/camera/color/camera_info'),
            ('odom', 'odom'),
            ('imu', 'imu/data'),
        ],
        condition=IfCondition(use_rtabmap),
    )

    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'subscribe_depth': True,
            'subscribe_rgb': True,
            'subscribe_scan': True,
            'approx_sync': True,
            'Reg/Force3DoF': 'true',
            'Grid/FromDepth': 'false',
            'Grid/Sensor': '0',
        }],
        remappings=[
            ('rgb/image', '/camera/camera/color/image_raw'),
            ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('rgb/camera_info', '/camera/camera/color/camera_info'),
            ('odom', 'odom'),
            ('scan', 'scan'),
        ],
        condition=IfCondition(use_rtabmap),
    )

    # ── Nav2 ──────────────────────────────────────────────────────
    nav2_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_pkg, '/launch/navigation_launch.py']),
        launch_arguments={
            'params_file':  PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml']),
            'use_sim_time': 'false',
        }.items(),
    )

    # ── Keepout zones (use_keepout, default false) ──────────────────
    # Standard Nav2 "keepout filter" pattern: a small map_server publishing
    # the binary mask + costmap_filter_info_server publishing how to
    # interpret it (base/multiplier/type, see nav2_params.yaml), both
    # lifecycle-managed by a dedicated lifecycle_manager since they're not
    # part of navigation_launch.py's own managed set. The KeepoutFilter
    # layer is already wired into both costmaps (nav2_params.yaml) and is
    # harmless when this is off — it just waits for filter info forever.
    # yaml_filename overridden here (not in nav2_params.yaml) so it
    # resolves to this package's installed share path, same pattern as
    # slam_toolbox's map_file_name override above.
    filter_mask_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='filter_mask_server',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml']),
            {'yaml_filename': PathJoinSubstitution([pkg, 'config', 'keepout_mask.yaml'])},
        ],
        remappings=[('map', 'keepout_filter_mask')],
        condition=IfCondition(use_keepout),
    )

    costmap_filter_info_server_node = Node(
        package='nav2_map_server',
        executable='costmap_filter_info_server',
        name='costmap_filter_info_server',
        output='screen',
        parameters=[PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml'])],
        condition=IfCondition(use_keepout),
    )

    keepout_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_filters',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['filter_mask_server', 'costmap_filter_info_server'],
        }],
        condition=IfCondition(use_keepout),
    )

    # ── Autonomous frontier exploration (m-explore-ros2 / explore_lite) ─
    # Drives Nav2 toward unexplored map edges on its own, so SLAM can build
    # the map without manual teleop. Needs Nav2 (always on) + an active map
    # (use_slam:=true or use_rtabmap:=true). return_to_init (config/explore.yaml)
    # sends it back to the start pose once no frontiers remain.
    explore_node = Node(
        package='explore_lite',
        executable='explore',
        name='explore_node',
        output='screen',
        parameters=[PathJoinSubstitution([pkg, 'config', 'explore.yaml'])],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        condition=IfCondition(use_explore),
    )

    # ── Intel RealSense D435 ──────────────────────────────────────
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'enable_color':    'true',
            'enable_depth':    'true',
            'enable_infra1':   'false',
            'enable_infra2':   'false',
            'align_depth.enable': 'true',
            'pointcloud.enable':  'true',
        }.items(),
        condition=IfCondition(use_camera),
    )

    # ── Object detector ───────────────────────────────────────────
    detector_node = Node(
        package='home_robot',
        executable='object_detector.py',
        name='object_detector',
        output='screen',
        condition=IfCondition(use_camera),
    )

    # ── Voice control ─────────────────────────────────────────────
    voice_node = Node(
        package='home_robot',
        executable='voice_control.py',
        name='voice_control',
        output='screen',
        condition=IfCondition(use_voice),
    )

    # ── Wake word detector (openWakeWord) ────────────────────────
    wake_word_node = Node(
        package='home_robot',
        executable='wake_word_node.py',
        name='wake_word_node',
        output='screen',
        condition=IfCondition(use_wake_word),
    )

    # ── Speech-to-text (faster-whisper, wake-word triggered) ─────
    stt_node = Node(
        package='home_robot',
        executable='stt_node.py',
        name='stt_node',
        output='screen',
        condition=IfCondition(use_stt),
    )

    # ── LLM bridge (Qwen3 tool calling, speech_text -> actions/speech_response) ─
    llm_bridge_node = Node(
        package='home_robot',
        executable='llm_bridge_node.py',
        name='llm_bridge_node',
        output='screen',
        condition=IfCondition(use_llm),
    )

    # ── Task planner (executes tidy/patrol via Nav2 + YOLO clutter check) ─
    # Consumes llm_bridge_node's tidy_command/patrol_command and narrates
    # progress on speech_response. Needs Nav2 (always on) + use_camera:=true
    # (default) for object_detector's detected_objects to be meaningful.
    planner_node = Node(
        package='home_robot',
        executable='task_planner_node.py',
        name='task_planner_node',
        output='screen',
        condition=IfCondition(use_planner),
    )

    # ── Vision Q&A (qwen3-vl:4b-instruct via ollama, "ask Max what he sees") ─
    # Answers llm_bridge_node's `look` tool from the latest camera frame
    # (vision/query -> vision/answer). Needs use_camera:=true for the
    # RealSense color stream to actually have a frame to look at.
    vision_node = Node(
        package='home_robot',
        executable='vision_node.py',
        name='vision_node',
        output='screen',
        condition=IfCondition(use_vision),
    )

    # ── Text-to-speech (edge-tts, speaks speech_response) ────────
    tts_node = Node(
        package='home_robot',
        executable='tts_node.py',
        name='tts_node',
        output='screen',
        condition=IfCondition(use_tts),
    )

    # ── Waveshare RoArm-M3 ──────────────────────────────────────────
    arm_node = Node(
        package='home_robot',
        executable='arm_driver.py',
        name='arm_driver',
        parameters=[{'port': arm_port}],
        output='screen',
        condition=IfCondition(use_arm),
    )

    # ── RViz2 ─────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg, 'config', 'robot.rviz'])],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_slam',      default_value='true'),
        DeclareLaunchArgument('use_camera',    default_value='true'),
        DeclareLaunchArgument('use_voice',     default_value='false'),
        DeclareLaunchArgument('use_arm',       default_value='false'),
        DeclareLaunchArgument('use_wake_word', default_value='false'),
        DeclareLaunchArgument('use_stt',       default_value='false'),
        DeclareLaunchArgument('use_llm',       default_value='false'),
        DeclareLaunchArgument('use_planner',   default_value='false'),
        DeclareLaunchArgument('use_vision',    default_value='false'),
        DeclareLaunchArgument('use_tts',       default_value='false'),
        DeclareLaunchArgument('use_rtabmap',   default_value='false'),
        DeclareLaunchArgument('use_explore',   default_value='false'),
        DeclareLaunchArgument('use_rviz',      default_value='true'),
        DeclareLaunchArgument('roomba_port',   default_value='/dev/roomba'),
        DeclareLaunchArgument('arm_port',      default_value='/dev/arm'),
        DeclareLaunchArgument('imu_port',      default_value='/dev/imu'),
        DeclareLaunchArgument('slam_mode',     default_value='mapping'),
        DeclareLaunchArgument('slam_map_file', default_value=''),
        DeclareLaunchArgument('use_keepout',   default_value='false'),

        tf_base_laser,
        tf_base_camera,
        tf_base_imu,
        roomba_node,
        imu_node,
        ekf_node,
        slam_toolbox_node,
        slam_configure_event,
        slam_activate_event,
        rgbd_odometry_node,
        rtabmap_node,
        nav2_node,
        filter_mask_server_node,
        costmap_filter_info_server_node,
        keepout_lifecycle_manager,
        explore_node,
        realsense_node,
        detector_node,
        voice_node,
        wake_word_node,
        stt_node,
        llm_bridge_node,
        planner_node,
        vision_node,
        tts_node,
        arm_node,
        rviz_node,
    ])
