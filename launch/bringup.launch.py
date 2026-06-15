"""Full robot bringup — LIDAR + SLAM + Nav2 + RealSense + Roomba driver + RViz2."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
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
    use_rviz      = LaunchConfiguration('use_rviz',      default='true')
    roomba_port   = LaunchConfiguration('roomba_port',   default='/dev/roomba')
    arm_port      = LaunchConfiguration('arm_port',      default='/dev/arm')

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

    # odom -> base_link is published dynamically by roomba_node below.
    # Do NOT add a static fallback here — a static odom->base_link
    # identity TF alongside the dynamic one makes slam_toolbox think
    # the robot never moves, so the map never grows past the first scan.

    # ── Static TF: base_link → laser ──────────────────────────────
    # base_link = midpoint of the wheel axle (the point the odometry
    # math in roomba_driver.py is computed about), at ground height.
    # All values measured by the user:
    #   - x: lidar sits 120mm forward of the wheel axle.
    #   - y: 0 — lidar is centered left/right.
    #   - z: lidar is 26mm above the wheel axle (corrected 2026-06-13,
    #     was 90mm).
    #   - roll/pitch: 0 — lidar mount confirmed level with a spirit level.
    # Known limitation (not fixed, by user's choice): this lidar mount
    # doesn't see a full 360 degrees — something blocks part of the
    # rear scan. May affect SLAM loop-closure/map completeness.
    tf_base_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_laser',
        arguments=['--x', '0.12', '--y', '0.0', '--z', '0.026',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'laser'],
    )

    # ── Static TF: base_link → camera_link ───────────────────────
    # Fully measured 2026-06-13: x=0.12m forward, z=0.021m above the
    # wheel axle (same mount/bracket as laser, centered left/right,
    # level, facing straight ahead — so y=0, roll=pitch=yaw=0, same as
    # tf_base_laser). Matters for RTAB-Map RGBD odometry
    # (use_rtabmap:=true) and for object_detector's detections being
    # placed correctly in the map/costmap.
    tf_base_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_camera',
        arguments=['--x', '0.12', '--y', '0.0', '--z', '0.021',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link'],
    )

    # ── SLAM Toolbox ──────────────────────────────────────────────
    slam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('slam_toolbox'), '/launch/online_async_launch.py'
        ]),
        launch_arguments={
            'slam_params_file': PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml']),
            'use_sim_time':     'false',
        }.items(),
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
        }],
        remappings=[
            ('rgb/image', '/camera/camera/color/image_raw'),
            ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('rgb/camera_info', '/camera/camera/color/camera_info'),
            ('odom', 'odom'),
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
        DeclareLaunchArgument('use_rviz',      default_value='true'),
        DeclareLaunchArgument('roomba_port',   default_value='/dev/roomba'),
        DeclareLaunchArgument('arm_port',      default_value='/dev/arm'),

        tf_base_laser,
        tf_base_camera,
        roomba_node,
        slam_node,
        rgbd_odometry_node,
        rtabmap_node,
        nav2_node,
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
