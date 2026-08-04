"""Full robot bringup — LIDAR + SLAM + Nav2 + RealSense + Roomba driver + RViz2."""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, GroupAction,
                            EmitEvent, RegisterEventHandler, TimerAction, ExecuteProcess)
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (AndSubstitution, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node, LifecycleNode, SetParameter
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
from lifecycle_msgs.msg import Transition
import os


def generate_launch_description():
    pkg = FindPackageShare('home_robot')
    nav2_pkg = FindPackageShare('nav2_bringup')

    use_slam      = LaunchConfiguration('use_slam',      default='true')
    use_localization = LaunchConfiguration('use_localization', default='false')
    localization_map = LaunchConfiguration('localization_map', default='')
    use_camera    = LaunchConfiguration('use_camera',    default='true')
    use_arm       = LaunchConfiguration('use_arm',       default='false')
    use_roomba    = LaunchConfiguration('use_roomba',    default='true')
    use_sim_time  = LaunchConfiguration('use_sim_time',  default='false')
    # Placeholder drop-off pose (arm_base frame) — no tray/bin measured yet.
    pick_drop_x          = LaunchConfiguration('pick_drop_x',          default='0.15')
    pick_drop_y          = LaunchConfiguration('pick_drop_y',          default='-0.15')
    pick_drop_z          = LaunchConfiguration('pick_drop_z',          default='0.10')
    pick_approach_height = LaunchConfiguration('pick_approach_height', default='0.10')
    use_wake_word = LaunchConfiguration('use_wake_word', default='false')
    use_stt       = LaunchConfiguration('use_stt',       default='false')
    use_doa             = LaunchConfiguration('use_doa',             default='true')
    doa_rotate_on_wake  = LaunchConfiguration('doa_rotate_on_wake',  default='true')
    doa_rotate_speed    = LaunchConfiguration('doa_rotate_speed',    default='0.6')
    doa_min_angle_deg   = LaunchConfiguration('doa_min_angle_deg',   default='20.0')
    doa_led_enabled     = LaunchConfiguration('doa_led_enabled',     default='true')
    use_person_follower = LaunchConfiguration('use_person_follower', default='true')
    use_llm       = LaunchConfiguration('use_llm',       default='false')
    use_dashboard = LaunchConfiguration('use_dashboard', default='false')
    llm_backend   = LaunchConfiguration('llm_backend',   default='gemini')
    # The `lemonade` backend is really "any OpenAI-compatible server". Exposed
    # because Lemonade 11.0.0 dropped every FLM/NPU model from its catalogue
    # (2026-07-16), so the NPU path now goes through FastFlowLM's own server on
    # :52625 instead — that is the default here. Note FLM's base path is `/v1`,
    # not Lemonade's `/api/v1`, and it spells the model `qwen3.5:4b`.
    # Override both together to point somewhere else.
    #
    # Model was qwen3vl-it:4b until 2026-07-30. An A/B over the real 15-tool
    # schema (24 Greek utterances x3) scored qwen3.5:4b at 100 % vs 90.5 %: the
    # VL model answered "Σταμάτησα." to a bare "σταμάτα" without ever calling
    # `stop`. qwen3.5 is multimodal as well, so use_vision is unaffected.
    llm_url       = LaunchConfiguration('llm_url',
                                        default='http://127.0.0.1:52625/v1')
    llm_model     = LaunchConfiguration('llm_model', default='qwen3.5:4b')
    vision_backend = LaunchConfiguration('vision_backend', default='gemini')
    use_planner   = LaunchConfiguration('use_planner',   default='false')
    use_vision    = LaunchConfiguration('use_vision',    default='false')
    use_memory    = LaunchConfiguration('use_memory',    default='false')
    use_tts       = LaunchConfiguration('use_tts',       default='false')
    use_rtabmap   = LaunchConfiguration('use_rtabmap',   default='false')
    use_explore   = LaunchConfiguration('use_explore',   default='false')
    use_obstacle_safety = LaunchConfiguration('use_obstacle_safety', default='true')
    obstacle_safety_distance = LaunchConfiguration('obstacle_safety_distance', default='0.5')
    use_joy       = LaunchConfiguration('use_joy',       default='false')
    joy_dev       = LaunchConfiguration('joy_dev',       default='/dev/input/js0')
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
    use_tracker         = LaunchConfiguration('use_tracker',         default='false')
    use_pose            = LaunchConfiguration('use_pose',            default='false')
    use_face_detection  = LaunchConfiguration('use_face_detection',  default='false')
    use_diarization     = LaunchConfiguration('use_diarization',     default='false')
    use_recovery        = LaunchConfiguration('use_recovery',        default='true')
    use_prediction      = LaunchConfiguration('use_prediction',      default='false')
    use_situational     = LaunchConfiguration('use_situational',     default='false')
    use_mission         = LaunchConfiguration('use_mission',         default='true')
    use_semantic_costmap = LaunchConfiguration('use_semantic_costmap', default='false')
    use_object_memory   = LaunchConfiguration('use_object_memory',   default='false')
    # RTAB-Map database path — placed alongside slam_toolbox maps so all
    # mapping artefacts live in one directory. Pass delete_db_on_start:=true
    # to recover from a corrupted database (see scripts/reset_rtabmap.sh).
    rtabmap_db    = LaunchConfiguration('rtabmap_db',
                        default=os.path.expanduser('~/robot_ws/maps/rtabmap.db'))
    delete_db     = LaunchConfiguration('delete_db_on_start', default='false')

    # ── SLAMTEC C1 LIDAR ─────────────────────────────────────────
    # Provided by ros-sllidar-c1.service (systemd, always running on
    # /dev/sllidar = same physical port as /dev/lidar). Do NOT start
    # another sllidar_node here — it would fail to open the serial
    # port (SL_RESULT_OPERATION_TIMEOUT) since it's already in use.

    # ── Roomba driver ─────────────────────────────────────────────
    # Listens on cmd_vel_safe, not cmd_vel directly — obstacle_safety_node
    # (below) is the gatekeeper between whatever publishes cmd_vel
    # (llm_bridge_node.py, teleop, Nav2's controller) and the wheels, so a
    # YOLO-detected obstacle right ahead can override any of those sources.
    # If use_obstacle_safety:=false, nothing relays cmd_vel -> cmd_vel_safe
    # and the robot won't move — only turn this off together with a manual
    # remap back to cmd_vel for direct testing.
    roomba_node = Node(
        package='home_robot',
        executable='roomba_driver.py',
        name='roomba_driver',
        parameters=[{'port': roomba_port}],
        remappings=[('cmd_vel', 'cmd_vel_safe')],
        output='screen',
        condition=IfCondition(use_roomba),
    )

    # ── YOLO/D435 forward obstacle safety stop ───────────────────
    # Independent of Nav2's own costmap-based avoidance (which only
    # applies to goal-directed navigation) — this catches raw cmd_vel
    # from voice control/teleop that never goes through Nav2's controller.
    obstacle_safety_node = Node(
        package='home_robot',
        executable='obstacle_safety_node.py',
        name='obstacle_safety_node',
        parameters=[{'safety_distance': obstacle_safety_distance}],
        output='screen',
        condition=IfCondition(use_obstacle_safety),
    )

    # ‼️ Where "raw" velocity publishers (llm_bridge's `move` tool, the mission
    # executor's final approach, doa_node turning toward a speaker) actually
    # land. This has to follow use_obstacle_safety, and hardcoding either end
    # breaks the other configuration:
    #
    #   safety ON  -> /cmd_vel, so obstacle_safety_node can zero the forward
    #                 component when the camera sees something ahead. Its
    #                 docstring names llm_bridge_node as the reason it exists;
    #                 publishing past it would silently remove that layer.
    #   safety OFF -> /cmd_vel_safe, straight at roomba_driver. Nothing relays
    #                 /cmd_vel then, and it has ZERO subscribers (measured live
    #                 2026-08-03), so those nodes were computing correct twists
    #                 and publishing them into nothing — spoken movement
    #                 commands did not move the robot at all.
    #
    # localize.launch.py runs with safety OFF, because obstacle_safety_node's
    # own fail-safe blocks forward motion when object_detector is not
    # publishing, and localize has no perception by default. roomba_driver
    # still has the bumper and cliff stops either way.
    voice_cmd_vel = PythonExpression(
        ["'cmd_vel' if '", use_obstacle_safety, "' == 'true' else 'cmd_vel_safe'"])
    voice_cmd_vel_remap = [('cmd_vel', voice_cmd_vel)]

    # odom -> base_link is published dynamically by ekf_node (below). The
    # EKF runs UNCONDITIONALLY (no launch condition) — it is required in
    # every mode that needs odom: SLAM mapping AND AMCL localization
    # (use_localization:=true, where use_slam:=false). roomba_node itself no
    # longer broadcasts this TF. CAVEAT: when use_rtabmap:=true,
    # rgbd_odometry_node ALSO publishes odom->base_link — running both would
    # double-publish the same TF edge, so don't combine use_rtabmap with the
    # EKF path until one is gated off.
    # Do NOT add a static fallback here — a static odom->base_link identity
    # TF alongside the dynamic one makes slam_toolbox think the robot never
    # moves, so the map never grows past the first scan.

    # ── IMU (ESP32 + BNO085) ───────────────────────────────────────
    imu_node = Node(
        package='home_robot',
        executable='imu_node.py',
        name='imu_node',
        parameters=[{'port': imu_port}],
        output='screen',
    )

    # IMU mounting orientation, re-measured 2026-07-21 on the Roomba 879 from
    # the BNO085's own gravity-referenced reading while the robot sat level.
    # The board is now mounted RIGHT-SIDE UP (roll ~= -0.4 deg, pitch ~= +1.5
    # deg) -- on the old 692 it was UPSIDE-DOWN (roll -177.3), so the previous
    # roll=-3.0952 would have flipped base_link over and inverted the yaw sense,
    # breaking EKF/AMCL. The static TF roll/pitch is set equal to the measured
    # raw roll/pitch so the fused base_link reads level (the same rule that gave
    # 0.0 deg over 448 samples on the 692). Yaw stays 0 -- the game rotation
    # vector yaw is arbitrary each boot and AMCL absorbs it via map->odom.
    # Translation: sits on the centred sensor stack (x=0.0 over the wheel axle,
    # y=0, z=0.536 to match the camera). NOTE: ekf.yaml fuses ONLY yaw +
    # yaw-rate from this IMU (no linear acceleration), so the exact translation
    # has no effect on the fused output -- it is set purely so the TF tree
    # matches the physical layout.
    tf_base_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_imu',
        # roll=pi REQUIRED: BNO085 game-vector yaw is opposite the calibrated
        # wheel odom; without it the EKF fuses contradictory yaws and the pose
        # spins out on turns. 5ffa712 set roll~=0 from the board's raw gravity
        # reading, but that only says the case is level, not which way yaw turns.
        # VERIFIED 2026-07-22: turn in place, roll~=0 gave opposite IMU/wheel
        # signs; roll=pi makes EKF-yaw track wheel-yaw (both +100 for a left turn).
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.536',
                   '--roll', '3.14159265', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'imu_link'],
    )

    # Fuses roomba_node's wheel velocity (vx) with the IMU's absolute yaw
    # (config/ekf.yaml). Runs in ALL modes (no condition) — both SLAM and
    # AMCL localization consume this odom->base_link TF. (Earlier comment
    # claimed a use_slam condition; that condition does not exist and must
    # not be added, or AMCL localize mode loses its odom source.)
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[PathJoinSubstitution([pkg, 'config', 'ekf.yaml'])],
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
        # Remeasured 2026-07-21 for the Roomba 879 chassis: centred over the
        # wheel axle (x=0, y=0) on the sensor mast, 570mm above the axle. The
        # user measured from the axle; base_link sits on the floor, so
        # z = 0.570 + 0.036 (wheel radius) = 0.606.
        # yaw=pi RE-CONFIRMED by the user 2026-07-21 on the new mast: the
        # connector still faces back while the unit scans forward, so the laser
        # frame is still rotated 180 deg about Z to align with base_link.
        arguments=['--x', '0.0', '--y', '0.0', '--z', '0.606',
                   '--roll', '0', '--pitch', '0', '--yaw', '3.14159265',
                   '--frame-id', 'base_link', '--child-frame-id', 'laser'],
    )

    # ── Static TF: base_link → camera_link ───────────────────────
    # Remeasured 2026-07-21 for the Roomba 879 chassis: the D435 is centred
    # left/right, 140mm forward of the wheel axle and 500mm above it ->
    # z = 0.500 + 0.036 (wheel radius) = 0.536, since base_link is on the
    # floor. Faces straight ahead (roll=pitch=yaw=0). Matters for RTAB-Map
    # RGBD odometry (use_rtabmap:=true) and for object_detector's detections
    # being placed correctly in the map/costmap.
    # NOTE: the camera is NOT rotated like the lidar (lidar has yaw=pi because
    # its connector faces back); the D435 faces forward, so yaw=0 here.
    tf_base_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_camera',
        arguments=['--x', '0.14', '--y', '0.0', '--z', '0.536',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link'],
    )

    # ── Static TF: base_link → arm_base ───────────────────────────
    # Remeasured 2026-07-21 for the Roomba 879 chassis: the arm base sits
    # 120mm forward of the wheel axle, centred left/right, 150mm above the
    # axle -> z = 0.150 + 0.036 (wheel radius) = 0.186, since base_link is on
    # the floor. Faces straight forward (roll=pitch=yaw=0).
    # pick_place_node.py uses this TF to convert object
    # positions from object_detector.py (camera_color_optical_frame) into
    # the arm_base frame that arm_driver.py's T:104 cartesian command expects —
    # wrong values here mean wrong reach/grasp targets.
    tf_base_arm = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_arm',
        arguments=['--x', '0.12', '--y', '0.0', '--z', '0.186',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'arm_base'],
        condition=IfCondition(use_arm),
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
                'use_sim_time': use_sim_time,
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

    slam_configure_event = TimerAction(
        period=5.0,
        actions=[EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_toolbox_node),
                transition_id=Transition.TRANSITION_CONFIGURE,
            ),
        )],
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
            # /cloud_map (RViz "3D Map (RTAB-Map)" display) density —
            # defaults (decimation=4, voxel=0.05m, max_depth=4.0m) made the
            # published map look sparse/low-detail. cloud_decimation=1 keeps
            # every depth pixel per keyframe instead of every 4th; smaller
            # voxel + longer max_depth trade more RAM/bandwidth for a denser,
            # more recognizable point cloud of the house.
            'cloud_decimation': 1,
            'cloud_max_depth': 6.0,
            'cloud_voxel_size': 0.01,
            'cloud_output_voxelized': True,
            # Keep the database alongside slam_toolbox maps. Pass
            # delete_db_on_start:=true to recover from a crash-corrupted DB.
            'database_path': rtabmap_db,
            'delete_db_on_start': delete_db,
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

    # ── AMCL localization (use_localization, default false) ──────
    # Loads a saved pgm/yaml map via map_server and localizes using AMCL
    # particle filter + LiDAR scan matching + EKF odometry (wheel+IMU).
    # AMCL publishes map->odom TF; slam_toolbox must be OFF (use_slam:=false).
    # NOTE: do NOT run use_slam:=true and use_localization:=true together —
    # both publish map->odom TF and will conflict.
    _localization_map_resolved = localization_map
    localization_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_pkg, '/launch/localization_launch.py']),
        launch_arguments={
            'map': _localization_map_resolved,
            'params_file': PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml']),
            'use_sim_time': use_sim_time,
        }.items(),
        condition=IfCondition(use_localization),
    )

    room_markers_node = Node(
        package='home_robot',
        executable='room_markers_node.py',
        name='room_markers_node',
        output='screen',
        condition=IfCondition(use_localization),
    )

    # Restores the last saved AMCL pose on startup (polls until AMCL is
    # active, then publishes to /initialpose).  global_localizer auto-runs
    # only when no per-map saved pose file exists.
    pose_saver_node = Node(
        package='home_robot',
        executable='pose_saver_node.py',
        name='pose_saver_node',
        output='screen',
        parameters=[{'save_interval': 10.0, 'map_yaml': localization_map,
                     # Don't restore the last pose on boot — global_localizer
                     # self-localizes from a random start instead (the user
                     # wants to power on anywhere without a manual 2D Pose
                     # Estimate). pose_saver still saves the live pose.
                     'restore_pose': False}],
        condition=IfCondition(use_localization),
    )

    # FFT scan-matching global localizer: finds the robot on the map by
    # correlating the LiDAR scan + D435 depth virtual scan against the
    # occupancy map likelihood field.  Auto-runs only when pose_saver has
    # no saved pose file for this map (defer_to_saved_pose:=true); otherwise
    # pose_saver restores the last AMCL pose and the LiDAR stays aligned.
    # Call /localize_globally manually when the robot was moved to a new spot.
    # The laser→base_link frame conversion (_laser_to_base in the node) is
    # required for this yaw=π-mounted lidar — without it every auto-match
    # lands 180° flipped.
    global_localizer_node = Node(
        package='home_robot',
        executable='global_localizer_node.py',
        name='global_localizer',
        output='screen',
        parameters=[{
            'auto_localize': True,
            # Run on EVERY boot, not just when no saved pose exists — the user
            # wants the robot to self-localize from wherever it is powered on,
            # never relying on a stale saved pose (which forced a manual 2D
            # Pose Estimate whenever the robot had been moved).
            'defer_to_saved_pose': False,
            'map_yaml': localization_map,
            'depth_weight': 0.5,
        }],
        condition=IfCondition(use_localization),
    )

    # Last-resort fallback: if global_localizer_node fails AND no saved pose
    # appeared within 35s, scatter AMCL particles across the whole map.
    global_localization_init = TimerAction(
        period=35.0,
        actions=[
            ExecuteProcess(
                cmd=['bash', '-c',
                     '[ -f ~/.ros/last_amcl_pose.yaml ] && exit 0; '
                     'until ros2 service list 2>/dev/null | '
                     'grep -q reinitialize_global_localization; do sleep 1; done; '
                     'ros2 service call /reinitialize_global_localization '
                     'std_srvs/srv/Empty "{}"'],
                output='screen',
            )
        ],
        condition=IfCondition(use_localization),
    )

    # ── Nav2 ──────────────────────────────────────────────────────
    # bond_timeout:=0.0 disables the lifecycle manager's bond watchdog: a
    # transient machine stall (>4s, seen under full-stack boot load and again
    # 2026-07-04) breaks the heartbeats and the manager tears the whole nav
    # chain down, leaving parts of it inactive — goals then silently go
    # nowhere. Crash recovery is handled by ensure_nav_active.sh below
    # instead. (Servers also set bond_heartbeat_period: 0.0 in
    # nav2_params.yaml so they don't create their side of the bond.)
    nav2_node = GroupAction([
        SetParameter('bond_timeout', 0.0),
        # Doorway-friendly recovery order (BackUp/DriveOnHeading before Spin)
        # — stock Jazzy tree otherwise; see the XML header for rationale.
        SetParameter('default_nav_to_pose_bt_xml', PathJoinSubstitution(
            [pkg, 'config', 'behavior_trees',
             'navigate_to_pose_w_replanning_and_recovery.xml'])),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([nav2_pkg, '/launch/navigation_launch.py']),
            launch_arguments={
                'params_file':  PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml']),
                'use_sim_time': use_sim_time,
                'autostart':    'true',
            }.items(),
        ),
    ])

    # Self-heal: 60s after launch, configure+activate any nav server that is
    # not active (see 2026-07-03 incident: behavior_server/velocity_smoother/
    # collision_monitor came up inactive under boot load). No-op when healthy.
    nav2_ensure_active = TimerAction(
        period=60.0,
        actions=[ExecuteProcess(
            cmd=['bash', PathJoinSubstitution([pkg, 'scripts', 'ensure_nav_active.sh'])],
            output='screen',
        )],
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
    # Wrapped in a non-forwarding GroupAction: rs_launch.py prints a warning —
    # with the full 80-entry list of supported parameters — for every launch
    # configuration it does not recognise, and an included launch file inherits
    # the parent's. So each of our own `use_*` args (use_tts, use_stt, use_llm,
    # use_slam, use_rviz, …) produced a screenful of noise that buried the real
    # errors in the log. They never reached the camera node (rs_launch only
    # passes its own declared parameters), so this is cosmetic upstream chatter;
    # scoped + forwarding=False just hands the include a clean scope. The
    # arguments below are still passed explicitly, and `condition` sits on the
    # group so `use_camera` is resolved in the parent scope.
    realsense_node = GroupAction(
        actions=[
            IncludeLaunchDescription(
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
            )
        ],
        scoped=True,
        forwarding=False,
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

    # The camera is the one part that goes missing without saying so: when the
    # driver segfaults (~2.5% of bringups) everything downstream just reports
    # "nothing there". This restarts it and says so on /rosout.
    camera_watchdog_node = Node(
        package='home_robot',
        executable='camera_watchdog_node.py',
        name='camera_watchdog',
        output='screen',
        condition=IfCondition(use_camera),
    )

    # ── Pose estimation (YOLO11n-pose on NPU) ────────────────────
    pose_node = Node(
        package='home_robot',
        executable='pose_node.py',
        name='pose_node',
        output='screen',
        condition=IfCondition(use_pose),
    )

    # ── Fall monitoring (geometry over pose_node's skeletons) ─────
    # Costs nothing extra on the GPU — it reads pose_detections. Gated on the
    # same use_pose flag because without skeletons it can only sit silent, and
    # a silent safety monitor is indistinguishable from a safe house.
    fall_monitor_node = Node(
        package='home_robot',
        executable='fall_monitor_node.py',
        name='fall_monitor_node',
        output='screen',
        parameters=[{
            'hold_s': LaunchConfiguration('fall_hold_s', default='6.0'),
            'speak':  LaunchConfiguration('fall_speak', default='true'),
        }],
        condition=IfCondition(use_pose),
    )

    # ── Pointing gestures (geometry over pose_node's skeletons) ───
    # Free on the GPU like fall_monitor — it reads pose_detections and the
    # aligned depth image. Gated on use_pose for the same reason: without
    # skeletons it can only sit silent.
    # ‼️ auto_goal stays false. Pointing is far too easy to do by accident
    # (reaching for a cup traces the same arc) to wire straight into Nav2; a
    # confirmed gesture is published and drawn, and the dashboard button or the
    # `goto_pointed` voice tool decides whether to act on it.
    gesture_node = Node(
        package='home_robot',
        executable='gesture_node.py',
        name='gesture_node',
        output='screen',
        parameters=[{
            'auto_goal':     LaunchConfiguration('gesture_auto_goal', default='false'),
            'confirm_hits':  LaunchConfiguration('gesture_confirm_hits', default='5'),
            'max_distance':  LaunchConfiguration('gesture_max_distance', default='8.0'),
        }],
        condition=IfCondition(use_pose),
    )

    # ── Proactive observations (the robot speaks first) ───────────
    # Needs object_memory for the "where things normally are" baseline, so it
    # rides the same flag. Says nothing at all until it has learned a room over
    # several separate visits — silence on a fresh map is correct.
    proactive_observer_node = Node(
        package='home_robot',
        executable='proactive_observer_node.py',
        name='proactive_observer_node',
        output='screen',
        parameters=[{
            'enabled':        LaunchConfiguration('proactive', default='true'),
            'min_interval':   LaunchConfiguration('proactive_min_interval', default='180.0'),
            'max_per_hour':   LaunchConfiguration('proactive_max_per_hour', default='4'),
            'require_person': LaunchConfiguration('proactive_require_person', default='true'),
        }],
        condition=IfCondition(use_object_memory),
    )

    # ── Episodic memory (the time axis) ───────────────────────────
    # Cheap and independent of perception: it listens to speech, room changes
    # and observations. On by default because its whole value is having been
    # recording before you thought to ask.
    episodic_memory_node = Node(
        package='home_robot',
        executable='episodic_memory_node.py',
        name='episodic_memory_node',
        output='screen',
        parameters=[{
            'max_age_days': LaunchConfiguration('episodic_days', default='14'),
        }],
        condition=IfCondition(LaunchConfiguration('use_episodic', default='true')),
    )

    # ── Open-vocabulary detection (YOLO-World) ────────────────────
    # Rides use_perception because it shares the camera and the iGPU, but it is
    # IDLE until something asks for a word COCO does not have, and returns to
    # idle after idle_timeout. A second detector running continuously on this
    # iGPU is precisely the load problem this project keeps hitting.
    open_vocab_detector = Node(
        package='home_robot',
        executable='open_vocab_detector.py',
        name='open_vocab_detector',
        output='screen',
        parameters=[{
            'idle_timeout': LaunchConfiguration('vocab_idle_timeout', default='90.0'),
            'conf':         LaunchConfiguration('vocab_conf', default='0.05'),
        }],
        condition=IfCondition(LaunchConfiguration('use_open_vocab', default='false')),
    )

    # ── Sound events (YAMNet over the shared mic topic) ───────────
    # Needs the wake word node: that is what owns the ALSA stream and
    # republishes /mic/audio. ~5 ms per window, so it is cheap enough to run
    # continuously — but only when there is actually audio to read.
    sound_event_node = Node(
        package='home_robot',
        executable='sound_event_node.py',
        name='sound_event_node',
        output='screen',
        parameters=[{
            'threshold': LaunchConfiguration('sound_threshold', default='0.35'),
            'speak':     LaunchConfiguration('sound_speak', default='true'),
        }],
        condition=IfCondition(LaunchConfiguration('use_sound_events', default='false')),
    )

    # ── Face identity (SFace over YuNet's landmarks) ──────────────
    # Rides use_face_detection: without those five landmarks per face there is
    # nothing to align, and an unaligned crop costs most of the accuracy.
    face_identity_node = Node(
        package='home_robot',
        executable='face_identity_node.py',
        name='face_identity_node',
        output='screen',
        parameters=[{
            'threshold': LaunchConfiguration('face_threshold', default='0.363'),
        }],
        condition=IfCondition(use_face_detection),
    )

    # ── Self-diagnosis ────────────────────────────────────────────
    # Cheap and always useful: it measures topic RATES and matches /rosout
    # against the signatures of faults this robot has actually had.
    self_diagnosis_node = Node(
        package='home_robot',
        executable='self_diagnosis_node.py',
        name='self_diagnosis_node',
        output='screen',
        parameters=[{
            'speak': LaunchConfiguration('diagnose_speak', default='true'),
        }],
        condition=IfCondition(LaunchConfiguration('use_diagnosis', default='true')),
    )

    # ── Face detection (YuNet on NPU) ─────────────────────────────
    face_detection_node = Node(
        package='home_robot',
        executable='face_detection_node.py',
        name='face_detection_node',
        output='screen',
        condition=IfCondition(use_face_detection),
    )

    # ── SORT multi-object tracker ─────────────────────────────────
    # Wraps detected_objects with persistent integer track IDs and estimated
    # pixel-space velocities. Publishes tracked_objects (same JSON format +
    # track_id/track_age/vel_x/vel_y). Needs use_camera:=true.
    tracker_node = Node(
        package='home_robot',
        executable='tracker_node.py',
        name='tracker_node',
        output='screen',
        condition=IfCondition(use_tracker),
    )

    # ── Speaker diarization (resemblyzer d-vectors + webrtcvad) ───
    # Continuously listens on the microphone and publishes who is speaking
    # on current_speaker. Enroll a voice by publishing the name to
    # diarization/register then speaking; profiles persist in
    # ~/.robot_speakers.npz. Runs independently of the STT/wake-word path.
    diarization_node_action = Node(
        package='home_robot',
        executable='diarization_node.py',
        name='diarization_node',
        output='screen',
        condition=IfCondition(use_diarization),
    )

    # ── Physical stuck detection + Nav2 backup/spin recovery ─────
    # Default ON — requires no extra hardware. Publishes recovery/status
    # and speech_response. Detects stuck via cmd_vel_safe vs /odom.
    recovery_node = Node(
        package='home_robot',
        executable='recovery_manager_node.py',
        name='recovery_manager_node',
        output='screen',
        condition=IfCondition(use_recovery),
    )

    # ── Moving-obstacle trajectory prediction ─────────────────────
    # Projects tracked_objects with velocity > min_speed ahead by `horizon`
    # seconds and publishes /predicted_obstacles PointCloud2 for Nav2's
    # local_costmap. Needs use_tracker:=true + use_camera:=true.
    prediction_node = Node(
        package='home_robot',
        executable='obstacle_prediction_node.py',
        name='obstacle_prediction_node',
        output='screen',
        condition=IfCondition(use_prediction),
    )

    # ── Situational awareness (room/objects/system → situation_context) ──
    # Publishes a JSON context snapshot at 1Hz; llm_bridge_node injects it
    # as a system message before every LLM turn so Max always knows his
    # location, nearby objects, and system health without tool calls.
    situational_node = Node(
        package='home_robot',
        executable='situational_awareness_node.py',
        name='situational_awareness_node',
        output='screen',
        condition=IfCondition(use_situational),
    )

    # ── Mission executor — patrol, find, dock, check_rooms ───────
    # Triggered via mission/start topic. Default ON — no extra hardware needed.
    mission_node = Node(
        package='home_robot',
        executable='mission_executor_node.py',
        name='mission_executor_node',
        output='screen',
        # fetch delivery: 'start_pose' (return to where the command was given) or
        # 'follow' (home in on the user's current position via person detections).
        parameters=[{'delivery_mode': LaunchConfiguration('delivery_mode', default='start_pose')}],
        remappings=voice_cmd_vel_remap,
        condition=IfCondition(use_mission),
    )

    # ── Semantic costmap — inflated person/animal zones ───────────
    # Publishes /semantic_obstacles with per-class radius expansion.
    # Needs use_camera:=true; enable when people are expected nearby.
    semantic_costmap_node_action = Node(
        package='home_robot',
        executable='semantic_costmap_node.py',
        name='semantic_costmap_node',
        output='screen',
        condition=IfCondition(use_semantic_costmap),
    )

    # ── Semantic object memory (remembers where things are, map frame) ──
    # Folds detected_objects into a persistent per-label spatial map, feeds
    # RAG recall ("πού είναι το X;"). Needs use_camera:=true for detections.
    object_memory_node = Node(
        package='home_robot',
        executable='object_memory_node.py',
        name='object_memory_node',
        output='screen',
        condition=IfCondition(use_object_memory),
    )

    # ── Wake word detector (openWakeWord) ────────────────────────
    wake_word_node = Node(
        package='home_robot',
        executable='wake_word_node.py',
        name='wake_word_node',
        output='screen',
        # 2026-07-25: wake phrase changed "Ρομπότ Μαξ" -> "Έι ρομπότ"/"Hey robot",
        # which needed a fresh model (training/wake_word_hey_robot/). Held-out
        # synthetic eval: positives min 0.609 / mean 0.991, negatives max 0.382
        # (next-highest 0.009), pure noise 0.000. Threshold back to 0.50: unlike
        # the old "max" model this one has NO real XVF3800 recordings in it yet,
        # and synthetic-only models score lower on real mic audio — 0.60 risks
        # misses. Raise it if real-world talking false-triggers, or add real
        # positives with training/wake_word_hey_robot/record_real.py.
        # Listen on the XVF3800 ASR-beam channel (ch4) — the beamformed/denoised
        # signal the model's augmentation was designed around.
        # Without these the node falls back to defaults (default device, ch0/3ch),
        # which is NOT what the model was trained on. Verified live 2026-07-03:
        # real "Μαξ" scores 1.00 on ch4 at threshold 0.60, 6/6 detections.
        # 2026-07-23: re-measured all six channels with the mic FREE (the node
        # holds it exclusively, so any reading taken while it runs is worthless):
        # ch0 0.0067, ch1 0.0067, ch2 0.0101, ch3 0.0128, ch4 0.0113, ch5 0.0129.
        # ch4 carries normal signal — the "ch4 went silent" note from 07-07 no
        # longer holds. What fixed "δεν ακούει" that day was restarting the node
        # (it caches config at init, so a restart is indistinguishable from a
        # channel change) — not ch0.
        #
        # Do NOT switch to ch0: the model is trained on the beamformed/denoised
        # ASR beam, so on the raw mic it false-fires constantly — measured 9
        # triggers in 75 s at scores 0.95-1.00. Each false trigger publishes
        # tts/stop, so the robot also interrupts its own speech every few
        # seconds and appears to never answer. On ch4: 0 false triggers in 30 s.
        # Left as a launch arg only for experiments; the default is the one to use.
        parameters=[{
            'threshold': 0.50,
            'device_name': 'XVF3800',
            'mic_channels': 6,
            'mic_channel': LaunchConfiguration('mic_channel', default='4'),
            # ...but transcribe from ch0. 2026-07-26: captured all six channels
            # during real speech and ch0 measured 39.5 dB SNR / peak 0.470 vs
            # ch4's 21.2 dB / 0.077 — ch4 is a raw capsule, not the ASR beam the
            # comment above assumed. Feeding Whisper the raw capsule is what
            # produced plausible-but-wrong Greek. ch0's AGC is exactly why it
            # cannot drive wake detection (it lifts the noise floor in silence),
            # so the two are split rather than switched.
            'stt_channel': LaunchConfiguration('stt_channel', default='0'),
            # Barge-in: saying "Έι ρομπότ" while the robot is talking aborts the
            # TTS and starts a new turn.
            #
            # Was false from 2026-07-24, when the robot's own replies self-fired
            # the detector at ~1.00 and it interrupted every answer. The cause is
            # that the ch4 AEC does not remove self-speech, so a reply and a real
            # interruption arrive as the same kind of audio and NO threshold can
            # separate them.
            #
            # 2026-08-03: default true. Not because the threshold got better, but
            # because the robot now checks WHAT it is about to say — a reply
            # containing the wake word disables barge-in for its duration
            # (utterance_is_risky in voice_gate.py), which is exactly the case
            # that used to fire. Ordinary replies, the large majority, keep it.
            # ‼️ HW-untested. If it still interrupts itself, pass
            # allow_barge_in:=false and say so — the next lever is routing TTS
            # through the ReSpeaker so the AEC finally has a reference signal
            # (tts_node's device_name parameter).
            'allow_barge_in': LaunchConfiguration('allow_barge_in', default='true'),
            # Wake acknowledgement ON again 2026-07-25: without any cue you
            # cannot tell whether the robot heard you, which the user asked for
            # explicitly ("like Hey Siri"). It was switched off because the old
            # cue — one 880 Hz sine, 0.6 amplitude, 350 ms, no envelope — was
            # grating; the sound is now a soft two-note chime (~0.2 s, a third
            # of the volume, fades in/out). Pass beep_on_wake:=false for silent.
            'beep_on_wake': LaunchConfiguration('beep_on_wake', default='true'),
        }],
        condition=IfCondition(use_wake_word),
    )

    # ── Speech-to-text (faster-whisper, wake-word triggered) ─────
    # energy_thresh is pinned instead of auto-calibrated. The boot calibration
    # measures ambient RMS for a moment and scales it, and on 2026-07-26 it
    # produced an unusable value BOTH times it ran: 0.065 (above the user's
    # speech, so nothing was ever recorded) and 0.012 (inside the room noise,
    # so every fan spike started a recording that closed 1.5 s later on
    # silence). One noisy moment at startup poisons the whole session. Measured
    # that evening: room noise 0.008-0.015, speech 0.042-0.095 → 0.030 sits
    # cleanly between them.
    stt_node = Node(
        package='home_robot',
        executable='stt_node.py',
        name='stt_node',
        parameters=[{'energy_thresh': 0.030, 'calibrate_on_start': False}],
        output='screen',
        condition=IfCondition(use_stt),
    )

    # ── Direction of Arrival (XVF3800 hardware DoA) ───────────────
    doa_node = Node(
        package='home_robot',
        executable='doa_node.py',
        name='doa_node',
        output='screen',
        condition=IfCondition(use_doa),
        remappings=voice_cmd_vel_remap,
        parameters=[{
            'rotate_on_wake': doa_rotate_on_wake,
            'rotate_speed':   doa_rotate_speed,
            'min_angle_deg':  doa_min_angle_deg,
            'led_enabled':    doa_led_enabled,
        }],
    )

    # ── Person follower (D435 depth + DoA) ────────────────────────
    person_follower_node = Node(
        package='home_robot',
        executable='person_follower_node.py',
        name='person_follower_node',
        output='screen',
        condition=IfCondition(use_person_follower),
    )

    # ── LLM bridge (Qwen3 tool calling, speech_text -> actions/speech_response) ─
    llm_bridge_node = Node(
        package='home_robot',
        executable='llm_bridge_node.py',
        name='llm_bridge_node',
        parameters=[{'backend': llm_backend, 'memory_enabled': use_memory,
                     'lemonade_url': llm_url, 'lemonade_model': llm_model}],
        output='screen',
        remappings=voice_cmd_vel_remap,
        condition=IfCondition(use_llm),
    )

    # ── RAG long-term memory (ChromaDB + embed-gemma-300m-FLM on NPU) ─────
    # Stores facts via the llm_bridge_node 'remember' tool (memory/store) and
    # answers similarity queries (memory/query -> memory/answer) injected
    # into every LLM call as extra context. Needs the embed-gemma-300m-FLM
    # model loaded in Lemonade (auto-loads on first request).
    memory_node = Node(
        package='home_robot',
        executable='rag_memory_node.py',
        name='rag_memory_node',
        output='screen',
        condition=IfCondition(use_memory),
    )

    explore_manager_node = Node(
        package='home_robot',
        executable='explore_manager_node.py',
        name='explore_manager',
        output='screen',
    )

    # ── Web dashboard (http://<host>:8080) ────────────────────────────────
    # The whole robot in a browser tab: map + click-to-navigate, camera,
    # arm sliders, vacuum telemetry, an LLM chat wired to speech_text, and the
    # real RViz/MoveIt/Gazebo streamed from headless VNC displays.
    # Read-only towards the ROS graph except for the controls it exposes, so it
    # is safe alongside everything else. `robot max` turns this on.
    dashboard_node = Node(
        package='home_robot',
        executable='web_dashboard_node.py',
        name='web_dashboard',
        output='screen',
        condition=IfCondition(use_dashboard),
    )

    # ── Task planner (executes tidy/patrol via Nav2 + YOLO clutter check) ─
    # Consumes llm_bridge_node's tidy_command/patrol_command and narrates
    # progress on speech_response. Needs Nav2 (always on) + use_camera:=true
    # (default) for object_detector's detected_objects to be meaningful.
    planner_node = Node(
        package='home_robot',
        executable='task_planner_node.py',
        name='task_planner_node',
        parameters=[{'use_arm': use_arm}],
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
        parameters=[{'backend': vision_backend}],
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
        parameters=[{
            'port': arm_port,
            # Usable envelope — MEASURED by hand 2026-07-31 with the servo
            # torque off, not taken from the wiki. ‼️ These must stay identical
            # to config/arm_joy_ps5.yaml's limit_*: arm_joy stops the jog
            # integrator, this node is what refuses the command. See that file
            # for the raw extremes and why the shoulder is capped at 1.57.
            'limit_base':     [-3.015, 0.016],
            'limit_shoulder': [-0.169, 1.570],
            'limit_elbow':    [0.141, 3.079],
            'limit_wrist':    [-1.143, 1.407],
            'limit_roll':     [-1.165, 1.372],
        }],
        output='screen',
        condition=IfCondition(use_arm),
    )

    # ── Pick-place (object_detector.py + arm_driver.py bridge) ──────
    # UNTESTED — see pick_place_node.py's module docstring. Gated on the
    # same use_arm flag as arm_node/tf_base_arm since it's meaningless
    # without either.
    pick_place_node = Node(
        package='home_robot',
        executable='pick_place_node.py',
        name='pick_place_node',
        parameters=[{
            'drop_x': pick_drop_x,
            'drop_y': pick_drop_y,
            'drop_z': pick_drop_z,
            'approach_height': pick_approach_height,
        }],
        output='screen',
        condition=IfCondition(use_arm),
    )

    # Forward RViz "2D Goal Pose" (/goal_pose) to the Nav2 action. Only useful
    # once the nav stack is up, so gate it on localization (the mode that runs
    # the planner/controller) rather than the web dashboard.
    goal_pose_bridge_node = Node(
        package='home_robot',
        executable='goal_pose_bridge.py',
        name='goal_pose_bridge',
        output='screen',
        condition=IfCondition(use_localization),
    )

    # ── PS5 DualSense teleop (manual mapping drive) ──────────────
    # joy_node reads the raw /dev/input/jsN device (already paired over
    # Bluetooth, see config/teleop_twist_joy_ps5.yaml header for the
    # confirmed axis/button layout); teleop_twist_joy_node turns it into
    # cmd_vel, which flows through obstacle_safety_node like any other
    # cmd_vel source (R1 is the required dead-man's switch).
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'dev': joy_dev}],
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
    # Sticky soft e-stop: Circle latches the robot stopped (roomba_driver zeros
    # the wheels + ignores cmd_vel), Share+Options together resets.
    joystick_estop_node = Node(
        package='home_robot',
        executable='joystick_estop_node.py',
        name='joystick_estop',
        output='screen',
        condition=IfCondition(use_joy),
    )
    # Right stick jogs the arm (base/shoulder), R1/R2 work the gripper.
    # Needs BOTH the controller and the arm, so gate on both flags — on its own
    # it would just publish arm/joint_cmd into the void.
    arm_joy_node = Node(
        package='home_robot',
        executable='arm_joy_node.py',
        name='arm_joy',
        parameters=[PathJoinSubstitution([pkg, 'config', 'arm_joy_ps5.yaml'])],
        output='screen',
        condition=IfCondition(AndSubstitution(use_joy, use_arm)),
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
        DeclareLaunchArgument('use_localization', default_value='false'),
        DeclareLaunchArgument('localization_map',
            default_value=PathJoinSubstitution([pkg, 'maps', 'malou2.yaml'])),
        DeclareLaunchArgument('use_camera',    default_value='true'),
        DeclareLaunchArgument('use_arm',       default_value='false'),
        DeclareLaunchArgument('pick_drop_x',          default_value='0.15'),
        DeclareLaunchArgument('pick_drop_y',          default_value='-0.15'),
        DeclareLaunchArgument('pick_drop_z',          default_value='0.10'),
        DeclareLaunchArgument('pick_approach_height', default_value='0.10'),
        DeclareLaunchArgument('use_wake_word', default_value='false'),
        DeclareLaunchArgument('use_stt',       default_value='false'),
        # ‼️ The DECLARED default is the one that wins — the LaunchConfiguration
        # default above is only a fallback for when this line does not exist.
        # Leaving this at 'false' would have silently kept barge-in off, the
        # same trap that kept use_doa dark for weeks.
        DeclareLaunchArgument('allow_barge_in', default_value='true'),
        DeclareLaunchArgument('beep_on_wake',  default_value='true'),
        DeclareLaunchArgument('mic_channel',   default_value='4'),
        DeclareLaunchArgument('stt_channel',   default_value='0'),
        DeclareLaunchArgument('delivery_mode',  default_value='start_pose'),
        DeclareLaunchArgument('use_doa',            default_value='false'),
        DeclareLaunchArgument('doa_rotate_on_wake', default_value='true'),
        DeclareLaunchArgument('doa_rotate_speed',   default_value='0.6'),
        DeclareLaunchArgument('doa_min_angle_deg',  default_value='20.0'),
        DeclareLaunchArgument('doa_led_enabled',    default_value='true'),
        DeclareLaunchArgument('use_person_follower', default_value='false'),
        DeclareLaunchArgument('use_llm',       default_value='false'),
        DeclareLaunchArgument('use_dashboard', default_value='false'),
        DeclareLaunchArgument('llm_backend',   default_value='gemini'),
        DeclareLaunchArgument('llm_url',
            default_value='http://127.0.0.1:52625/v1'),
        DeclareLaunchArgument('llm_model',     default_value='qwen3.5:4b'),
        DeclareLaunchArgument('vision_backend', default_value='lemonade'),
        DeclareLaunchArgument('use_planner',   default_value='false'),
        DeclareLaunchArgument('use_vision',    default_value='false'),
        DeclareLaunchArgument('use_memory',    default_value='false'),
        DeclareLaunchArgument('use_tts',       default_value='false'),
        DeclareLaunchArgument('use_rtabmap',   default_value='false'),
        DeclareLaunchArgument('use_explore',   default_value='false'),
        DeclareLaunchArgument('use_joy',       default_value='false'),
        DeclareLaunchArgument('joy_dev',       default_value='/dev/input/js0'),
        DeclareLaunchArgument('use_rviz',      default_value='true'),
        DeclareLaunchArgument('roomba_port',   default_value='/dev/roomba'),
        DeclareLaunchArgument('arm_port',      default_value='/dev/arm'),
        DeclareLaunchArgument('imu_port',      default_value='/dev/imu'),
        DeclareLaunchArgument('slam_mode',     default_value='mapping'),
        DeclareLaunchArgument('slam_map_file', default_value=''),
        DeclareLaunchArgument('use_keepout',         default_value='false'),
        DeclareLaunchArgument('use_tracker',         default_value='false'),
        DeclareLaunchArgument('use_pose',            default_value='false'),
        DeclareLaunchArgument('use_face_detection',  default_value='false'),
        DeclareLaunchArgument('use_diarization',     default_value='false'),
        DeclareLaunchArgument('use_recovery',           default_value='true'),
        DeclareLaunchArgument('use_prediction',         default_value='false'),
        DeclareLaunchArgument('use_situational',        default_value='false'),
        DeclareLaunchArgument('use_mission',            default_value='true'),
        DeclareLaunchArgument('use_semantic_costmap',   default_value='false'),
        DeclareLaunchArgument('use_object_memory',      default_value='false'),
        DeclareLaunchArgument('use_episodic',           default_value='true'),
        DeclareLaunchArgument('use_open_vocab',         default_value='false'),
        DeclareLaunchArgument('use_sound_events',       default_value='false'),
        DeclareLaunchArgument('use_diagnosis',          default_value='true'),
        DeclareLaunchArgument('gesture_auto_goal',      default_value='false'),
        DeclareLaunchArgument('proactive',              default_value='true'),
        DeclareLaunchArgument('use_sim_time',            default_value='false'),
        DeclareLaunchArgument('use_roomba',              default_value='true'),
        DeclareLaunchArgument('rtabmap_db',
            default_value=os.path.expanduser('~/robot_ws/maps/rtabmap.db')),
        DeclareLaunchArgument('delete_db_on_start',  default_value='false'),
        DeclareLaunchArgument('use_obstacle_safety', default_value='true'),
        DeclareLaunchArgument('obstacle_safety_distance', default_value='0.5'),

        tf_base_laser,
        tf_base_camera,
        tf_base_arm,
        tf_base_imu,
        roomba_node,
        obstacle_safety_node,
        imu_node,
        ekf_node,
        slam_toolbox_node,
        slam_configure_event,
        slam_activate_event,
        rgbd_odometry_node,
        rtabmap_node,
        localization_node,
        room_markers_node,
        pose_saver_node,
        global_localizer_node,
        global_localization_init,
        nav2_node,
        nav2_ensure_active,
        filter_mask_server_node,
        costmap_filter_info_server_node,
        keepout_lifecycle_manager,
        explore_node,
        realsense_node,
        detector_node,
        camera_watchdog_node,
        pose_node,
        fall_monitor_node,
        face_detection_node,
        gesture_node,
        proactive_observer_node,
        episodic_memory_node,
        open_vocab_detector,
        sound_event_node,
        face_identity_node,
        self_diagnosis_node,
        tracker_node,
        diarization_node_action,
        recovery_node,
        prediction_node,
        situational_node,
        object_memory_node,
        mission_node,
        semantic_costmap_node_action,
        wake_word_node,
        stt_node,
        doa_node,
        person_follower_node,
        llm_bridge_node,
        memory_node,
        explore_manager_node,
        dashboard_node,
        planner_node,
        vision_node,
        tts_node,
        arm_node,
        pick_place_node,
        goal_pose_bridge_node,
        joy_node,
        teleop_twist_joy_node,
        joystick_estop_node,
        arm_joy_node,
        rviz_node,
    ])
