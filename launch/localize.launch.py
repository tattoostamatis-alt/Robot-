"""One command to open a saved map and localize on it — no manual 2D Pose Estimate.

Loads a saved map (default 'malou2') and brings up AMCL + pose_saver (restores
the last pose when available) + global_localizer (FFT scan-match only when
no saved pose) + RViz, by including bringup.launch.py with the heavy
AI/voice/camera stack switched off.

If the robot was moved since the last session, call:
  ros2 service call /localize_globally std_srvs/srv/Empty "{}"
or delete ~/.ros/last_amcl_pose_<map>.yaml and relaunch.

The LiDAR runs as a systemd service (ros-sllidar-c1.service) and is always up,
so it is NOT started here. Wheel odometry + IMU + EKF (odom->base_link) and
map_server + AMCL (map->odom) come from bringup.

  ros2 launch home_robot localize.launch.py             # uses maps/malou2.yaml
  ros2 launch home_robot localize.launch.py map:=home   # a different saved map
  ros2 launch home_robot localize.launch.py map:=/abs/path/to/my.yaml
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _resolve_map(map_arg: str, share_dir: str) -> str:
    """A name like 'kela' -> full path to a maps/<name>.yaml; a path is used as-is."""
    if map_arg.endswith('.yaml') and os.path.isabs(map_arg):
        return map_arg
    name = map_arg if map_arg.endswith('.yaml') else f'{map_arg}.yaml'
    # Prefer the source tree (always current), fall back to the installed share.
    candidates = [
        os.path.expanduser(f'~/robot_ws/src/home_robot/maps/{name}'),
        os.path.join(share_dir, 'maps', name),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise RuntimeError(
        f"Map '{map_arg}' not found. Looked for: {', '.join(c for c in candidates if c)}")


def _launch_setup(context, *args, **kwargs):
    share_dir = FindPackageShare('home_robot').perform(context)
    map_yaml = _resolve_map(LaunchConfiguration('map').perform(context), share_dir)
    use_depth = LaunchConfiguration('use_depth_camera').perform(context).lower() in ('true', '1')
    use_joy = LaunchConfiguration('use_joy').perform(context).lower() in ('true', '1')
    # Inherited by the nested bringup (which starts arm_driver); read here too
    # so the right stick can jog the arm in localize mode as well.
    use_arm = LaunchConfiguration('use_arm').perform(context).lower() in ('true', '1')
    use_apriltag = LaunchConfiguration('use_apriltag').perform(context).lower() in ('true', '1')
    use_rviz = LaunchConfiguration('use_rviz').perform(context).lower() in ('true', '1')
    use_obstacle_safety = LaunchConfiguration('use_obstacle_safety').perform(context).lower() in ('true', '1')
    # Opt-in perception stack for navigation: YOLO detector + tracker feeding the
    # dynamic-obstacle costmap layers (predicted/semantic) + semantic object
    # memory. Off by default because the detector costs iGPU/CPU; when on, the
    # full realsense (color+depth+aligned+pointcloud) from bringup replaces the
    # lean depth-only stream we'd otherwise start below.
    use_perception = LaunchConfiguration('use_perception').perform(context).lower() in ('true', '1')
    perc = 'true' if use_perception else 'false'
    # Physical stuck detection + escape (BackUp/creep) before Nav2's raw
    # Spin/BackUp recovery loop. Default ON: without it, a near-wall room goal
    # drives the footprint into high inflation (cost ~99), the planner can no
    # longer plan (start-in-collision) and Nav2 loops recoveries until ABORT.
    # HW-confirmed 2026-07-07 (kela3, goto domatio tou max).
    use_recovery = LaunchConfiguration('use_recovery').perform(context).lower() in ('true', '1')
    use_dashboard = LaunchConfiguration('use_dashboard').perform(context).lower() in ('true', '1')

    pkg = FindPackageShare('home_robot')
    actions = []

    # bringup: with use_perception the heavy object detector (YOLO/iGPU) + the
    # dynamic-obstacle layers come up and bringup starts the full camera; without
    # it, use_camera:=false and we start a lean depth-only stream below.
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg, '/launch/bringup.launch.py']),
        launch_arguments={
            'use_localization':    'true',
            'localization_map':    map_yaml,
            'use_slam':            'false',   # AMCL owns map->odom; no slam_toolbox
            'use_rtabmap':         'false',
            'use_camera':          perc,      # detector on only with perception
            'use_tracker':         perc,      # track_id/velocity for prediction
            'use_prediction':      perc,      # /predicted_obstacles costmap layer
            'use_semantic_costmap': perc,     # /semantic_obstacles costmap layer
            'use_object_memory':   perc,      # remembers where objects are (map frame)
            # Without this the LLM's "Τρέχουσα κατάσταση" block only ever holds
            # the clock: no room, no battery, and no nearby-object distances.
            # That is why "πόση απόσταση έχει;" answered "δεν ξέρω" on
            # 2026-07-30 — the node was never started, on any flag combination,
            # because this argument was simply not forwarded (bringup defaults
            # it to false).
            #
            # Tied to `perc` it was still off under plain `robot max`, which is
            # how the robot ships: /situation_context had no publisher at all
            # (checked live 2026-08-02). But the room name comes from TF plus
            # the room mask and the CPU/RAM from psutil — none of that needs the
            # detector. Perception only enriches the "nearby objects" line, so
            # this is unconditional and the node degrades to a shorter context
            # when the detections aren't there.
            'use_situational':     'true',    # fills the LLM's situation context
            # Docking is a mission (navigate to the handover point in front of
            # the base, relocalize on the tag above it, then hand over to IR
            # homing), and this is the launch `robot max` uses — with the
            # executor off, "πήγαινε να φορτίσεις" would publish mission/start
            # to nobody. It needs no extra hardware.
            'use_mission':         'true',
            # Same omission as use_situational above, one node over: llm_bridge
            # always offers the `tidy` and `patrol` tools, but their topics had
            # no subscriber at all under `robot max` (checked live 2026-07-31:
            # tidy_command and patrol_command both Subscription count: 0). So
            # the robot answered "Ξεκινάω περιπολία" and then stood still. The
            # planner only waits on topics; the clutter check it runs at each
            # stop needs use_perception, the driving does not.
            'use_planner':         'true',
            # Forwarded EXPLICITLY, not left to inheritance. An unforwarded
            # `use_*` only reaches bringup when the command line happens to set
            # it, which is exactly how use_situational stayed off under every
            # flag combination for weeks. `robot max` should always get the
            # dashboard, so it is decided here.
            'use_dashboard':       'true' if use_dashboard else 'false',
            # Third instance of the same omission (use_situational, use_planner,
            # now this): `llm_backend` was not forwarded at all, so
            # `robot max llm_backend:=gemini` was accepted on the command line
            # and then silently ignored — bringup fell back to its 'lemonade'
            # default and the robot kept thinking on the NPU. Nothing errored;
            # the only symptom was the 4.7 GB that never got freed.
            'llm_backend':         LaunchConfiguration('llm_backend', default='gemini'),
            'use_recovery':        'true' if use_recovery else 'false',
            'use_obstacle_safety': 'true' if use_obstacle_safety else 'false',
            # Forwarded, not hardcoded: `robot max` decides. It was pinned to
            # 'true' here, which silently swallowed use_rviz:=false — rviz2 then
            # started with no DISPLAY exported (tty/SSH launch) and just died.
            'use_rviz':            'true' if use_rviz else 'false',
            # We start joy/teleop ourselves below (correct device-by-name +
            # cmd_vel_safe remap). Force bringup's own joy OFF so it doesn't
            # also spawn one — our 'use_joy' arg is inherited into bringup
            # otherwise and would double-launch it with the wrong settings.
            'use_joy':             'false',
        }.items(),
    ))

    # D435 depth driver only (no color/pointcloud, no detector) so the
    # global_localizer can fuse the forward depth virtual-scan with the 360°
    # LiDAR — much better global localization from a random start position.
    # Skipped under use_perception: bringup's full camera already covers depth
    # (starting a second realsense would fight over the USB device).
    if use_depth and not use_perception:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
            ]),
            launch_arguments={
                'enable_color':       'true',    # needed for AprilTag reference-tag relocalization
                'enable_depth':       'true',
                'enable_infra1':      'false',
                'enable_infra2':      'false',
                'align_depth.enable': 'false',
                'pointcloud.enable':  'false',
            }.items(),
        ))

    # PS5 DualSense teleop, started here (NOT via bringup's use_joy) for two
    # reasons specific to localize mode:
    #   1. bringup's joy_node uses the ROS1 'dev' param, ignored on Jazzy —
    #      it would open device_id 0 (the wrong pad). We select the DualSense
    #      by name so it works regardless of which /dev/input/jsN it landed on.
    #   2. localize runs with use_obstacle_safety:=false, so nothing relays
    #      cmd_vel -> cmd_vel_safe. The Roomba obeys only cmd_vel_safe (the
    #      collision_monitor's output topic), so teleop must publish straight
    #      to cmd_vel_safe or the robot won't move.
    # R1 is the dead-man's switch (see teleop_twist_joy_ps5.yaml); left stick
    # drives. autorepeat_rate keeps cmd_vel flowing while a stick is held.
    if use_joy:
        actions.append(Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'device_name': 'DualSense Wireless Controller',
                'deadzone': 0.05,
                'autorepeat_rate': 20.0,
            }],
        ))
        actions.append(Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_twist_joy_node',
            parameters=[PathJoinSubstitution([pkg, 'config', 'teleop_twist_joy_ps5.yaml'])],
            remappings=[('cmd_vel', 'cmd_vel_safe')],
        ))
        # Sticky soft e-stop: Circle latches the robot stopped (roomba_driver
        # zeros the wheels + ignores cmd_vel), Share+Options together resets.
        actions.append(Node(
            package='home_robot',
            executable='joystick_estop_node.py',
            name='joystick_estop',
        ))
        # Right stick jogs the arm (base/shoulder), R1/R2 the gripper. Started
        # here for the same reason as the teleop above: bringup's own use_joy
        # is forced off, so its copy of this node never runs in localize mode.
        if use_arm:
            actions.append(Node(
                package='home_robot',
                executable='arm_joy_node.py',
                name='arm_joy',
                parameters=[PathJoinSubstitution([pkg, 'config', 'arm_joy_ps5.yaml'])],
            ))

    # AprilTag reference-tag relocalization: detect the single saloni tag off the
    # D435 color stream (publishes TF camera_color_optical_frame -> saloni_tag),
    # then apriltag_relocalizer turns a sighting into a one-shot /initialpose fix
    # (no manual 2D Pose Estimate, no wrong-room scan-match). Needs use_depth_camera
    # (the color stream) to be on.
    if use_apriltag:
        actions.append(Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_node',
            output='screen',
            parameters=[PathJoinSubstitution([pkg, 'config', 'apriltag.yaml'])],
            remappings=[
                ('image_rect', '/camera/camera/color/image_raw'),
                ('camera_info', '/camera/camera/color/camera_info'),
            ],
        ))
        actions.append(Node(
            package='home_robot',
            executable='apriltag_relocalizer.py',
            name='apriltag_relocalizer',
            output='screen',
            parameters=[{
                'tag_frame': 'saloni_tag',
                'base_frame': 'base_link',
                'map_frame': 'map',
            }],
        ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map', default_value='malou2',
            description='Saved map name (in maps/) or a full path to a .yaml'),
        DeclareLaunchArgument(
            'use_depth_camera', default_value='true',
            description='Start the D435 depth stream so global_localizer fuses it '
                        'with the LiDAR (set false for LiDAR-only)'),
        DeclareLaunchArgument(
            'use_joy', default_value='true',
            description='Start PS5 DualSense teleop (R1 = dead-man, left stick) '
                        'wired straight to cmd_vel_safe for localize mode'),
        DeclareLaunchArgument(
            'use_arm', default_value='false',
            description='Start the RoArm-M3 (inherited by bringup) and, with '
                        'use_joy, the right-stick jog for it'),
        DeclareLaunchArgument(
            'use_apriltag', default_value='true',
            description='Detect the saloni reference AprilTag off the D435 color '
                        'stream and relocalize from a sighting (needs use_depth_camera)'),
        DeclareLaunchArgument(
            'use_obstacle_safety', default_value='false',
            description='Relay cmd_vel -> cmd_vel_safe through velocity_smoother + '
                        'collision_monitor (needed for autonomous drive tests; teleop '
                        'still bypasses via its direct cmd_vel_safe remap)'),
        DeclareLaunchArgument(
            'use_recovery', default_value='true',
            description='Run recovery_manager_node: detects the robot physically '
                        'stuck (cmd_vel commanded but no displacement) near walls/'
                        'doorways and does a BackUp/creep escape before Nav2 loops '
                        'raw Spin/BackUp into an ABORT. HW-confirmed 2026-07-07.'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Open RViz on the local display (`robot max` sets this; '
                        'the VNC :2 session runs its own rviz2 for the phone)'),
        DeclareLaunchArgument(
            'use_dashboard', default_value='true',
            description='Serve the web dashboard on :8080 — map, camera, arm, '
                        'vacuum, LLM chat, and RViz/MoveIt/Gazebo streamed from '
                        'VNC. Token printed at startup; set false to skip it.'),
        DeclareLaunchArgument(
            'use_perception', default_value='false',
            description='Bring up the YOLO detector + tracker, the dynamic-obstacle '
                        'costmap layers (predicted/semantic) and semantic object '
                        'memory during navigation. Starts the full camera (replaces '
                        'the lean depth-only stream); costs iGPU/CPU.'),
        DeclareLaunchArgument(
            'llm_backend', default_value='gemini',
            description="Which LLM answers: 'gemini' is the cloud (the default — "
                        "same tool-call rate as the NPU model at 0.45 s instead of "
                        "6.7 s, and it frees 4.7 GB; needs ~/.home_robot/gemini_api_key "
                        "and a network), 'lemonade' is FastFlowLM on the NPU "
                        "(offline, holds that 4.7 GB), 'ollama' a local GGUF server."),
        OpaqueFunction(function=_launch_setup),
    ])
