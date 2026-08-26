"""One command to open a saved map and localize on it — no manual 2D Pose Estimate.

Loads a saved map (default 'room4') and brings up AMCL + pose_saver (restores
the last pose when available) + global_localizer (FFT scan-match only when
no saved pose) + RViz, by including bringup.launch.py with the heavy
AI/voice/camera stack switched off.

If the robot was moved since the last session, call:
  ros2 service call /localize_globally std_srvs/srv/Empty "{}"
or delete ~/.ros/last_amcl_pose_<map>.yaml and relaunch.

The LiDAR runs as the on-demand systemd service ros-sllidar-c1.service.  The
`robot` wrapper starts it for `robot max` / `robot map` and stops it for
`robot stop`, so it is NOT started again here. Wheel odometry + IMU + EKF
(odom->base_link) and map_server + AMCL (map->odom) come from bringup.

  ros2 launch home_robot localize.launch.py             # uses maps/room4.yaml
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
    # ‼️ MAPPING MODE. slam_toolbox owns map->odom, there is no saved map to
    # load, and AMCL/AprilTag would fight it — but EVERYTHING ELSE `robot max`
    # brings up stays identical: voice, DOA, dashboard, perception, and the PS5
    # pad selected BY NAME. That is the whole reason the dashboard's «Νέος
    # χάρτης» routes through this file. It used to shell out to a bare
    # `ros2 launch bringup.launch.py use_slam:=true`, and bringup defaults
    # use_wake_word/use_stt/use_tts/use_llm to FALSE — so pressing the button
    # tore the stack down and brought back a robot with no voice at all, no
    # Foxglove, no VNC :2 (phone RViz) and bringup's own joy_node, which picks
    # the pad with the ROS1 `dev` param that Jazzy ignores. From the browser it
    # looked like the whole stack had simply died.
    use_slam = LaunchConfiguration('use_slam').perform(context).lower() in ('true', '1')
    # Not resolved while mapping: there is no map to open, and _resolve_map
    # RAISES on a missing file — a fresh install with an empty maps/ could then
    # never start its first mapping run.
    map_yaml = '' if use_slam else _resolve_map(
        LaunchConfiguration('map').perform(context), share_dir)
    map_name = (os.path.splitext(os.path.basename(map_yaml))[0]
                if map_yaml else 'mapping')
    use_depth = LaunchConfiguration('use_depth_camera').perform(context).lower() in ('true', '1')
    use_joy = LaunchConfiguration('use_joy').perform(context).lower() in ('true', '1')
    # Inherited by the nested bringup (which starts arm_driver); read here too
    # so the right stick can jog the arm in localize mode as well.
    # No arm hardware is installed. Keep the legacy argument below harmless
    # until a replacement model is integrated.
    use_arm = False
    use_apriltag = LaunchConfiguration('use_apriltag').perform(context).lower() in ('true', '1')
    # A tag sighting publishes /initialpose, which is AMCL's input. Under SLAM
    # there is no AMCL to accept it and slam_toolbox is already producing
    # map->odom, so the relocalizer would only log at an empty topic.
    use_apriltag = use_apriltag and not use_slam
    use_rviz = LaunchConfiguration('use_rviz').perform(context).lower() in ('true', '1')
    use_obstacle_safety = LaunchConfiguration('use_obstacle_safety').perform(context).lower() in ('true', '1')
    # Opt-in perception stack for navigation: YOLO detector + tracker feeding the
    # dynamic-obstacle costmap layers (predicted/semantic) + semantic object
    # memory. Off by default because the detector costs iGPU/CPU; when on, the
    # full realsense (color+depth+aligned+pointcloud) from bringup replaces the
    # lean depth-only stream we'd otherwise start below.
    use_perception = LaunchConfiguration('use_perception').perform(context).lower() in ('true', '1')
    perc = 'true' if use_perception else 'false'
    # Defaults to whatever use_perception is, so `use_perception:=true` brings
    # the skeleton with it; pass use_pose:=false explicitly to keep the iGPU for
    # the detector alone.
    use_nerf = LaunchConfiguration('use_nerf').perform(context).lower() in ('true', '1')
    _pose_arg = LaunchConfiguration('use_pose').perform(context).strip().lower()
    use_pose = perc if _pose_arg in ('', 'auto') else (
        'true' if _pose_arg in ('true', '1') else 'false')
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
            # Exactly one of these two owns map->odom: AMCL against a saved map
            # (the normal case), or slam_toolbox building a new one.
            'use_localization':    'false' if use_slam else 'true',
            'localization_map':    map_yaml,
            'use_slam':            'true' if use_slam else 'false',
            'use_rtabmap':         'false',
            'use_camera':          perc,      # detector on only with perception
            'use_tracker':         perc,      # track_id/velocity for prediction
            'use_prediction':      perc,      # /predicted_obstacles costmap layer
            # Disabled: synthetic semantic cylinders produced persistent false
            # obstacles. LiDAR/prediction/cliff sources remain active.
            'use_semantic_costmap': 'false',
            'use_object_memory':   perc,      # remembers where objects are (map frame)
            # ‼️ Same omission as use_pose/use_situational below: declared in
            # bringup, defaulting false, and never forwarded here — so no
            # `robot max` could start it. That is why «πιάσε την παντόφλα»
            # failed on 2026-08-06: the arm can only grasp what a detector
            # publishes, object_detector publishes the COCO 80, and a slipper
            # is not one of them. Defaults to `perc` (shares the camera and the
            # iGPU) but the node idles until a tool asks for a non-COCO word.
            'use_open_vocab':      LaunchConfiguration('use_open_vocab'),
            # ‼️ Skeleton tracking was UNREACHABLE from `robot max`: use_pose is
            # declared in bringup and defaults false, and this file never
            # forwarded it — the exact shape of the use_situational bug below.
            # pose_node was written, tested and wired into the dashboard's
            # camera overlay, and no flag combination could start it.
            # Tied to `perc` because it is a perception feature and shares the
            # iGPU with the detector; use_pose:=false opts out.
            'use_pose':            use_pose,
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
            'llm_backend':         LaunchConfiguration('llm_backend', default='lemonade'),
            # Fourth instance of the same omission (use_situational, use_planner,
            # llm_backend, now this): `use_doa` was never forwarded, and bringup
            # declares it default false, so doa_node NEVER started under
            # `robot max` — confirmed live 2026-08-03 on a fully running stack
            # (every other voice node present, zero doa_node). The XVF3800's
            # direction-of-arrival and its hardware VAD were simply unused.
            'use_doa':             LaunchConfiguration('use_doa', default='true'),
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
            'moveit_transit':      LaunchConfiguration('moveit_transit', default='false'),
        }.items(),
    ))

    # D435 depth driver only (no detector) so the global_localizer can fuse the
    # forward depth virtual-scan with the 360° LiDAR — much better global
    # localization from a random start position.
    # Skipped under use_perception: bringup's full camera already covers depth
    # (starting a second realsense would fight over the USB device).
    #
    # ‼️ pointcloud.enable was 'false' here, and that quietly disarmed the
    # costmap. nav2_params.yaml has listed `depth_camera` as a voxel_layer
    # observation source all along, but on a default `robot max` the topic had
    # ZERO publishers (measured 2026-08-05: 2 subscribers, 0 publishers), so
    # everything the LiDAR's single 0.606 m slice misses — table tops, steps,
    # a bent-over person — reached the planner never. The Sensor fusion tab
    # made the gap visible; this is what closes it.
    #
    # decimation x2 is not a compromise for turning it on, it is a saving: the
    # camera node measured 34.1% CPU with no cloud at all, 63.3% with a full
    # 30 Hz cloud, and 32.6% with cloud + decimation x2 — BELOW the old
    # baseline, because the smaller depth frame makes every downstream filter
    # cheaper too. 37k points per cloud is still far more than a 5 Hz costmap
    # update can use.
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
                'pointcloud.enable':  'true',
                'decimation_filter.enable': 'true',
                'decimation_filter.filter_magnitude': '2',
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
            # Raw joy autorepeats at 20 Hz, including zero while the sticks are
            # idle. Sending that directly to cmd_vel_safe cuts between web
            # D-pad commands. The gate below forwards motion plus one release
            # STOP, and suppresses the remaining idle zeros.
            remappings=[('cmd_vel', 'cmd_vel_joy_raw')],
        ))
        actions.append(Node(
            package='home_robot',
            executable='joy_cmd_gate_node.py',
            name='joy_cmd_gate',
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
                # A map->tag calibration is meaningful for exactly one map.
                # Reusing saloni_tag_map_pose.yaml after switching to room4
                # could teleport AMCL into the old map's coordinate system.
                'calib_file': os.path.expanduser(
                    f'~/.ros/{map_name}_tag_map_pose.yaml'),
            }],
        ))

    # NeRF dataset capture. Idle until told to record, so it is on by default:
    # its whole value is that you can hit the button in the dashboard while the
    # robot happens to be somewhere interesting, and the poses it needs
    # (map -> camera_color_optical_frame) only exist while localisation runs.
    # It does NOT train — see scripts/train_nerf.py, which will not even start
    # while the perception stack holds the iGPU.
    if use_nerf:
        actions.append(Node(
            package='home_robot',
            executable='nerf_capture_node.py',
            name='nerf_capture_node',
            output='screen',
        ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map', default_value='room4',
            description='Saved map name (in maps/) or a full path to a .yaml'),
        DeclareLaunchArgument(
            'use_slam', default_value='false',
            description='MAPPING run instead of localization: slam_toolbox '
                        'builds a new map (no saved map, no AMCL, no AprilTag) '
                        'while the rest of the stack — voice, dashboard, DOA, '
                        'perception, the by-name PS5 pad — comes up exactly as '
                        'in `robot max`. This is what `robot map` and the '
                        "dashboard's «Νέος χάρτης» button use."),
        DeclareLaunchArgument(
            'use_depth_camera', default_value='true',
            description='Start the D435 depth stream so global_localizer fuses it '
                        'with the LiDAR (set false for LiDAR-only)'),
        DeclareLaunchArgument(
            'use_joy', default_value='false',
            description='Start PS5 DualSense teleop (R1 = dead-man, left stick) '
                        'wired straight to cmd_vel_safe for localize mode. Off by '
                        'default because its 20 Hz zero autorepeat fights Nav2; '
                        'enable explicitly only for a manual-driving session.'),
        DeclareLaunchArgument(
            'use_arm', default_value='false',
            description='Start the RoArm-M3 (inherited by bringup) and, with '
                        'use_joy, the right-stick jog for it'),
        DeclareLaunchArgument(
            'moveit_transit', default_value='false',
            description='pick_place_node.py routes its big transit moves through '
                        'MoveIt2 (collision-aware) instead of direct cartesian — '
                        'see pick_place_node.py\'s module docstring. Forwarded '
                        'explicitly below, not left to inheritance (see use_situational '
                        'et al. for why that has bitten this file before).'),
        DeclareLaunchArgument(
            'use_apriltag', default_value='true',
            description='Detect the saloni reference AprilTag off the D435 color '
                        'stream and relocalize from a sighting (needs use_depth_camera)'),
        DeclareLaunchArgument(
            'use_obstacle_safety', default_value='true',
            description='Relay cmd_vel -> cmd_vel_safe through velocity_smoother + '
                        'collision_monitor. Required for autonomous navigation: '
                        'roomba_driver intentionally listens only to cmd_vel_safe.'),
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
            'use_nerf', default_value='true',
            description='Record RGB frames + MEASURED camera poses for a NeRF '
                        'while driving (dashboard «NeRF» tab). Idle until asked; '
                        'training is a separate offline step.'),
        DeclareLaunchArgument(
            'use_pose', default_value='auto',
            description="YOLO11n-pose on the iGPU: 17 COCO keypoints per person, "
                        "drawn as a skeleton on the dashboard's camera tab. "
                        "'auto' follows use_perception; true/false to force."),
        DeclareLaunchArgument(
            'use_open_vocab', default_value='false',
            description='Bring up the YOLO-World + CLIP detector so `pick`, '
                        '`fetch` and `find` can target things outside the COCO '
                        '80 (a slipper, a charger, keys). Loads its own weights '
                        'but stays idle — and off the iGPU — until a tool names '
                        'a word COCO does not have.'),
        DeclareLaunchArgument(
            'use_perception', default_value='false',
            description='Bring up the YOLO detector + tracker, the dynamic-obstacle '
                        'costmap layers (predicted/semantic) and semantic object '
                        'memory during navigation. Starts the full camera (replaces '
                        'the lean depth-only stream); costs iGPU/CPU.'),
        DeclareLaunchArgument(
            'llm_backend', default_value='lemonade',
            description="Which LLM answers: 'lemonade' is FastFlowLM on the NPU "
                        "(the default since the 2026-08-12 RAM upgrade to 32 GB — "
                        "offline, holds 4.7 GB, plenty of headroom now), 'gemini' "
                        "is the cloud (same tool-call rate at 0.45 s instead of "
                        "6.7 s, frees 4.7 GB; needs ~/.home_robot/gemini_api_key "
                        "and a network), 'ollama' a local GGUF server."),
        DeclareLaunchArgument(
            'use_doa', default_value='true',
            description='Direction of Arrival from the reSpeaker XVF3800: the '
                        'angle a voice came from (/doa/angle), the LED ring, an '
                        'optional turn toward the speaker, and the DSP hardware '
                        'VAD on /voice_activity. Harmless without the mic — the '
                        'node just retries and logs.'),
        OpaqueFunction(function=_launch_setup),
    ])
