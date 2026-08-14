#!/usr/bin/env python3
"""Self-contained Gazebo nav sim — exercises the REAL nav2_params.yaml stack
(planner / MPPI / velocity_smoother / collision_monitor / AMCL) against the dimi
map without any hardware.

Deliberately standalone (does NOT go through bringup.launch.py): a minimal
diff-drive robot subscribes to cmd_vel_safe — the last topic in the real command
chain — so the whole smoother + collision-monitor path is under test. gz supplies
/scan, /odom and the odom->base_link TF that the real EKF would provide, AMCL
supplies map->odom, and goal_pose_bridge forwards RViz "2D Goal Pose" to Nav2.

    ros2 launch home_robot sim.launch.py

Then in RViz drop a "2D Goal Pose" and watch cmd_vel_safe: after the 2026-07-04
accel-limit fix v should ramp smoothly instead of pinning w at ±0.6 (the old
spin-in-place bug).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue

# Free pose with ~1 m clearance, computed from dimi.pgm (see scripts/map_to_world.py).
# Defaults for the dimi world; override world/map/init_* together to sim another
# apartment, e.g. kela3:
#   sim.launch.py world:=.../kela3.world map:=.../kela3.yaml init_x:=... init_y:=...
INIT_X, INIT_Y, INIT_YAW = 1.885, -1.045, 0.0


def generate_launch_description():
    pkg = get_package_share_directory('home_robot')
    nav2_bringup = get_package_share_directory('nav2_bringup')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = LaunchConfiguration('world')
    init_x = LaunchConfiguration('init_x')
    init_y = LaunchConfiguration('init_y')
    init_yaw = LaunchConfiguration('init_yaw')
    xacro_file = os.path.join(pkg, 'description', 'robot_sim.urdf.xacro')
    # nav2_params.yaml hard-codes use_sim_time:false throughout (hardware config).
    # Rewrite every occurrence to true for sim so all Nav2 nodes run on /clock —
    # a SetParameter override loses to the params file, so patch the file itself.
    # Also sim-only: disable the GLOBAL costmap's obstacle_layer. The generated
    # world's 0.10m wall boxes protrude up to 5cm past the real map walls, so
    # scan marks + inflation seal kela3's ~0.78m doorways below the inscribed
    # radius and NavFn stops finding plans (verified live 2026-07-07: BFS on
    # the live costmap showed start/goal disconnected under cost<99). The
    # LOCAL costmap keeps its scan for reactive avoidance. RewrittenYaml can't
    # target one node's key, so patch the file directly.
    def _make_sim_params():
        import tempfile
        import yaml as _yaml
        with open(os.path.join(pkg, 'config', 'nav2_params.yaml')) as fh:
            data = _yaml.safe_load(fh)
        def _walk(d):
            for k, v in d.items():
                if k == 'use_sim_time':
                    d[k] = True
                elif isinstance(v, dict):
                    _walk(v)
        _walk(data)
        data['global_costmap']['global_costmap']['ros__parameters'][
            'obstacle_layer']['enabled'] = False
        # Same geometry error, local side: give MPPI back the ~8cm the chunky
        # sim walls steal from kela3's borderline corridors, else FollowPath
        # aborts at the corridor pinch that the real robot squeezes through.
        data['local_costmap']['local_costmap']['ros__parameters'][
            'inflation_layer']['inflation_radius'] = 0.22
        out = tempfile.NamedTemporaryFile(
            'w', suffix='_sim_nav2.yaml', delete=False)
        _yaml.safe_dump(data, out)
        out.close()
        return out.name
    params_file = _make_sim_params()
    map_yaml = LaunchConfiguration('map')
    use_rviz = LaunchConfiguration('use_rviz')
    headless = LaunchConfiguration('headless')

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    # ── Gazebo ───────────────────────────────────────────────────
    # headless:=true runs the server only (-s), for CI / display-less hosts.
    #
    # render_engine defaults to Ogre 1.x, NOT the ogre2 (Ogre-Next) default.
    # ‼️ This machine has no usable GL on the displays the sim ever runs on: the
    # dashboard's Gazebo tab renders into a headless Xvnc (scripts/gui_session.sh)
    # and ogre2 takes the EGL path, where Mesa dies:
    #
    #   libEGL warning: Not allowed to force software rendering when API
    #                   explicitly selects a hardware device.
    #   ...libgz-sim-sensors-system.so -> libgz-rendering-ogre2.so
    #      -> RenderSystem_GL3Plus.so -> libEGL_mesa -> driCreateNewScreen3
    #   Segmentation fault (Address not mapped to object [0x8])
    #
    # Two things that look like fixes are not. LIBGL_ALWAYS_SOFTWARE=1 steers
    # GLX, and that warning is Mesa saying it ignored it on the EGL path. And
    # --render-engine-gui only re-points the GUI: the crash above is in the
    # SERVER's sensors system, which renders the sim's lidar and camera and
    # keeps its own engine. Fixing only the GUI left the tab black exactly as
    # before. So this sets the engine for both halves.
    #
    # Ogre 1.x uses GLX, honours the software flag, and renders on the same
    # display (verified on :3, 2026-08-02). Pass render_engine:=ogre2 on a host
    # with a real GPU.
    #
    # One expression for the whole mode switch — headless still needs the engine,
    # because --headless-rendering is exactly the sensor path that crashed.
    gz_mode = PythonExpression([
        "('-s --headless-rendering ' if '", headless, "'.lower()=='true' else '') "
        "+ '--render-engine ' + '", LaunchConfiguration('render_engine'), "' + ' '",
    ])
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r -v3 ', gz_mode, world]}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'robot_sim',
                   '-x', init_x, '-y', init_y, '-z', '0.05',
                   '-Y', init_yaw],
        output='screen',
    )

    # ── gz <-> ROS bridge ────────────────────────────────────────
    # Camera topics are bridged under their gz-side names (/camera/image etc,
    # the rgbd_camera sensor's auto-suffixed <topic>camera</topic> from the
    # xacro) and then remapped to the exact names rtabmap.launch.py's
    # hardcoded `remaps` expect (see its docstring — this only exists to
    # feed that launch file for a sim test of the anchor_frame fix, real
    # bringup.launch.py/localize.launch.py own those names normally).
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel_safe@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        remappings=[
            ('/camera/image', '/camera/camera/color/image_raw'),
            ('/camera/depth_image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('/camera/camera_info', '/camera/camera/color/camera_info'),
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # gz's own /tf output was unreliable over the bridge; derive odom->base_link
    # from the (reliable) /odom topic instead so it's up before Nav2 activates.
    odom_tf = Node(
        package='home_robot',
        executable='odom_tf_broadcaster.py',
        name='odom_tf_broadcaster',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # ── Nav2: localization (map_server + AMCL) + navigation ──────
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup, 'launch', 'localization_launch.py')),
        launch_arguments={
            'map': map_yaml,
            'params_file': params_file,
            'use_sim_time': 'true',
            'autostart': 'true',
        }.items(),
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'params_file': params_file,
            'use_sim_time': 'true',
            'autostart': 'true',
        }.items(),
    )

    goal_pose_bridge = Node(
        package='home_robot',
        executable='goal_pose_bridge.py',
        name='goal_pose_bridge',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Real robot's doorway-pinch recovery (bringup.launch.py). Watches
    # cmd_vel_smoothed vs /odom; when motion is commanded but the body doesn't
    # move (footprint wedged in a tight doorway where Nav2's full-footprint Spin
    # sweep aborts) it runs translation-first BackUp/DriveOnHeading + a direct
    # creep nudge. Without it the sim only exercises bare Nav2 recovery, which
    # thrashes at kela3's ~0.78 m doorways (e.g. exiting domatio_max).
    recovery_manager = Node(
        package='home_robot',
        executable='recovery_manager_node.py',
        name='recovery_manager_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Seed AMCL once it (and TF) are up — AMCL only starts after the 12 s nav delay.
    set_initial_pose = TimerAction(period=22.0, actions=[ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '-1', '/initialpose',
             'geometry_msgs/msg/PoseWithCovarianceStamped',
             ['{header: {frame_id: map}, pose: {pose: {position: {x: ', init_x,
              ', y: ', init_y, ', z: 0.0}, orientation: {z: 0.0, w: 1.0}}}}']],
        output='screen')])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', os.path.join(nav2_bringup, 'rviz', 'nav2_default_view.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz),
        output='log',
    )

    # Hold Nav2 back until gz has loaded the 574-wall world, spawned the robot and
    # started odom (~10 s), then bring up localization + navigation with the
    # lifecycle bond watchdog disabled (bond_timeout:=0.0) — same as the real robot
    # (bringup.launch.py, commit 5fa73a0). Without this the costmap's brief wait for
    # the odom->base_link TF at activation trips the 4 s bond and aborts the chain.
    delayed_nav = TimerAction(period=12.0, actions=[GroupAction([
        SetParameter('bond_timeout', 0.0),
        # Same doorway-friendly recovery tree as the real robot (bringup).
        # default_nav_to_pose_bt_xml is NOT in the params file, so this
        # SetParameter wins (unlike use_sim_time above).
        SetParameter('default_nav_to_pose_bt_xml', os.path.join(
            pkg, 'config', 'behavior_trees',
            'navigate_to_pose_w_replanning_and_recovery.xml')),
        localization,
        navigation,
        recovery_manager,
    ])])

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=os.path.expanduser('~/robot_ws/maps/dimi.yaml')),
        DeclareLaunchArgument('world', default_value=os.path.join(pkg, 'worlds', 'dimi.world')),
        DeclareLaunchArgument('init_x', default_value=str(INIT_X)),
        DeclareLaunchArgument('init_y', default_value=str(INIT_Y)),
        DeclareLaunchArgument('init_yaw', default_value=str(INIT_YAW)),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='run gz server only (no GUI) for display-less hosts'),
        DeclareLaunchArgument('render_engine', default_value='ogre',
                              description='Render engine for BOTH the GUI and the server-side '
                                          'sensors: ogre (Ogre 1.x, GLX — works on headless '
                                          'Xvnc) or ogre2 (Ogre-Next, EGL — needs a real GPU)'),
        gz,
        robot_state_publisher,
        spawn,
        bridge,
        odom_tf,
        delayed_nav,
        goal_pose_bridge,
        set_initial_pose,
        rviz,
    ])
