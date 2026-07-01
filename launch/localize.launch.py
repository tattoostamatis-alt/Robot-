"""One command to open a saved map and localize on it — no manual 2D Pose Estimate.

Loads a saved map (default 'kela') and brings up AMCL + pose_saver (restores
the last pose when available) + global_localizer (FFT scan-match only when
no saved pose) + RViz, by including bringup.launch.py with the heavy
AI/voice/camera stack switched off.

If the robot was moved since the last session, call:
  ros2 service call /localize_globally std_srvs/srv/Empty "{}"
or delete ~/.ros/last_amcl_pose_<map>.yaml and relaunch.

The LiDAR runs as a systemd service (ros-sllidar-c1.service) and is always up,
so it is NOT started here. Wheel odometry + IMU + EKF (odom->base_link) and
map_server + AMCL (map->odom) come from bringup.

  ros2 launch home_robot localize.launch.py             # uses maps/kela.yaml
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

    pkg = FindPackageShare('home_robot')
    actions = []

    # bringup with use_camera:=false so the heavy object detector (YOLO/NPU)
    # stays off — we only want the D435 *depth* stream for localization, which
    # we start leanly below.
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg, '/launch/bringup.launch.py']),
        launch_arguments={
            'use_localization':    'true',
            'localization_map':    map_yaml,
            'use_slam':            'false',   # AMCL owns map->odom; no slam_toolbox
            'use_rtabmap':         'false',
            'use_camera':          'false',   # detector off; depth started separately
            'use_mission':         'false',
            'use_recovery':        'false',
            'use_obstacle_safety': 'false',
            'use_rviz':            'true',
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
    if use_depth:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
            ]),
            launch_arguments={
                'enable_color':       'false',
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

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map', default_value='kela',
            description='Saved map name (in maps/) or a full path to a .yaml'),
        DeclareLaunchArgument(
            'use_depth_camera', default_value='true',
            description='Start the D435 depth stream so global_localizer fuses it '
                        'with the LiDAR (set false for LiDAR-only)'),
        DeclareLaunchArgument(
            'use_joy', default_value='true',
            description='Start PS5 DualSense teleop (R1 = dead-man, left stick) '
                        'wired straight to cmd_vel_safe for localize mode'),
        OpaqueFunction(function=_launch_setup),
    ])
