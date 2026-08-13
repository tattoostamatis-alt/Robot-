"""Standalone AMCL localization stack — map_server + AMCL + pose_saver +
global_localizer + room_markers + the AprilTag reference-tag relocalizer.

This is the localization slice of bringup.launch.py / localize.launch.py,
pulled out so it can be started and killed on its own. `switch_map_mode.sh`
uses it to hot-swap between mapping (map_mode_slam.launch.py) and localizing
against a saved map WITHOUT tearing down the rest of the stack (voice,
camera, arm, Nav2 planner/controller, dashboard, VNC, Foxglove) — that full
restart is what the dashboard's «Νέος χάρτης» / «Αλλαγή χάρτη» buttons used
to do, and it took ~90s and dropped the voice pipeline mid-swap on the way
back up.

‼️ NOT the same thing as launch/slam_only.launch.py — that is an older,
unrelated standalone test rig (its own roomba_driver/imu_node/ekf_node) for
mapping runs with nothing else up. Launching it alongside a running full
stack would start duplicate driver nodes fighting over the same serial
ports. This file starts nothing bringup.launch.py doesn't already own; it
only adds the localization NODES on top of an already-running stack.

Nav2's planner/controller/costmaps, started separately by bringup.launch.py,
keep running the whole time and just pick up the new map_server/AMCL (or
slam_toolbox) once it starts publishing /map and map->odom TF — costmap's
static_layer subscribes with transient_local durability, so a fresh
publisher's latched message reaches it with no resubscribe needed.

  ros2 launch home_robot map_mode_localize.launch.py             # maps/malou2.yaml
  ros2 launch home_robot map_mode_localize.launch.py map:=l      # a different saved map
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _resolve_map(map_arg: str, share_dir: str) -> str:
    """A name like 'kela' -> full path to a maps/<name>.yaml; a path is used as-is.

    Same lookup as localize.launch.py's helper (source tree preferred over
    installed share, since the source tree is what map_session.sh writes new
    saves into) — kept in sync deliberately, not imported, because launch
    files here are standalone entry points, not a shared module.
    """
    if map_arg.endswith('.yaml') and os.path.isabs(map_arg):
        return map_arg
    name = map_arg if map_arg.endswith('.yaml') else f'{map_arg}.yaml'
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
    use_apriltag = LaunchConfiguration('use_apriltag').perform(context).lower() in ('true', '1')

    pkg = FindPackageShare('home_robot')
    nav2_pkg = FindPackageShare('nav2_bringup')
    actions = []

    # map_server + AMCL + lifecycle_manager_localization. Mirrors
    # bringup.launch.py's use_localization block exactly (same params file,
    # same node names) so it is a drop-in swap for the copy that a full
    # `robot max`/`robot map` launch would have started.
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_pkg, '/launch/localization_launch.py']),
        launch_arguments={
            'map': map_yaml,
            'params_file': PathJoinSubstitution([pkg, 'config', 'nav2_params.yaml']),
            'use_sim_time': 'false',
        }.items(),
    ))

    actions.append(Node(
        package='home_robot',
        executable='room_markers_node.py',
        name='room_markers_node',
        output='screen',
    ))

    actions.append(Node(
        package='home_robot',
        executable='pose_saver_node.py',
        name='pose_saver_node',
        output='screen',
        parameters=[{'save_interval': 10.0, 'map_yaml': map_yaml,
                     'restore_pose': False}],
    ))

    actions.append(Node(
        package='home_robot',
        executable='global_localizer_node.py',
        name='global_localizer',
        output='screen',
        parameters=[{
            'auto_localize': True,
            'defer_to_saved_pose': False,
            'map_yaml': map_yaml,
            'depth_weight': 0.5,
        }],
    ))

    actions.append(TimerAction(
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
    ))

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
            'use_apriltag', default_value='true',
            description='Detect the saloni reference AprilTag off the D435 color '
                        'stream and relocalize from a sighting'),
        OpaqueFunction(function=_launch_setup),
    ])
