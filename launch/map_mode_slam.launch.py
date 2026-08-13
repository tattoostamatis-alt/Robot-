"""Standalone slam_toolbox — the mapping half of bringup.launch.py's
use_slam block, pulled out so it can be started/killed on its own.

`switch_map_mode.sh` uses this to hot-swap INTO mapping mode without
restarting the rest of the stack. Everything slam_toolbox needs (LiDAR
service, odom->base_link from the EKF, nav2_params.yaml) is already running
— this file only ever adds/removes the map->odom half of the TF tree.

‼️ NOT the same thing as launch/slam_only.launch.py — that is an older,
unrelated standalone test rig (its own roomba_driver/imu_node/ekf_node) for
mapping runs with nothing else up. Launching it alongside a running full
stack would start duplicate driver nodes fighting over the same serial
ports. This file starts nothing bringup.launch.py doesn't already own; it
only adds slam_toolbox on top of an already-running stack.

  ros2 launch home_robot map_mode_slam.launch.py                        # fresh map
  ros2 launch home_robot map_mode_slam.launch.py slam_mode:=localization \\
      slam_map_file:=/path/to/map   # relocalize against a fixed .posegraph
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler, TimerAction
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    pkg = FindPackageShare('home_robot')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    slam_mode = LaunchConfiguration('slam_mode', default='mapping')
    slam_map_file = LaunchConfiguration('slam_map_file', default='')

    # Same node, same params file, same manual configure->activate sequence
    # as bringup.launch.py's slam_toolbox_node — kept in sync deliberately
    # (not shared) because this file is a standalone entry point.
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
    )

    slam_configure_event = TimerAction(
        period=5.0,
        actions=[EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_toolbox_node),
                transition_id=Transition.TRANSITION_CONFIGURE,
            ),
        )],
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
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('slam_mode', default_value='mapping'),
        DeclareLaunchArgument('slam_map_file', default_value=''),
        slam_toolbox_node,
        slam_configure_event,
        slam_activate_event,
    ])
