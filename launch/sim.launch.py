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
from nav2_common.launch import RewrittenYaml


# Free pose with ~1 m clearance, computed from dimi.pgm (see scripts/map_to_world.py).
INIT_X, INIT_Y, INIT_YAW = 1.885, -1.045, 0.0


def generate_launch_description():
    pkg = get_package_share_directory('home_robot')
    nav2_bringup = get_package_share_directory('nav2_bringup')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(pkg, 'worlds', 'dimi.world')
    xacro_file = os.path.join(pkg, 'description', 'robot_sim.urdf.xacro')
    # nav2_params.yaml hard-codes use_sim_time:false throughout (hardware config).
    # Rewrite every occurrence to true for sim so all Nav2 nodes run on /clock —
    # a SetParameter override loses to the params file, so patch the file itself.
    params_file = RewrittenYaml(
        source_file=os.path.join(pkg, 'config', 'nav2_params.yaml'),
        param_rewrites={'use_sim_time': 'true'},
        convert_types=True,
    )
    map_yaml = LaunchConfiguration('map')
    use_rviz = LaunchConfiguration('use_rviz')
    headless = LaunchConfiguration('headless')

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    # ── Gazebo ───────────────────────────────────────────────────
    # headless:=true runs the server only (-s), for CI / display-less hosts.
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r -v3 ',
                                      PythonExpression(["'-s --headless-rendering ' if '", headless,
                                                        "'.lower()==\"true\" else ''"]),
                                      world]}.items(),
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
                   '-x', str(INIT_X), '-y', str(INIT_Y), '-z', '0.05',
                   '-Y', str(INIT_YAW)],
        output='screen',
    )

    # ── gz <-> ROS bridge ────────────────────────────────────────
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel_safe@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
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

    # Seed AMCL once it (and TF) are up — AMCL only starts after the 12 s nav delay.
    set_initial_pose = TimerAction(period=22.0, actions=[ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '-1', '/initialpose',
             'geometry_msgs/msg/PoseWithCovarianceStamped',
             '{header: {frame_id: map}, pose: {pose: {position: {x: %f, y: %f, z: 0.0}, '
             'orientation: {z: 0.0, w: 1.0}}}}' % (INIT_X, INIT_Y)],
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
        localization,
        navigation,
    ])])

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=os.path.expanduser('~/robot_ws/maps/dimi.yaml')),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='run gz server only (no GUI) for display-less hosts'),
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
