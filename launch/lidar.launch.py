"""LiDAR bringup, with an optional arm self-filter.

Replaces the bare `sllidar_ros2 sllidar_c1_launch.py` that the
ros-sllidar-c1.service used to run.

The arm filter is OFF by default since 2026-07-21. On the old Roomba 692 the
LiDAR sat 220mm up and the RoArm-M3 base cut straight through the scan plane
~0.13m away, so it had to be boxed out. On the 879 the LiDAR is on a mast at
606mm while the arm base is at 186mm -- 420mm below it -- and the user
confirmed live that there is no phantom return any more. Leaving the filter on
would be actively dangerous: it blanks everything within 0.26m in front, so
real obstacles would be deleted exactly where the robot hits them first.

Set use_arm_filter:=true to bring it back if the arm ever reaches into the
scan plane again (see config/lidar_arm_filter.yaml, whose box is still sized
for the OLD geometry and would need re-measuring first).

With the filter off the LiDAR publishes straight to /scan; with it on it
publishes to /scan_raw and the filter republishes the clean scan on /scan.

Started by /home/dimi/bin/start_sllidar_c1 (systemd: ros-sllidar-c1.service).
Params mirror sllidar_ros2/launch/sllidar_c1_launch.py (serial C1 defaults).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)


def generate_launch_description():
    use_arm_filter = LaunchConfiguration('use_arm_filter')

    filter_cfg = PathJoinSubstitution(
        [FindPackageShare('home_robot'), 'config', 'lidar_arm_filter.yaml'])

    # Publish to /scan_raw only when the filter is there to clean it up and
    # republish /scan; otherwise go straight to /scan so consumers still get data.
    scan_topic = PythonExpression(
        ["'scan_raw' if '", use_arm_filter, "' == 'true' else 'scan'"])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_arm_filter', default_value='false',
            description='Box-filter the RoArm-M3 out of the scan. Off on the '
                        '879 chassis: the arm is no longer in the scan plane, '
                        'and the box would delete real obstacles.'),
        Node(
            package='sllidar_ros2',
            executable='sllidar_node',
            name='sllidar_node',
            output='screen',
            parameters=[{
                'channel_type': 'serial',
                'serial_port': '/dev/sllidar',
                'serial_baudrate': 460800,
                'frame_id': 'laser',
                'inverted': False,
                'angle_compensate': True,
                'scan_mode': 'Standard',
            }],
            remappings=[('scan', scan_topic)],
        ),
        Node(
            package='laser_filters',
            executable='scan_to_scan_filter_chain',
            name='lidar_arm_filter',
            output='screen',
            parameters=[filter_cfg],
            condition=IfCondition(use_arm_filter),
            remappings=[
                ('scan', 'scan_raw'),        # input  <- raw lidar
                ('scan_filtered', 'scan'),   # output -> clean scan for everyone
            ],
        ),
    ])
