"""LiDAR bringup with an arm self-filter.

Replaces the bare `sllidar_ros2 sllidar_c1_launch.py` that the
ros-sllidar-c1.service used to run. The SLLIDAR C1 now publishes to /scan_raw,
and a laser_filters box filter strips the RoArm-M3 arm out of it and republishes
the clean scan on /scan (the topic every consumer already subscribes to).

Started by /home/dimi/bin/start_sllidar_c1 (systemd: ros-sllidar-c1.service).
Params mirror sllidar_ros2/launch/sllidar_c1_launch.py (serial C1 defaults).
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    filter_cfg = PathJoinSubstitution(
        [FindPackageShare('home_robot'), 'config', 'lidar_arm_filter.yaml'])

    return LaunchDescription([
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
            remappings=[('scan', 'scan_raw')],
        ),
        Node(
            package='laser_filters',
            executable='scan_to_scan_filter_chain',
            name='lidar_arm_filter',
            output='screen',
            parameters=[filter_cfg],
            remappings=[
                ('scan', 'scan_raw'),        # input  <- raw lidar
                ('scan_filtered', 'scan'),   # output -> clean scan for everyone
            ],
        ),
    ])
