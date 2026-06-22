"""Load a saved map and view it in RViz — auto-configures and activates map_server
(it boots as a lifecycle node in `unconfigured` state, so without a lifecycle
manager it never actually publishes /map)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('home_robot')

    map_yaml_arg = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([pkg, 'maps', 'home.yaml']),
        description='Path to the saved map .yaml file',
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': LaunchConfiguration('map')}],
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([pkg, 'config', 'robot.rviz'])],
    )

    return LaunchDescription([
        map_yaml_arg,
        map_server_node,
        lifecycle_manager_node,
        rviz_node,
    ])
