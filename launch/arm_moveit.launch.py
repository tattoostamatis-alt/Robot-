"""MoveIt 2 (drag-the-gripper) control for the RoArm-M3 on ROS 2 Jazzy.

Reuses Waveshare's roarm_moveit config (SRDF / joint limits / controllers),
with kinematics.yaml switched to KDL. Instead of ros2_control + a fake system,
motion is executed on the REAL arm through arm_moveit_bridge.py, which answers
MoveIt's FollowJointTrajectory / GripperCommand actions and forwards them to the
existing arm_driver.py topics. The bridge also relays arm_driver's feedback onto
/joint_states so the RViz model mirrors the physical arm.

Run this ALONGSIDE bringup (which already owns /dev/arm via arm_driver); do NOT
start a second arm_driver. Launch it on the user's screen with:
    DISPLAY=:0 XAUTHORITY=<mutter xauth> QT_QPA_PLATFORM=xcb \
        ros2 launch home_robot arm_moveit.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("roarm_m3", package_name="roarm_moveit")
        .robot_description(file_path="config/roarm_m3/roarm_m3.urdf.xacro")
        .robot_description_semantic(file_path="config/roarm_m3/roarm_m3.srdf")
        .robot_description_kinematics(file_path="config/roarm_m3/kinematics.yaml")
        .trajectory_execution(file_path="config/roarm_m3/moveit_controllers.yaml")
        .joint_limits(file_path="config/roarm_m3/joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Publishes TF for the arm links from /joint_states (fed by the bridge relay).
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description,
                    {"publish_frequency": 15.0}],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"publish_robot_description_semantic": True,
             "publish_planning_scene": True,
             "publish_geometry_updates": True,
             "publish_state_updates": True,
             "publish_transforms_updates": True},
        ],
    )

    # MoveIt <-> arm_driver bridge (action servers + /joint_states relay).
    arm_moveit_bridge = Node(
        package="home_robot",
        executable="arm_moveit_bridge.py",
        output="screen",
    )

    rviz_config = os.path.join(
        get_package_share_directory("roarm_moveit"), "rviz", "interact.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription([
        robot_state_publisher,
        move_group,
        arm_moveit_bridge,
        rviz,
    ])
