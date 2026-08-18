"""Eye-on-base hand-eye calibration: body-mounted D435 <-> RoArm-M3.

Computes the transform between the arm's URDF root ('world', published by
arm_moveit.launch.py's robot_state_publisher) and the camera's optical frame
('camera_color_optical_frame', published by realsense2_camera_node), using the
wrist-mounted gripper_tag (AprilTag id=1, config/apriltag.yaml) as the tracked
marker — apriltag_node already publishes its tf live, so no aruco_ros needed.

Run ALONGSIDE bringup (camera + apriltag_node) and arm_moveit.launch.py (arm +
MoveIt), on the user's display:
    DISPLAY=:0 XAUTHORITY=<mutter xauth> QT_QPA_PLATFORM=xcb \
        ros2 launch home_robot handeye_calibrate.launch.py

Move the arm via the MoveIt RViz plugin (drag hand_tcp -> Plan -> Execute) to
poses where gripper_tag stays visible to the camera — per
project_robot_camera_fov_vs_arm_reach, that means keeping the gripper roughly
0.20-0.40m above the floor, varied in x/y/orientation for a good sample spread.
Click "take sample" in the rqt calibrator GUI at each pose (aim for >=10-15).
"Compute" then "Save" writes the result to ~/.ros/easy_handeye2/calibrations/.

After saving, see home_robot/scripts/apply_handeye_calibration.py to fold the
result into bringup.launch.py's base_link->arm_base static transform.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    calibrate_launch = os.path.join(
        get_package_share_directory('easy_handeye2'), 'launch', 'calibrate.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(calibrate_launch),
            launch_arguments={
                'name': 'd435_arm',
                'calibration_type': 'eye_on_base',
                'robot_base_frame': 'world',
                'robot_effector_frame': 'hand_tcp',
                'tracking_base_frame': 'camera_color_optical_frame',
                'tracking_marker_frame': 'gripper_tag',
            }.items(),
        ),
    ])
