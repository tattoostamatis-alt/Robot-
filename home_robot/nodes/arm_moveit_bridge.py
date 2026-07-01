#!/usr/bin/env python3
"""Bridge between MoveIt and the existing arm_driver.py.

MoveIt (roarm_moveit config) drives the arm through the
moveit_simple_controller_manager, which expects two action servers:

  * hand_controller    : control_msgs/FollowJointTrajectory
                         at  hand_controller/follow_joint_trajectory
  * gripper_controller : control_msgs/GripperCommand
                         at  gripper_controller/gripper_cmd

Rather than run a second driver on /dev/arm (Waveshare's roarm_driver, which
would fight arm_driver.py for the serial port), this node answers those two
actions and forwards the motion onto the topics arm_driver already listens to:

  /arm/joint_cmd   (sensor_msgs/JointState)  -> T:102 all-joint command
  /arm/gripper_cmd (std_msgs/Float32)        -> T:106 gripper command

It also relays /arm/joint_states (arm_driver's own feedback, named
base/shoulder/.../hand) back onto /joint_states under the URDF joint names, so
robot_state_publisher + MoveIt's current-state monitor see the live pose and the
RViz model mirrors the real arm.

Name/scale reconciliation
  URDF arm joints  base_link_to_link1 .. link4_to_link5  <-> base/shoulder/elbow/wrist/roll
  URDF gripper     link5_to_gripper_link [0=closed .. 1.5=open]
  arm_driver hand  [1.08=closed .. 3.14=open]
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from control_msgs.action import FollowJointTrajectory, GripperCommand

# URDF (MoveIt) arm joints, in the order the hand_controller lists them, paired
# with the joint names arm_driver.py / the real hardware use.
ARM_URDF_TO_DRIVER = {
    'base_link_to_link1': 'base',
    'link1_to_link2':     'shoulder',
    'link2_to_link3':     'elbow',
    'link3_to_link4':     'wrist',
    'link4_to_link5':     'roll',
}
DRIVER_TO_ARM_URDF = {v: k for k, v in ARM_URDF_TO_DRIVER.items()}

GRIPPER_URDF_JOINT = 'link5_to_gripper_link'
# URDF gripper range (from the RoArm-M3 URDF) and arm_driver's 'hand' range.
GRIPPER_URDF_LO, GRIPPER_URDF_HI = 0.0, 1.5
HAND_LO, HAND_HI = 1.08, 3.14


def _lerp(v, in_lo, in_hi, out_lo, out_hi):
    if in_hi == in_lo:
        return out_lo
    t = (v - in_lo) / (in_hi - in_lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)


class ArmMoveItBridge(Node):
    def __init__(self):
        super().__init__('arm_moveit_bridge')

        self.joint_cmd_pub = self.create_publisher(JointState, 'arm/joint_cmd', 10)
        self.gripper_pub = self.create_publisher(Float32, 'arm/gripper_cmd', 10)
        # robot_state_publisher + MoveIt current-state monitor read /joint_states.
        self.joint_states_pub = self.create_publisher(JointState, 'joint_states', 10)

        cb = ReentrantCallbackGroup()

        self._traj_server = ActionServer(
            self, FollowJointTrajectory,
            'hand_controller/follow_joint_trajectory',
            execute_callback=self._execute_trajectory,
            goal_callback=lambda g: GoalResponse.ACCEPT,
            cancel_callback=lambda g: CancelResponse.ACCEPT,
            callback_group=cb)

        self._gripper_server = ActionServer(
            self, GripperCommand,
            'gripper_controller/gripper_cmd',
            execute_callback=self._execute_gripper,
            goal_callback=lambda g: GoalResponse.ACCEPT,
            cancel_callback=lambda g: CancelResponse.ACCEPT,
            callback_group=cb)

        self.create_subscription(
            JointState, 'arm/joint_states', self._relay_joint_states, 10,
            callback_group=cb)

        self.get_logger().info(
            'arm_moveit_bridge ready: hand_controller/follow_joint_trajectory + '
            'gripper_controller/gripper_cmd -> /arm/joint_cmd, /arm/gripper_cmd')

    # ------------------------------------------------------------------
    # /arm/joint_states (base/.../hand) -> /joint_states (URDF names)
    # ------------------------------------------------------------------
    def _relay_joint_states(self, msg: JointState):
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        names, positions = [], []
        pos_by_name = dict(zip(msg.name, msg.position))
        for driver_name, value in pos_by_name.items():
            if driver_name in DRIVER_TO_ARM_URDF:
                names.append(DRIVER_TO_ARM_URDF[driver_name])
                positions.append(value)
            elif driver_name == 'hand':
                names.append(GRIPPER_URDF_JOINT)
                positions.append(_lerp(value, HAND_LO, HAND_HI,
                                       GRIPPER_URDF_LO, GRIPPER_URDF_HI))
        if not names:
            return
        out.name = names
        out.position = positions
        self.joint_states_pub.publish(out)

    # ------------------------------------------------------------------
    # FollowJointTrajectory -> stream waypoints to /arm/joint_cmd
    # ------------------------------------------------------------------
    def _execute_trajectory(self, goal_handle):
        traj = goal_handle.request.trajectory
        urdf_names = list(traj.joint_names)
        self.get_logger().info(
            f'trajectory goal: {len(traj.points)} points, joints {urdf_names}')

        start = time.monotonic()
        for pt in traj.points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().warn('trajectory canceled')
                return FollowJointTrajectory.Result()

            # Pace to the point's time_from_start (best-effort; arm_driver's
            # T:102 moves at its own spd/acc, so this just avoids flooding).
            due = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            wait = due - (time.monotonic() - start)
            if wait > 0:
                time.sleep(min(wait, 5.0))

            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            names, positions = [], []
            for urdf_name, p in zip(urdf_names, pt.positions):
                if urdf_name in ARM_URDF_TO_DRIVER:
                    names.append(ARM_URDF_TO_DRIVER[urdf_name])
                    positions.append(float(p))
            if names:
                js.name = names
                js.position = positions
                self.joint_cmd_pub.publish(js)

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    # ------------------------------------------------------------------
    # GripperCommand -> /arm/gripper_cmd
    # ------------------------------------------------------------------
    def _execute_gripper(self, goal_handle):
        pos_urdf = goal_handle.request.command.position
        hand = _lerp(pos_urdf, GRIPPER_URDF_LO, GRIPPER_URDF_HI, HAND_LO, HAND_HI)
        self.get_logger().info(f'gripper goal: urdf={pos_urdf:.3f} -> hand={hand:.3f}')
        self.gripper_pub.publish(Float32(data=float(hand)))
        # No position feedback loop; assume it reaches the commanded opening.
        time.sleep(0.5)
        goal_handle.succeed()
        result = GripperCommand.Result()
        result.position = pos_urdf
        result.reached_goal = True
        result.stalled = False
        return result


def main():
    rclpy.init()
    node = ArmMoveItBridge()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
