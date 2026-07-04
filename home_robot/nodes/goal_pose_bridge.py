#!/usr/bin/env python3
"""Bridge /goal_pose to Nav2 NavigateToPose action."""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


class GoalPoseBridgeNode(Node):
    def __init__(self):
        super().__init__('goal_pose_bridge')
        self._goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._on_goal_pose,
            10,
        )
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._active_goal_handle = None
        self.get_logger().info('GoalPoseBridge ready — forwarding /goal_pose to Nav2')

    def _on_goal_pose(self, msg: PoseStamped):
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 navigate_to_pose action server unavailable')
            return

        if self._active_goal_handle is not None:
            self.get_logger().info('Cancelling previous Nav2 goal before sending new one')
            self._active_goal_handle.cancel_goal_async()
            self._active_goal_handle = None

        goal = NavigateToPose.Goal()
        goal.pose = msg

        self.get_logger().info(
            f'Forwarding nav goal: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}'
        )

        send_future = self._nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_accepted)

    def _on_goal_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 goal rejected')
            return

        self._active_goal_handle = goal_handle
        self.get_logger().info('Nav2 goal accepted')
        goal_handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        result = future.result()
        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Nav2 goal succeeded')
        else:
            self.get_logger().warn(f'Nav2 goal finished with status={status}')
        self._active_goal_handle = None


def main():
    rclpy.init()
    node = GoalPoseBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
