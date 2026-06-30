#!/usr/bin/env python3
"""Send the robot to a named room from locations.yaml via Nav2 NavigateToPose."""

import math
import sys
import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory


def load_locations():
    path = get_package_share_directory('home_robot') + '/config/locations.yaml'
    with open(path) as f:
        return yaml.safe_load(f) or {}


class GotoRoom(Node):
    def __init__(self, room: str, pose: dict):
        super().__init__('goto_room')
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._room = room
        self._pose = pose

    def send_goal(self):
        self.get_logger().info(f'Αναμονή για Nav2...')
        self._client.wait_for_server()

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(self._pose['x'])
        goal.pose.pose.position.y = float(self._pose['y'])
        yaw = float(self._pose.get('yaw', 0.0))
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)

        self.get_logger().info(
            f'Πορεία προς "{self._room}" '
            f'(x={self._pose["x"]:.2f}, y={self._pose["y"]:.2f})'
        )
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_accepted)

    def _on_goal_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal απορρίφθηκε από Nav2')
            rclpy.shutdown()
            return
        self.get_logger().info('Goal έγινε δεκτό — πορεύομαι...')
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        result = future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info(f'Έφτασα στο "{self._room}"!')
        else:
            self.get_logger().warn(f'Απέτυχε (status={result.status})')
        rclpy.shutdown()


def main():
    if len(sys.argv) < 2:
        locs = load_locations()
        print('Χρήση: goto_room.py <δωμάτιο>')
        print('Διαθέσιμα δωμάτια:')
        for name, p in locs.items():
            print(f'  {name}  (x={p["x"]:.2f}, y={p["y"]:.2f})')
        sys.exit(0)

    room = ' '.join(sys.argv[1:])
    locs = load_locations()

    if room not in locs:
        # fuzzy match (contains)
        matches = [n for n in locs if room.lower() in n.lower()]
        if len(matches) == 1:
            room = matches[0]
        elif len(matches) > 1:
            print(f'Ασαφές: {matches}')
            sys.exit(1)
        else:
            print(f'Άγνωστο δωμάτιο: "{room}"')
            print('Διαθέσιμα:', list(locs.keys()))
            sys.exit(1)

    rclpy.init()
    node = GotoRoom(room, locs[room])
    node.send_goal()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
