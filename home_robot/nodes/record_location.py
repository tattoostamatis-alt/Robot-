#!/usr/bin/env python3
"""record_location.py — Save the robot's current map-frame pose as a named goal.

Drive the robot to the spot (teleop) with localization running
(localize.launch.py → AMCL publishing map→odom), then:

    ros2 run home_robot record_location.py kouzina
    ros2 run home_robot record_location.py domatio tou max   # spaces OK
    ros2 run home_robot record_location.py --list

Writes config/locations.yaml through the install symlink, so the source file
in the repo is updated in place — commit it when all rooms are re-taught.
Restart nodes that cache locations at startup (room_markers, mission_executor,
situational_awareness) to pick up the new poses.
"""

import math
import os
import sys

import yaml
import rclpy
import tf2_ros
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from ament_index_python.packages import get_package_share_directory

# AMCL x/y variance above this → pose is probably not converged; warn loudly.
_COV_WARN = 0.10  # m² (σ ≈ 0.32 m)


def _locations_path() -> str:
    share = os.path.join(get_package_share_directory('home_robot'),
                         'config', 'locations.yaml')
    return os.path.realpath(share)  # follow the symlink into src/ so git sees it


def _load_locations(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _save_locations(path: str, locations: dict):
    with open(path, 'w') as f:
        f.write('# Named navigation goals (map frame).\n')
        yaml.safe_dump(locations, f, allow_unicode=True, sort_keys=True,
                       default_flow_style=False)


class RecordLocation(Node):
    def __init__(self):
        super().__init__('record_location')
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._amcl_cov = None
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                 self._on_amcl, 10)

    def _on_amcl(self, msg: PoseWithCovarianceStamped):
        cov = msg.pose.covariance
        self._amcl_cov = max(cov[0], cov[7])  # var(x), var(y)

    def get_map_pose(self, timeout_s: float = 5.0):
        """Return (x, y, yaw) of base_link in the map frame, or None."""
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                t = self._tf_buffer.lookup_transform('map', 'base_link',
                                                     rclpy.time.Time())
            except Exception:
                continue
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return t.transform.translation.x, t.transform.translation.y, yaw
        return None


def main():
    args = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    path = _locations_path()
    locations = _load_locations(path)

    if not args or args[0] in ('--list', '-l'):
        print(f'locations.yaml: {path}')
        for name, p in sorted(locations.items()):
            print(f'  {name:22s} x={p["x"]:7.3f}  y={p["y"]:7.3f}  '
                  f'yaw={p.get("yaw", 0.0):6.3f}')
        if not args:
            print('\nΧρήση: record_location.py <όνομα>   (αποθηκεύει την τρέχουσα θέση)')
        sys.exit(0)

    name = ' '.join(args)

    rclpy.init()
    node = RecordLocation()
    pose = node.get_map_pose()
    if pose is None:
        node.get_logger().error(
            'Δεν υπάρχει TF map→base_link — τρέχει το localize.launch.py; '
            'Έχει γίνει initial pose στο AMCL;')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    x, y, yaw = pose
    if node._amcl_cov is not None and node._amcl_cov > _COV_WARN:
        node.get_logger().warning(
            f'Το AMCL δεν έχει συγκλίνει καλά (var={node._amcl_cov:.2f} m²) — '
            'η θέση ίσως είναι ανακριβής. Κουνήσου λίγο και ξαναδοκίμασε.')

    old = locations.get(name)
    locations[name] = {'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 3)}
    _save_locations(path, locations)

    if old:
        moved = math.hypot(x - old['x'], y - old['y'])
        print(f'Ενημερώθηκε "{name}": x={x:.3f} y={y:.3f} yaw={yaw:.3f} '
              f'(παλιά θέση απείχε {moved:.2f} m)')
    else:
        print(f'Νέο location "{name}": x={x:.3f} y={y:.3f} yaw={yaw:.3f}')
    print(f'Γράφτηκε στο {path}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
