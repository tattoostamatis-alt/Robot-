#!/usr/bin/env python3
"""record_location.py — Save the robot's current map-frame pose as a named goal.

Drive the robot to the spot (teleop) with localization running
(localize.launch.py → AMCL publishing map→odom), then:

    ros2 run home_robot record_location.py kouzina
    ros2 run home_robot record_location.py domatio tou max   # spaces OK
    ros2 run home_robot record_location.py --list

Or teach rooms by clicking on the map in RViz ("Publish Point" tool),
no robot/localization needed — only a map view (view_map.launch.py):

    ros2 run home_robot record_location.py --click            # all rooms in --list order
    ros2 run home_robot record_location.py --click kouzina    # just one
    ros2 run home_robot record_location.py --click kouzina,saloni   # a subset, in order

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
    # ‼️ Atomic: write a temp file, fsync it, then rename over the target.
    # This machine loses power abruptly (8 unclean shutdowns in 3 days as of
    # 2026-08-01), and a truncate-then-write leaves a half-file on disk if the
    # cut lands in the middle. object_memory_node has done it this way all
    # along; these had not.
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('# Named navigation goals (map frame).\n')
        yaml.safe_dump(locations, f, allow_unicode=True, sort_keys=True,
                       default_flow_style=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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


def _click_mode(names, path, locations):
    """Record each name from an RViz 'Publish Point' click on /clicked_point."""
    from geometry_msgs.msg import PointStamped

    rclpy.init()
    node = rclpy.create_node('record_location_click')
    clicked = []
    node.create_subscription(PointStamped, '/clicked_point',
                             lambda m: clicked.append(m), 10)

    print('Στο RViz διάλεξε το εργαλείο "Publish Point" και κλίκαρε με τη σειρά:')
    for i, name in enumerate(names, 1):
        print(f'  {i}. {name}')
    try:
        for name in names:
            print(f'\n🖱  Κλικ για: {name} ...', flush=True)
            while not clicked:
                rclpy.spin_once(node, timeout_sec=0.2)
            m = clicked.pop(0)
            if m.header.frame_id != 'map':
                print(f'   ⚠ το κλικ ήρθε σε frame "{m.header.frame_id}" (όχι map) '
                      '— άλλαξε το Fixed Frame σε map και ξαναδοκίμασε')
            old = locations.get(name) or {}
            locations[name] = {'x': round(m.point.x, 3),
                               'y': round(m.point.y, 3),
                               'yaw': old.get('yaw', 0.0)}
            _save_locations(path, locations)  # save per click — crash-safe
            print(f'   ✔ {name}: x={m.point.x:.3f} y={m.point.y:.3f}')
    except KeyboardInterrupt:
        print('\nΔιακόπηκε — ό,τι κλικαρίστηκε ως εδώ έχει σωθεί.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f'\nΓράφτηκε στο {path}')


def main():
    args = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    path = _locations_path()
    locations = _load_locations(path)

    if args and args[0] == '--click':
        rest = args[1:]
        # One name may contain spaces ("domatio tou max"), so join first and
        # split on commas — that way a comma-separated list teaches several
        # rooms in one run without clobbering the ones left out (e.g. dock).
        if rest:
            names = [n.strip() for n in ' '.join(rest).split(',') if n.strip()]
        else:
            names = sorted(locations)
        if not names:
            print('Το locations.yaml είναι άδειο — δώσε όνομα: --click <όνομα>')
            sys.exit(1)
        _click_mode(names, path, locations)
        sys.exit(0)

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
