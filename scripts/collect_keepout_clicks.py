#!/usr/bin/env python3
"""Define keepout zones by clicking on the map in RViz (Publish Point tool).

Every TWO clicks = one rectangular zone (two opposite corners). Open a map
view first (ros2 launch home_robot view_map.launch.py), then:

    ros2 run home_robot collect_keepout_clicks.py            # auto names zone_1..
    ros2 run home_robot collect_keepout_clicks.py kanapes trapezi

With names given, it exits after 2 clicks per name. Without, it keeps
collecting pairs and finishes after 45 s with no new clicks. Writes
maps/<active-map>_keepout_zones.yaml (overwrites that map's zones list — the
active map is read from /map_server, same as every other per-map tool; see
home_robot/keepout_files.py), then run:

    python3 scripts/draw_keepout.py
"""
import os
import sys
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rcl_interfaces.srv import GetParameters

sys.path.insert(0, os.path.expanduser('~/robot_ws/src/home_robot'))
from home_robot import keepout_files  # noqa: E402

IDLE_EXIT_S = 45.0
MIN_SIZE_M = 0.10  # a zone thinner than this is probably a misclick


def _active_map(node) -> str:
    cli = node.create_client(GetParameters, '/map_server/get_parameters')
    if not cli.wait_for_service(timeout_sec=5.0):
        sys.exit('map_server not reachable — is the robot up and localized?')
    req = GetParameters.Request(names=['yaml_filename'])
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    if not fut.done() or not fut.result().values:
        sys.exit('could not read /map_server yaml_filename')
    path = fut.result().values[0].string_value
    return os.path.splitext(os.path.basename(path))[0]


def main():
    names = rclpy.utilities.remove_ros_args(sys.argv)[1:]

    rclpy.init()
    node = rclpy.create_node('collect_keepout_clicks')
    map_name = _active_map(node)
    existing = keepout_files.load_zones(map_name)
    clicks = []
    node.create_subscription(PointStamped, '/clicked_point',
                             lambda m: clicks.append(m), 10)

    print(f'Ενεργός χάρτης: {map_name} ({len(existing)} υπάρχουσες ζώνες)')
    print('Publish Point στο RViz: 2 κλικ (απέναντι γωνίες) για κάθε ζώνη.')
    if names:
        print(f'Ζώνες: {", ".join(names)} — τέλος μετά από {2*len(names)} κλικ.')
    else:
        print(f'Ελεύθερη λειτουργία — τέλος μετά από {IDLE_EXIT_S:.0f}s χωρίς κλικ.')

    zones = dict(existing)   # additive: re-running keeps what's already there
    corner = None
    zone_i = 0
    last_click_t = time.monotonic()
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            if clicks:
                m = clicks.pop(0)
                last_click_t = now
                if m.header.frame_id != 'map':
                    print(f'  ⚠ κλικ σε frame "{m.header.frame_id}" — Fixed Frame=map και ξανά')
                    continue
                if corner is None:
                    corner = (m.point.x, m.point.y)
                    print(f'  γωνία 1: ({corner[0]:.2f}, {corner[1]:.2f}) — κλίκαρε την απέναντι')
                else:
                    x1, y1 = corner
                    x2, y2 = m.point.x, m.point.y
                    corner = None
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    if w < MIN_SIZE_M or h < MIN_SIZE_M:
                        print(f'  ⚠ ζώνη {w:.2f}×{h:.2f}m πολύ μικρή — μάλλον misclick, αγνοήθηκε· ξανά από γωνία 1')
                        continue
                    zone_i += 1
                    name = names[zone_i - 1] if zone_i <= len(names) else f'zone_{zone_i}'
                    zones[name] = {'shape': 'rect',
                                   'x': round((x1 + x2) / 2, 3),
                                   'y': round((y1 + y2) / 2, 3),
                                   'width': round(w, 2),
                                   'height': round(h, 2)}
                    print(f'  ✔ {name}: κέντρο ({zones[name]["x"]}, {zones[name]["y"]}) '
                          f'{zones[name]["width"]}×{zones[name]["height"]}m')
                    if names and zone_i >= len(names):
                        break
            elif not names and zones and now - last_click_t > IDLE_EXIT_S:
                break  # idle — a dangling single corner (if any) is dropped below
            elif not names and now - last_click_t > 10 * 60:
                print('10 λεπτά χωρίς κανένα κλικ — τα παρατάω.')
                break
    except KeyboardInterrupt:
        print('\nΔιακόπηκε.')
    finally:
        node.destroy_node()
        # try_shutdown, NOT shutdown: on SIGINT rclpy has already shut the
        # context down, and a second shutdown() raises — which would skip
        # the file write below and lose every clicked zone.
        rclpy.try_shutdown()

    if corner is not None:
        print('⚠ έμεινε μονό κλικ χωρίς ταίρι — αγνοήθηκε.')
    if not zones:
        print('Καμία ζώνη — το keepout_zones.yaml δεν αλλάζει.')
        sys.exit(1)

    keepout_files.save_zones(map_name, zones)
    path = keepout_files.zones_path(map_name)
    print(f'{len(zones)} ζώνες γράφτηκαν στο {path}')
    print('Τώρα: python3 scripts/draw_keepout.py '
         '(ή ενεργοποίησε από το dashboard\'s Χάρτες tab)')


if __name__ == '__main__':
    main()
