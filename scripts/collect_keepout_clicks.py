#!/usr/bin/env python3
"""Define keepout zones by clicking on the map in RViz (Publish Point tool).

Every TWO clicks = one rectangular zone (two opposite corners). Open a map
view first (ros2 launch home_robot view_map.launch.py), then:

    ros2 run home_robot collect_keepout_clicks.py            # auto names zone_1..
    ros2 run home_robot collect_keepout_clicks.py kanapes trapezi

With names given, it exits after 2 clicks per name. Without, it keeps
collecting pairs and finishes after 45 s with no new clicks. Writes
config/keepout_zones.yaml (overwrites the zones list), then run:

    python3 scripts/draw_keepout.py
"""
import os
import sys
import time

import yaml
import rclpy
from geometry_msgs.msg import PointStamped
from ament_index_python.packages import get_package_share_directory

IDLE_EXIT_S = 45.0
MIN_SIZE_M = 0.10  # a zone thinner than this is probably a misclick


def _zones_path() -> str:
    share = os.path.join(get_package_share_directory('home_robot'),
                         'config', 'keepout_zones.yaml')
    return os.path.realpath(share)  # through the symlink into src/


def main():
    names = rclpy.utilities.remove_ros_args(sys.argv)[1:]

    rclpy.init()
    node = rclpy.create_node('collect_keepout_clicks')
    clicks = []
    node.create_subscription(PointStamped, '/clicked_point',
                             lambda m: clicks.append(m), 10)

    print('Publish Point στο RViz: 2 κλικ (απέναντι γωνίες) για κάθε ζώνη.')
    if names:
        print(f'Ζώνες: {", ".join(names)} — τέλος μετά από {2*len(names)} κλικ.')
    else:
        print(f'Ελεύθερη λειτουργία — τέλος μετά από {IDLE_EXIT_S:.0f}s χωρίς κλικ.')

    zones = {}
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

    path = _zones_path()
    with open(path, 'w') as f:
        f.write('# Keepout zones — areas the robot must NOT navigate (map frame,\n'
                '# clicked on the ACTIVE map). Recreate after a remap:\n'
                '#   ros2 run home_robot collect_keepout_clicks.py   (RViz Publish Point)\n'
                '#   python3 scripts/draw_keepout.py\n'
                '# Shapes: rect (x,y=centre, width,height) | circle (x,y,radius), metres.\n'
                '# To USE the zones: bringup with use_keepout:=true AND set the two\n'
                '# keepout_filter layers to enabled: True in nav2_params.yaml.\n')
        yaml.safe_dump({'zones': zones}, f, allow_unicode=True, sort_keys=False,
                       default_flow_style=False)
    print(f'{len(zones)} ζώνες γράφτηκαν στο {path}')
    print('Τώρα: python3 scripts/draw_keepout.py')


if __name__ == '__main__':
    main()
