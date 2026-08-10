#!/usr/bin/env python3
"""Rasterize a map's keepout zones onto config/keepout_mask.pgm/.yaml — the
FIXED path Nav2's filter_mask_server actually loads (see
launch/bringup.launch.py's yaml_filename override). Zones themselves are
per-map (maps/<map>_keepout_zones.yaml, home_robot/keepout_files.py, written
by scripts/collect_keepout_clicks.py or the dashboard's Χάρτες tab).

Usage:
    python3 scripts/draw_keepout.py [map_name]    # default: malou2

REGENERATE after a remap or after editing that map's zones — the mask is a
snapshot, not live. After running, restart the stack with use_keepout:=true
(the dashboard's Χάρτες tab does both steps at once).
"""
import os
import sys

sys.path.insert(0, os.path.expanduser('~/robot_ws/src/home_robot'))
from home_robot import keepout_files  # noqa: E402

PKG = os.path.expanduser('~/robot_ws/src/home_robot')
MAP_NAME = sys.argv[1] if len(sys.argv) > 1 else 'malou2'
MAP_YAML = os.path.join(PKG, 'maps', f'{MAP_NAME}.yaml')
MAP_PGM = os.path.join(PKG, 'maps', f'{MAP_NAME}.pgm')


def main():
    if not os.path.exists(MAP_YAML):
        sys.exit(f'no such map: {MAP_YAML}')
    from PIL import Image
    with Image.open(MAP_PGM) as im:
        width, height = im.size

    zones = keepout_files.load_zones(MAP_NAME)
    if not zones:
        print(f'no zones for "{MAP_NAME}" — clearing mask to all-free.')

    keepout_files.render_active_mask(MAP_NAME, MAP_YAML, (width, height))
    pgm_path, yaml_path = keepout_files.mask_paths()
    print(f'{pgm_path}\n{yaml_path}')
    print(f'{width}x{height}px, {len(zones)} zone(s): ' + ', '.join(sorted(zones)))
    print()
    print('Done. Restart with use_keepout:=true for it to take effect:')
    print('  robot stop && robot max use_keepout:=true')


if __name__ == '__main__':
    main()
