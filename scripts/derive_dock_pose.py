#!/usr/bin/env python3
"""Derive the dock and dock_staging poses from the calibrated AprilTag.

    scripts/derive_dock_pose.py [map_name] [--write]

The station sits directly under the tag, so the tag's calibration fixes the
station too — no driving the robot onto the base and recording a pose, which
is how `dock` came to be 0.96 m off on the malou map. Run this after every
remap, once the tag has been recalibrated against the new map
(`ros2 service call /apriltag_relocalizer/calibrate_tag std_srvs/srv/Empty`).

Without --write it only prints, so you can sanity-check the numbers against
the map first.
"""
import argparse
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.expanduser('~/robot_ws/src/home_robot'))
from home_robot import dock_geometry as dg   # noqa: E402

PKG = os.path.expanduser('~/robot_ws/src/home_robot')
CALIB = os.path.expanduser('~/.ros/saloni_tag_map_pose.yaml')

ap = argparse.ArgumentParser()
ap.add_argument('map_name', nargs='?', default='malou')
ap.add_argument('--write', action='store_true',
                help='update config/locations.yaml in place')
ap.add_argument('--seat-offset', type=float, default=dg.DEFAULT_SEAT_OFFSET,
                help='wall plane to robot centre when parked, metres')
args = ap.parse_args()

if not os.path.exists(CALIB):
    sys.exit(f'No tag calibration at {CALIB} — calibrate the tag first.')
tag = yaml.safe_load(open(CALIB))

meta = yaml.safe_load(open(f'{PKG}/maps/{args.map_name}.yaml'))
res = meta['resolution']
ox, oy = meta['origin'][:2]
img = np.array(Image.open(f'{PKG}/maps/{args.map_name}.pgm'))
h, w = img.shape
# Unknown counts as free here on purpose: the corner behind the base is poorly
# observed, and treating it as blocked would push staging needlessly far out.
dist = ndimage.distance_transform_edt(img >= 65) * res


def clearance(x, y):
    c, r = int((x - ox) / res), h - 1 - int((y - oy) / res)
    if not (0 <= r < h and 0 <= c < w):
        return -1.0
    return float(dist[r, c])


normal = dg.tag_normal_yaw(tag['qx'], tag['qy'], tag['qz'], tag['qw'])
sx, sy, syaw = dg.seated_pose(tag['x'], tag['y'], normal, args.seat_offset)

print(f'map:  {args.map_name}')
print(f'tag:  x={tag["x"]:+.3f} y={tag["y"]:+.3f} z={tag["z"]:+.3f} '
      f'(height {tag["z"] * 100:.0f} cm)')
print(f'      outward normal {math.degrees(normal):+.1f} deg')
print()
print(f'station (floor under the tag): x={tag["x"]:+.3f} y={tag["y"]:+.3f}')
print(f'seated  (robot centre parked): x={sx:+.3f} y={sy:+.3f} '
      f'yaw={math.degrees(syaw):+.1f} deg   clearance={clearance(sx, sy):.3f}')
print()

staging = dg.find_staging(tag['x'], tag['y'], normal, clearance)
if staging is None:
    sys.exit('No staging pose within reach — the station is walled in on this '
             'map. Move the base, or remap that corner.')
gx, gy, gyaw, gd, gc = staging
print(f'staging (Nav2 goal):           x={gx:+.3f} y={gy:+.3f} '
      f'yaw={math.degrees(gyaw):+.1f} deg')
print(f'      {gd:.2f} m from the station, clearance {gc:.3f} m, '
      f'{math.degrees(dg.off_axis(tag["x"], tag["y"], normal, gx, gy)):.1f} deg off-axis')
print(f'      IR homing then covers {gd - args.seat_offset:.2f} m')
print()

print('corridor from staging in to the base (Nav2 inflation 0.30, robot 0.175):')
steps = int((gd - args.seat_offset) / 0.1) + 1
for i in range(steps + 1):
    d = gd - i * 0.1
    if d < args.seat_offset:
        d = args.seat_offset
    x, y, _ = dg.approach_point(tag['x'], tag['y'], normal, d)
    c = clearance(x, y)
    print(f'  {d:4.2f} m out: clearance {c:.3f}  '
          f'{"robot fits" if c >= 0.175 else "TIGHT"}, '
          f'{"Nav2 ok" if c >= 0.30 else "Nav2 blocked"}')
    if d <= args.seat_offset:
        break

locs_path = f'{PKG}/config/locations.yaml'
locs = yaml.safe_load(open(locs_path))
old = locs.get('dock')
if old:
    moved = math.hypot(old['x'] - sx, old['y'] - sy)
    print()
    print(f'existing dock in locations.yaml: x={old["x"]:+.3f} y={old["y"]:+.3f} '
          f'yaw={math.degrees(old["yaw"]):+.1f} deg  ({moved:.3f} m from derived)')

if not args.write:
    print()
    print('(dry run — pass --write to update config/locations.yaml)')
    sys.exit(0)

locs['dock'] = {'x': round(sx, 3), 'y': round(sy, 3), 'yaw': round(syaw, 3)}
locs['dock_staging'] = {'x': round(gx, 3), 'y': round(gy, 3),
                        'yaw': round(gyaw, 3)}
header = ('# Named navigation goals (map frame).\n'
          '#\n'
          '# dock / dock_staging are GENERATED — scripts/derive_dock_pose.py.\n'
          '# They come from the AprilTag calibration, since the station sits\n'
          '# under the tag; re-run it after a remap instead of re-teaching them.\n'
          '# `dock` is the seated pose and is NOT plannable (the base is an\n'
          '# obstacle) — navigate to `dock_staging` and let IR homing finish.\n')
with open(locs_path, 'w') as f:
    f.write(header)
    yaml.safe_dump(locs, f, default_flow_style=False, sort_keys=True,
                   allow_unicode=True)
print()
print(f'wrote {locs_path}')
