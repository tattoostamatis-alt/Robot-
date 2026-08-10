#!/usr/bin/env python3
"""Find the rooms in a map by shape alone, and colour them.

    scripts/auto_rooms.py [map_name] [--apply] [--min-area 1.5] [--door 0.55]

Unlike make_room_mask.py this needs NO taught locations: it segments straight
off the occupancy grid, so it works on a map made five minutes ago, before
anything has been named. That is the point — after a remap you want to SEE the
rooms first and name them second.

How it works, and why this way:

A home's floor plan is wide rooms joined by narrow doorways. Take the distance
from every free cell to the nearest wall and that difference is explicit: deep
inside a room the distance is large, in a doorway it is at most half the door's
width. So threshold the distance transform above half a door — what survives is
one blob per room, with every doorway erased. Label those blobs, then grow them
back over the free space (nearest-seed, which is a watershed by distance) until
the rooms meet again in the doorways they were split at.

The alternatives are worse here. Plain connected components cannot work at all:
the whole floor is one component precisely because the doors connect it. Corner
or line detection needs walls that are straight and complete, and a SLAM map's
walls are neither — they are speckled, and they have gaps where a mirror or a
window ate the scan.

Two knobs, both in metres, both physical rather than tuning constants:

  --door   the widest opening that should still count as a doorway (0.90 m).
           Raise it if two rooms joined by a wide arch are being merged; lower
           it if one room is splitting in two at a narrow point.
  --min-area  discard rooms smaller than this (2.0 m^2), which is what stops
           every closet and every noise speckle becoming a "room".

Output goes to maps/<map>_autorooms.png plus a _preview.png you can actually
look at. Nothing is installed until you pass --apply, and even then the old
mask is kept as maps/<map>_room_mask.png.bak — the mask decides what the robot
CALLS each room in THAT map (situational_awareness, room_markers, the
dashboard tint all read the file scoped to the active map), so overwriting it
silently would change the robot's answers with no way back.

Room files are per-map (maps/<map>_room_mask.png + maps/<map>_room_colors.yaml,
see home_robot/room_files.py) — switching maps in the dashboard's Settings tab
switches rooms with it. You rarely need this CLI at all any more: the Χάρτες
tab's "click inside a room" tool runs this same segmentation on demand and
lets you name/colour a room without SSH.
"""
import argparse
import os
import sys

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.expanduser('~/robot_ws/src/home_robot'))
from home_robot.room_segment import PALETTE, colourise, segment  # noqa: E402
from home_robot import room_files  # noqa: E402

PKG = os.path.expanduser('~/robot_ws/src/home_robot')


def load_map(name):
    meta_path = f'{PKG}/maps/{name}.yaml'
    if not os.path.exists(meta_path):
        sys.exit(f'no such map: {meta_path}')
    meta = yaml.safe_load(open(meta_path))
    pgm = meta.get('image', f'{name}.pgm')
    if not os.path.isabs(pgm):
        pgm = os.path.join(f'{PKG}/maps', os.path.basename(pgm))
    img = np.array(Image.open(pgm))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img, float(meta['resolution'])


def preview(img, rgba):
    """The map with the rooms washed over it, for eyeballing."""
    base = np.dstack([img] * 3).astype(np.float32)
    tint = rgba[:, :, :3].astype(np.float32)
    m = rgba[:, :, 3] > 0
    base[m] = base[m] * 0.55 + tint[m] * 0.45
    return base.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map', nargs='?', default='malou2')
    # 0.90 / 2.0 m² were chosen by running this over maps/malou2.pgm (the flat
    # it actually drives) and looking at the preview: they split it into the
    # rooms a person would draw — two bedrooms, kitchen, living room, hall.
    # 0.55 merged the whole flat into one region; above ~1.1 the living room
    # started shedding slivers into the hall.
    ap.add_argument('--door', type=float, default=0.90,
                    help='widest opening still treated as a doorway, metres')
    ap.add_argument('--min-area', type=float, default=2.0,
                    help='smallest thing that counts as a room, m^2')
    ap.add_argument('--apply', action='store_true',
                    help='install as maps/<map>_room_mask.png (keeps a .bak)')
    args = ap.parse_args()

    img, res = load_map(args.map)
    labels, count = segment(img, res, args.door, args.min_area)
    print(f'{args.map}: {img.shape[1]}x{img.shape[0]} @ {res} m/px '
          f'-> {count} rooms')
    if not count:
        sys.exit('found no rooms — try a smaller --door or --min-area')

    cell_area = res * res
    for i in range(1, count + 1):
        area = (labels == i).sum() * cell_area
        r, g, b = PALETTE[(i - 1) % len(PALETTE)]
        print(f'  room {i}: {area:5.1f} m²   rgb({r},{g},{b})')

    rgba = colourise(labels, count)
    out = f'{PKG}/maps/{args.map}_autorooms.png'
    prev = f'{PKG}/maps/{args.map}_autorooms_preview.png'
    Image.fromarray(rgba).save(out)
    Image.fromarray(preview(img, rgba)).save(prev)
    print(f'wrote {out}\n      {prev}')

    # Names are the one thing shape cannot tell you. Emit placeholders so the
    # file is directly editable: rename the keys, keep the colours.
    colours = {f'room{i}': list(PALETTE[(i - 1) % len(PALETTE)])
               for i in range(1, count + 1)}
    ypath = f'{PKG}/maps/{args.map}_autorooms.yaml'
    with open(ypath, 'w') as f:
        yaml.safe_dump(colours, f, allow_unicode=True)
    print(f'      {ypath}  (rename room1/room2/... to the real names)')

    if args.apply:
        mask, colours_path = room_files.paths_for(args.map)
        if os.path.exists(mask):
            os.replace(mask, mask + '.bak')
            print(f'backed up {mask} -> {os.path.basename(mask)}.bak')
        Image.fromarray(rgba).save(mask)
        with open(colours_path, 'w') as f:
            yaml.safe_dump(colours, f, allow_unicode=True)
        print(f'installed as {mask}\n           + {colours_path}')
        print(f'‼️  the names are room1, room2, … — edit {colours_path} '
              'before the robot has to say them out loud (or rename/recolour '
              'them from the dashboard\'s Χάρτες tab instead)')


if __name__ == '__main__':
    main()
