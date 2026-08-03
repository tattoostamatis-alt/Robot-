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
mask is kept as room_mask.png.bak — the mask decides what the robot CALLS each
room (situational_awareness, room_markers, the dashboard tint all read it), so
overwriting it silently would change the robot's answers with no way back.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

PKG = os.path.expanduser('~/robot_ws/src/home_robot')

# Reused for room 1, 2, 3... Distinct hues rather than a gradient, because
# these are labels and not a scale. Kept clear of the greys the map is drawn
# in so a room is never confused with a wall.
PALETTE = [
    (204, 68, 255), (255, 204, 0), (68, 136, 255), (68, 204, 102),
    (255, 68, 68), (0, 214, 214), (255, 136, 0), (170, 102, 255),
    (102, 204, 170), (255, 102, 170), (140, 180, 60), (90, 150, 255),
]


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


def segment(img, res, door_m, min_area_m2, close_px=2):
    """Label the rooms. Returns (labels, count); 0 means 'not a room'."""
    free = img >= 250

    # Speckle inside a room would punch holes in the distance transform and
    # split one room into several. A binary close is enough; the walls are
    # thick relative to a couple of pixels.
    if close_px:
        free = ndimage.binary_closing(free, np.ones((close_px, close_px), bool))

    # Distance (in metres) from each free cell to the nearest wall.
    dist = ndimage.distance_transform_edt(free) * res

    # A doorway is at most `door_m` wide, so its centre line sits at most
    # door_m/2 from a wall. Anything deeper than that is room interior.
    cores = dist > (door_m / 2.0)
    lab, n = ndimage.label(cores)
    if n == 0:
        return np.zeros_like(img, np.int32), 0

    # Grow the seeds back over all free space: every free cell takes the label
    # of the nearest seed. This is the watershed, done with one EDT rather than
    # an explicit flood, and it puts the boundary in the doorway — the far side
    # of a door is nearer the room beyond it than the room behind.
    _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    grown = lab[iy, ix]
    grown[~free] = 0

    # Only NOW drop the small ones. Filtering the seeds instead measured the
    # cores — which are much smaller than the rooms they become — so a bigger
    # --min-area produced FEWER, larger rooms in a way that made no sense from
    # the outside: 3.0 m² merged the whole flat into one, while 1.5 m² found
    # two. Measure the thing the user is actually thinking about.
    cell_area = res * res
    out = np.zeros_like(grown)
    next_id = 1
    for i in range(1, n + 1):
        m = grown == i
        if m.sum() * cell_area >= min_area_m2:
            out[m] = next_id
            next_id += 1
    return out, next_id - 1


def colourise(labels, count):
    """RGBA image: room colour where labelled, fully transparent elsewhere."""
    h, w = labels.shape
    out = np.zeros((h, w, 4), np.uint8)
    for i in range(1, count + 1):
        r, g, b = PALETTE[(i - 1) % len(PALETTE)]
        m = labels == i
        out[m] = (r, g, b, 255)
    return out


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
                    help='install as maps/room_mask.png (keeps a .bak)')
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
        mask = f'{PKG}/maps/room_mask.png'
        if os.path.exists(mask):
            shutil.copy2(mask, mask + '.bak')
            print(f'backed up {mask} -> room_mask.png.bak')
        shutil.copy2(out, mask)
        shutil.copy2(ypath, f'{PKG}/maps/room_colors.yaml')
        print('installed as maps/room_mask.png + maps/room_colors.yaml')
        print('‼️  the names are room1, room2, … — edit maps/room_colors.yaml '
              'before the robot has to say them out loud')


if __name__ == '__main__':
    main()
