#!/usr/bin/env python3
"""Put fetchable objects into a Gazebo world, one per room.

map_to_world.py turns the occupancy map into walls; this adds the things the
robot is asked to go and get. Each object is a small coloured box placed in a
verified-free spot near a room's navigation waypoint, so the robot can park in
front of it without the approach pose landing inside a wall.

Writes two files:
  <world>            the world with the object models appended
  <world>.objects.json   ground truth: label -> {x, y, room}

The JSON is what the simulated perception publishes, standing in for YOLO. A
Gazebo camera would render these boxes perfectly well, but the detector would
report "box", not "κούπα" — labelling them is the whole point of the exercise
and ground truth is the honest way to do it. What is under test here is
navigation and the mission, not the object classifier.

Usage:
    sim_place_objects.py maps/malou2.yaml worlds/malou2.world
"""

import argparse
import json
import math
import os
import sys

import yaml
from PIL import Image

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# label -> RGB, so each object is visually distinct in the Gazebo GUI.
OBJECTS = [
    ('μπάλα',       'ball',   (0.85, 0.20, 0.20)),
    ('κούπα',       'mug',    (0.20, 0.55, 0.85)),
    ('βιβλίο',      'book',   (0.90, 0.70, 0.15)),
    ('τηλεκοντρόλ', 'remote', (0.30, 0.75, 0.35)),
    ('μπουκάλι',    'bottle', (0.65, 0.35, 0.80)),
    ('κλειδιά',     'keys',   (0.95, 0.50, 0.15)),
]

OBJ_SIZE = 0.12          # m — small enough to sit inside a doorway-width gap
OBJ_HEIGHT = 0.18


def load_grid(map_yaml):
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(map_yaml)), img_path)
    img = Image.open(img_path).convert('L')
    return meta, img


def free_checker(meta, img):
    """Return is_free(x, y, clearance) in map coordinates.

    Free means: known free space (not occupied, not unknown) for every cell
    within `clearance` metres. Unknown counts as blocked — an object dropped
    into unmapped space is an object the robot can never reach.
    """
    res = float(meta['resolution'])
    ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
    occ_thresh = float(meta.get('occupied_thresh', 0.65))
    free_thresh = float(meta.get('free_thresh', 0.196))
    negate = int(meta.get('negate', 0))
    w, h = img.size
    px = img.load()

    def occ_prob(col, row):
        val = px[col, row]
        return val / 255.0 if negate else (255 - val) / 255.0

    def is_free(x, y, clearance):
        cells = max(1, int(math.ceil(clearance / res)))
        col0 = int((x - ox) / res)
        row0 = int(h - 1 - (y - oy) / res)
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                c, r = col0 + dc, row0 + dr
                if not (0 <= c < w and 0 <= r < h):
                    return False
                p = occ_prob(c, r)
                if p > occ_thresh:      # wall
                    return False
                if p > free_thresh:     # unknown / grey
                    return False
        return True

    return is_free


def find_spot(is_free, cx, cy, clearance, min_r=0.5, max_r=2.0):
    """A free point near (cx, cy) but not on top of it.

    Spirals outward: the room waypoint itself is where the robot parks, so an
    object sitting exactly there would be trivially "found" without the
    approach ever being tested.
    """
    r = min_r
    while r <= max_r:
        for i in range(24):
            ang = i * math.pi / 12.0
            x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
            if is_free(x, y, clearance):
                return x, y
        r += 0.15
    return None


def model_sdf(name, x, y, rgb):
    r, g, b = rgb
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {OBJ_HEIGHT / 2:.4f} 0 0 0</pose>
      <link name="link">
        <!-- Visual/perception target only.  The fetch simulator models grasp
             state explicitly; a static collision here cannot be attached to
             the virtual gripper and behaves like an immovable wall, which can
             seal a narrow corridor before the robot reaches grasp range. -->
        <visual name="v">
          <geometry><box><size>{OBJ_SIZE} {OBJ_SIZE} {OBJ_HEIGHT}</size></box></geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map_yaml')
    ap.add_argument('world')
    ap.add_argument('--clearance', type=float, default=0.35,
                    help='m of free space required around an object')
    ap.add_argument('--exclude-room', action='append', default=[],
                    help='room to omit from this scenario (repeatable)')
    args = ap.parse_args()

    meta, img = load_grid(args.map_yaml)
    is_free = free_checker(meta, img)

    with open(os.path.join(SRC, 'config', 'locations.yaml')) as f:
        locs = yaml.safe_load(f) or {}
    rooms = {k: (float(v['x']), float(v['y']))
             for k, v in locs.items()
             if (isinstance(v, dict) and 'x' in v and not k.startswith('dock')
                 and k not in set(args.exclude_room))}

    with open(args.world) as f:
        world = f.read()
    if '</world>' not in world:
        print(f'{args.world}: no </world> — not an SDF world?')
        return 2
    # Idempotent: strip any objects a previous run appended.
    start = world.find('<!-- fetchable objects -->')
    if start != -1:
        world = world[:start] + '  </world>\n</sdf>\n'

    placed, models, skipped = {}, [], []
    for (label, slug, rgb), room in zip(OBJECTS, sorted(rooms)):
        cx, cy = rooms[room]
        spot = find_spot(is_free, cx, cy, args.clearance)
        if spot is None:
            skipped.append((label, room))
            continue
        x, y = spot
        placed[label] = {'x': round(x, 4), 'y': round(y, 4), 'room': room}
        models.append(model_sdf(slug, x, y, rgb))
        print(f'  {label:12s} → {room:20s} at ({x:6.2f}, {y:6.2f})')

    for label, room in skipped:
        print(f'  {label:12s} → SKIPPED, no free spot near {room}')

    body = '  <!-- fetchable objects -->\n' + ''.join(models)
    world = world.replace('</world>', body + '  </world>')
    with open(args.world, 'w') as f:
        f.write(world)

    gt_path = args.world + '.objects.json'
    with open(gt_path, 'w') as f:
        json.dump(placed, f, ensure_ascii=False, indent=2)

    print(f'\nWrote {len(placed)} objects into {args.world}')
    print(f'Ground truth: {gt_path}')
    return 0 if placed else 1


if __name__ == '__main__':
    sys.exit(main())
