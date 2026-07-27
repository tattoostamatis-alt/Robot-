#!/usr/bin/env python3
"""Convert a Nav2 occupancy map (.pgm + .yaml) into a Gazebo SDF world.

Occupied cells are extruded into 1 m walls so a simulated lidar sees the same
geometry the real robot mapped — that lets AMCL localise against the very same
map file in sim as on hardware. Consecutive occupied cells in an image row are
merged into a single box to keep the wall count (and Gazebo load) sane.

Usage:
    map_to_world.py <map.yaml> <out.world> [--wall-height H] [--wall-thick T]
"""

import argparse
import sys

import yaml
from PIL import Image


WALL_HEIGHT_DEFAULT = 1.0


def load_map(yaml_path):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    import os
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
    img = Image.open(img_path).convert('L')
    return meta, img


def occupied_mask(img, occupied_thresh, negate):
    """Return a [row][col] bool grid: True where the cell is an obstacle."""
    w, h = img.size
    px = img.load()
    mask = [[False] * w for _ in range(h)]
    for row in range(h):
        for col in range(w):
            val = px[col, row]
            p = val / 255.0 if negate else (255 - val) / 255.0
            mask[row][col] = p > occupied_thresh
    return mask, w, h


def merge_runs(mask, w, h):
    """Merge horizontal runs of occupied cells per row → list of (row, c0, c1)."""
    runs = []
    for row in range(h):
        c = 0
        while c < w:
            if mask[row][c]:
                c0 = c
                while c < w and mask[row][c]:
                    c += 1
                runs.append((row, c0, c - 1))
            else:
                c += 1
    return runs


def merge_rects(runs):
    """Stack identical runs from consecutive rows into one box.

    merge_runs() alone emits a separate wall per image ROW, so a plain vertical
    wall 40 cells tall becomes 40 boxes. On the malou map that produced 1551
    links and Gazebo sat at 4 GB and 100% CPU without ever stepping physics —
    the sim never started. Merging vertically as well costs one pass and turns
    every straight wall into a single box.

    Returns (row0, row1, c0, c1) inclusive.
    """
    by_row = {}
    for row, c0, c1 in runs:
        by_row.setdefault(row, []).append((c0, c1))

    rects, consumed = [], set()
    for row, c0, c1 in runs:
        if (row, c0, c1) in consumed:
            continue
        last = row
        while (c0, c1) in by_row.get(last + 1, ()):
            last += 1
            consumed.add((last, c0, c1))
        rects.append((row, last, c0, c1))
    return rects


def build_world(meta, img, wall_height, wall_thick):
    res = float(meta['resolution'])
    ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
    occ_thresh = float(meta.get('occupied_thresh', 0.65))
    negate = int(meta.get('negate', 0))

    mask, w, h = occupied_mask(img, occ_thresh, negate)
    rects = merge_rects(merge_runs(mask, w, h))

    boxes = []
    for (r0, r1, c0, c1) in rects:
        ncols = c1 - c0 + 1
        nrows = r1 - r0 + 1
        # map_server convention: origin = bottom-left of image, y flips.
        cx = ox + (c0 + ncols / 2.0) * res
        cy = oy + (h - 1 - r1 + nrows / 2.0) * res
        boxes.append((cx, cy, ncols * res, nrows * res))

    links = []
    for i, (cx, cy, sx, sy) in enumerate(boxes):
        # sy is the merged vertical extent; wall_thick is only a floor for
        # single-row runs so a one-cell wall still has substance.
        depth = max(sy, wall_thick)
        links.append(f"""
      <link name="wall_{i}">
        <pose>{cx:.4f} {cy:.4f} {wall_height / 2:.4f} 0 0 0</pose>
        <collision name="c">
          <geometry><box><size>{sx:.4f} {depth:.4f} {wall_height:.4f}</size></box></geometry>
        </collision>
        <visual name="v">
          <geometry><box><size>{sx:.4f} {depth:.4f} {wall_height:.4f}</size></box></geometry>
          <material><ambient>0.5 0.5 0.55 1</ambient><diffuse>0.6 0.6 0.65 1</diffuse></material>
        </visual>
      </link>""")

    world = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="dimi">
    <physics name="1ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <direction>-0.3 0.2 -1</direction>
    </light>

    <model name="ground">
      <static>true</static>
      <link name="ground_link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
          <material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.35 0.35 0.35 1</diffuse></material>
        </visual>
      </link>
    </model>

    <model name="walls">
      <static>true</static>{''.join(links)}
    </model>
  </world>
</sdf>
"""
    return world, len(boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map_yaml')
    ap.add_argument('out_world')
    ap.add_argument('--wall-height', type=float, default=WALL_HEIGHT_DEFAULT)
    ap.add_argument('--wall-thick', type=float, default=None,
                    help='wall thickness in m (default: one map cell)')
    args = ap.parse_args()

    meta, img = load_map(args.map_yaml)
    thick = args.wall_thick if args.wall_thick is not None else float(meta['resolution'])
    world, n = build_world(meta, img, args.wall_height, thick)
    with open(args.out_world, 'w') as f:
        f.write(world)
    print(f'Wrote {args.out_world}: {n} wall segments from {args.map_yaml}', file=sys.stderr)


if __name__ == '__main__':
    main()
