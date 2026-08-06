#!/usr/bin/env python3
"""Halve the resolution of a Nav2 occupancy map, for Gazebo world generation.

map_to_world.py extrudes every occupied cell into a wall box. At the real
0.05 m map resolution a whole apartment is over a thousand boxes, and gz then
spends all its time on collision geometry: the server pins one core, allocates
gigabytes and never steps physics, so /clock stays silent and every Nav2 node
sits waiting for a transform that will never arrive. It does not crash, which
is what makes it confusing — it looks like a Nav2 problem, not a world problem.

The recipe (docs/SESSION_2026-07-07.md, proven on kela3): downsample 2x before
generating the world, keep the origin, and treat a coarse cell as occupied if
ANY of its sub-cells was. Walls stay closed — never eroded into gaps a
simulated lidar could see through — and the box count drops to something gz
can actually simulate at real time.

    ‼️ The DOWNSAMPLED map is only ever the world's geometry. AMCL keeps
    localising against the full-resolution map, exactly as on hardware.

Usage:
    map_downsample.py maps/malou2.yaml maps/malou2_lo.yaml [--factor 2]
"""

import argparse
import os
import sys

import yaml
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_yaml')
    ap.add_argument('dst_yaml')
    ap.add_argument('--factor', type=int, default=2)
    args = ap.parse_args()

    with open(args.src_yaml) as f:
        meta = yaml.safe_load(f)

    src_dir = os.path.dirname(os.path.abspath(args.src_yaml))
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(src_dir, img_path)
    img = Image.open(img_path).convert('L')
    w, h = img.size
    px = img.load()

    occ_thresh = float(meta.get('occupied_thresh', 0.65))
    negate = int(meta.get('negate', 0))
    f = args.factor

    def occupied(c, r):
        val = px[c, r]
        p = val / 255.0 if negate else (255 - val) / 255.0
        return p > occ_thresh

    nw, nh = w // f, h // f
    out = Image.new('L', (nw, nh), 255)
    op = out.load()
    n_occ = 0
    for r in range(nh):
        for c in range(nw):
            # ANY occupied sub-cell wins. Averaging would thin a one-cell wall
            # below the threshold and open a hole the lidar shoots through,
            # which AMCL then cannot reconcile with the full-res map.
            hit = any(occupied(c * f + dc, r * f + dr)
                      for dr in range(f) for dc in range(f)
                      if c * f + dc < w and r * f + dr < h)
            if hit:
                op[c, r] = 0
                n_occ += 1

    dst_dir = os.path.dirname(os.path.abspath(args.dst_yaml))
    base = os.path.splitext(os.path.basename(args.dst_yaml))[0]
    pgm_name = base + '.pgm'
    out.save(os.path.join(dst_dir, pgm_name))

    new_meta = dict(meta)
    new_meta['image'] = pgm_name
    new_meta['resolution'] = round(float(meta['resolution']) * f, 6)
    # The origin is the bottom-left corner in metres — unchanged by resampling.
    with open(args.dst_yaml, 'w') as fh:
        yaml.safe_dump(new_meta, fh, default_flow_style=False)

    print(f'{w}x{h} @ {meta["resolution"]} m  ->  {nw}x{nh} @ '
          f'{new_meta["resolution"]} m  ({n_occ} occupied cells)')
    print(f'wrote {args.dst_yaml} and {pgm_name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
