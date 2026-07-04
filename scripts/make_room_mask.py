#!/usr/bin/env python3
"""Auto-generate maps/room_mask.png for a map via multi-seed BFS
(watershed-by-distance) from the taught locations in config/locations.yaml.

    scripts/make_room_mask.py [map_name]   # default: kela3

Re-teach the locations on the new map first (record_location.py --click),
then run this and review the *_preview.display.png overlay before renaming
the output to maps/room_mask.png.
"""
import os
import sys
from collections import deque

import numpy as np
import yaml
from PIL import Image

PKG = os.path.expanduser('~/robot_ws/src/home_robot')
MAP = sys.argv[1] if len(sys.argv) > 1 else 'kela3'

meta = yaml.safe_load(open(f'{PKG}/maps/{MAP}.yaml'))
ox, oy = meta['origin'][:2]
res = meta['resolution']
img = np.array(Image.open(f'{PKG}/maps/{MAP}.pgm'))
h, w = img.shape
free = img >= 250  # occupied ~0, unknown ~205, free 254

locs = yaml.safe_load(open(f'{PKG}/config/locations.yaml'))
colors = yaml.safe_load(open(f'{PKG}/maps/room_colors.yaml'))

names = sorted(locs)
label = np.full((h, w), -1, dtype=np.int16)   # -1 = unassigned
q = deque()
for i, name in enumerate(names):
    col = int((locs[name]['x'] - ox) / res)
    row = h - 1 - int((locs[name]['y'] - oy) / res)
    assert free[row, col], f'seed {name} not on free space!'
    label[row, col] = i
    q.append((row, col))

# Equal-speed BFS: each free pixel is claimed by whichever seed reaches it first.
while q:
    r, c = q.popleft()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and label[rr, cc] == -1:
            label[rr, cc] = label[r, c]
            q.append((rr, cc))

# RGBA mask: colored where labeled, transparent elsewhere.
mask = np.zeros((h, w, 4), dtype=np.uint8)
for i, name in enumerate(names):
    rgb = colors[name]
    sel = label == i
    mask[sel, 0], mask[sel, 1], mask[sel, 2], mask[sel, 3] = rgb[0], rgb[1], rgb[2], 255
    print(f'{name:22s} {sel.sum():5d} px = {sel.sum()*res*res:5.1f} m²  RGB={rgb}')
print(f'free unreached: {(free & (label < 0)).sum()} px')

out = f'{PKG}/maps/room_mask_{MAP}_preview'
Image.fromarray(mask).save(f'{out}.png')

# Preview: map underneath, mask at 45%, seeds as black dots, upscaled 4x.
base = np.stack([img] * 3, axis=-1).astype(np.float32)
a = (mask[..., 3:4] / 255.0) * 0.45
over = base * (1 - a) + mask[..., :3].astype(np.float32) * a
over = over.astype(np.uint8)
for name in names:
    col = int((locs[name]['x'] - ox) / res)
    row = h - 1 - int((locs[name]['y'] - oy) / res)
    over[max(0, row-2):row+3, max(0, col-2):col+3] = 0
Image.fromarray(over).resize((w*4, h*4), Image.NEAREST).save(f'{out}.display.png')
print(f'wrote {out}.png + .display.png')
