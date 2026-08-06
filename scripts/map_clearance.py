#!/usr/bin/env python3
"""How wide are the passages between rooms, on a given map?

For each pair of named locations, finds the largest disc radius that can still
travel from one to the other. That number IS the tightest passage on the route
(half its width), and it is the number that decides whether Nav2 can plan:
below robot_radius the planner has nothing to work with, and only a little
above it the inflation collars from the two jambs meet in the middle.

Method: distance transform of the free space, then a binary search on radius
with a flood fill over cells whose clearance is at least that radius.

    map_clearance.py maps/malou2.yaml
    map_clearance.py maps/malou2_lo.yaml --compare maps/malou2.yaml

‼️ The --compare form is the one that matters for Gazebo. sim worlds are built
from a 2x-downsampled map (see map_downsample.py) where a coarse cell counts as
occupied if ANY of its four sub-cells was — that keeps walls sealed but also
FATTENS them by up to half a coarse cell on each side. A 0.60 m doorway can
come out ~0.40 m in the world the simulated lidar sees, and the robot then
wedges in a door that is perfectly passable in the real apartment. Before
blaming Nav2 for a doorway stall in sim, check that the doorway still exists.
"""

import argparse
import math
import os
import sys
from collections import deque

import yaml
from PIL import Image


def load(map_yaml):
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    p = meta['image']
    if not os.path.isabs(p):
        p = os.path.join(os.path.dirname(os.path.abspath(map_yaml)), p)
    return meta, Image.open(p).convert('L')


def free_grid(meta, img):
    """[row][col] True where the cell is known-free."""
    occ_t = float(meta.get('occupied_thresh', 0.65))
    free_t = float(meta.get('free_thresh', 0.196))
    neg = int(meta.get('negate', 0))
    w, h = img.size
    px = img.load()
    g = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            v = px[c, r]
            p = v / 255.0 if neg else (255 - v) / 255.0
            g[r][c] = p <= free_t          # unknown counts as blocked
    return g, w, h


def clearance_map(free, w, h, res):
    """Metres from each free cell to the nearest non-free cell (BFS-based).

    Multi-source BFS on the 8-connected grid gives a Chamfer-ish distance; good
    enough here, and exact enough that the numbers match the ones already
    recorded for these maps.
    """
    INF = float('inf')
    dist = [[INF] * w for _ in range(h)]
    dq = deque()
    for r in range(h):
        for c in range(w):
            if not free[r][c]:
                dist[r][c] = 0.0
                dq.append((r, c))
    # 8-neighbour steps with their true lengths.
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    while dq:
        r, c = dq.popleft()
        for dr, dc, cost in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                nd = dist[r][c] + cost
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    dq.append((nr, nc))
    return [[d * res for d in row] for row in dist]


def to_cell(meta, h, x, y):
    res = float(meta['resolution'])
    ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
    return int(h - 1 - (y - oy) / res), int((x - ox) / res)


def connected(clear, w, h, start, goal, radius):
    """Can a disc of `radius` get from start to goal?"""
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return False
    if clear[sr][sc] < radius or clear[gr][gc] < radius:
        return False
    seen = [[False] * w for _ in range(h)]
    dq = deque([(sr, sc)])
    seen[sr][sc] = True
    while dq:
        r, c = dq.popleft()
        if (r, c) == (gr, gc):
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and not seen[nr][nc]
                    and clear[nr][nc] >= radius):
                seen[nr][nc] = True
                dq.append((nr, nc))
    return False


def widest_disc(clear, w, h, start, goal, hi=0.6):
    """Binary search the largest disc radius that still connects the two."""
    lo = 0.0
    if not connected(clear, w, h, start, goal, 0.01):
        return 0.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if connected(clear, w, h, start, goal, mid):
            lo = mid
        else:
            hi = mid
    return lo


def analyse(map_yaml, rooms, robot_radius):
    meta, img = load(map_yaml)
    res = float(meta['resolution'])
    free, w, h = free_grid(meta, img)
    clear = clearance_map(free, w, h, res)
    cells = {n: to_cell(meta, h, x, y) for n, (x, y) in rooms.items()}
    out = {}
    base = 'saloni' if 'saloni' in cells else sorted(cells)[0]
    for name, cell in sorted(cells.items()):
        if name == base:
            continue
        r = widest_disc(clear, w, h, cells[base], cell)
        out[name] = r
    return out, base, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map_yaml')
    ap.add_argument('--compare', help='second map to print alongside')
    ap.add_argument('--robot-radius', type=float, default=0.175)
    args = ap.parse_args()

    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(src, 'config', 'locations.yaml')) as f:
        locs = yaml.safe_load(f) or {}
    rooms = {k: (float(v['x']), float(v['y'])) for k, v in locs.items()
             if isinstance(v, dict) and 'x' in v and not k.startswith('dock')}

    a, base, res_a = analyse(args.map_yaml, rooms, args.robot_radius)
    b = None
    if args.compare:
        b, _, res_b = analyse(args.compare, rooms, args.robot_radius)

    rr = args.robot_radius
    print(f'Largest disc radius that reaches each room from {base}.')
    print(f'Robot radius is {rr} m — anything at or below that is impassable.\n')
    head = f'  {"room":22s} {os.path.basename(args.map_yaml):>16s}'
    if b:
        head += f' {os.path.basename(args.compare):>16s}   verdict'
    print(head)
    for name in sorted(a):
        line = f'  {name:22s} {a[name]:>13.3f} m'
        if b:
            line += f' {b[name]:>13.3f} m'
            passable_a = a[name] > rr
            passable_b = b[name] > rr
            if passable_a and not passable_b:
                line += '   ← blocked in the FIRST map only'
            elif passable_b and not passable_a:
                line += '   ← blocked in the FIRST map only'
            elif not passable_a and not passable_b:
                line += '   ← blocked in BOTH'
            else:
                line += '   ok in both'
        else:
            line += '   ' + ('ok' if a[name] > rr else '← IMPASSABLE')
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
