#!/usr/bin/env python3
"""Draw keepout zones onto keepout_mask.pgm from keepout_zones.yaml.

Usage:
    python3 scripts/draw_keepout.py

After running, restart nav2 (or full bringup) for the new mask to take effect.
No ROS or build step needed — the config files are symlinked.
"""

import math
import os
import struct
import sys

import yaml

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PKG_DIR     = os.path.dirname(SCRIPT_DIR)
ZONES_FILE  = os.path.join(PKG_DIR, 'config', 'keepout_zones.yaml')
MASK_PGM    = os.path.join(PKG_DIR, 'config', 'keepout_mask.pgm')
MASK_YAML   = os.path.join(PKG_DIR, 'config', 'keepout_mask.yaml')
MAP_YAML    = os.path.join(PKG_DIR, 'maps', 'home.yaml')


# ── helpers ────────────────────────────────────────────────────────────────────

def read_pgm(path: str):
    """Read a binary PGM (P5) → (pixels 2D list, width, height, maxval)."""
    with open(path, 'rb') as f:
        def next_token():
            while True:
                line = f.readline()
                if not line:
                    raise EOFError
                line = line.split(b'#')[0].strip()
                for tok in line.split():
                    yield tok
        tok = next_token()
        magic  = next(tok)
        assert magic == b'P5', f'Expected P5 PGM, got {magic}'
        width  = int(next(tok))
        height = int(next(tok))
        maxval = int(next(tok))
        raw    = f.read(width * height)
    pixels = [[raw[r * width + c] for c in range(width)] for r in range(height)]
    return pixels, width, height, maxval


def write_pgm(path: str, pixels, width: int, height: int, maxval: int = 255):
    """Write a binary PGM (P5)."""
    with open(path, 'wb') as f:
        f.write(f'P5\n{width} {height}\n{maxval}\n'.encode())
        for row in pixels:
            f.write(bytes(row))


def world_to_pixel(wx: float, wy: float,
                   ox: float, oy: float, res: float,
                   height: int) -> tuple[int, int]:
    """Map-frame (x, y) → pixel (col, row).  Row 0 = bottom of map."""
    col = int(round((wx - ox) / res))
    row = height - 1 - int(round((wy - oy) / res))
    return col, row


# ── drawing ────────────────────────────────────────────────────────────────────

def draw_circle(pixels, width: int, height: int,
                cx: int, cy: int, r_px: int, value: int = 0):
    for row in range(max(0, cy - r_px), min(height, cy + r_px + 1)):
        for col in range(max(0, cx - r_px), min(width, cx + r_px + 1)):
            if (col - cx) ** 2 + (row - cy) ** 2 <= r_px ** 2:
                pixels[row][col] = value


def draw_rect(pixels, width: int, height: int,
              cx: int, cy: int, w_px: int, h_px: int, value: int = 0):
    r0 = max(0, cy - h_px // 2)
    r1 = min(height, cy + h_px // 2 + 1)
    c0 = max(0, cx - w_px // 2)
    c1 = min(width, cx + w_px // 2 + 1)
    for row in range(r0, r1):
        for col in range(c0, c1):
            pixels[row][col] = value


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    # Load map metadata
    with open(MAP_YAML) as f:
        map_info = yaml.safe_load(f)
    map_res    = float(map_info['resolution'])
    map_origin = map_info['origin']            # [x, y, yaw]
    ox, oy     = float(map_origin[0]), float(map_origin[1])

    # Load zones
    with open(ZONES_FILE) as f:
        cfg = yaml.safe_load(f)
    zones = cfg.get('zones') or {}

    if not zones:
        print('No zones defined in keepout_zones.yaml — clearing mask to all-free.')

    # Determine mask size from home.pgm
    # We read the PGM header to get width/height without PIL
    home_pgm = os.path.join(PKG_DIR, 'maps', 'home.pgm')
    _, map_w, map_h, _ = read_pgm(home_pgm)

    # Start with all-free mask (254 = passable)
    FREE    = 254
    BLOCKED = 0
    pixels = [[FREE] * map_w for _ in range(map_h)]

    drawn = []
    for name, z in zones.items():
        shape = z.get('shape', 'rect').lower()
        wx, wy = float(z['x']), float(z['y'])
        cx, cy = world_to_pixel(wx, wy, ox, oy, map_res, map_h)

        if shape == 'circle':
            r_m  = float(z['radius'])
            r_px = max(1, int(round(r_m / map_res)))
            draw_circle(pixels, map_w, map_h, cx, cy, r_px)
            drawn.append(f'  {name}: circle  centre=({wx:.2f},{wy:.2f})  r={r_m}m  →  pixel ({cx},{cy}) r={r_px}px')

        elif shape == 'rect':
            w_m, h_m = float(z['width']), float(z['height'])
            w_px = max(1, int(round(w_m / map_res)))
            h_px = max(1, int(round(h_m / map_res)))
            draw_rect(pixels, map_w, map_h, cx, cy, w_px, h_px)
            drawn.append(f'  {name}: rect    centre=({wx:.2f},{wy:.2f})  {w_m}×{h_m}m  →  pixel ({cx},{cy}) {w_px}×{h_px}px')

        else:
            print(f'  WARNING: unknown shape "{shape}" for zone {name} — skipped')

    # Write new PGM
    write_pgm(MASK_PGM, pixels, map_w, map_h)

    # Update mask YAML to match map size/origin/resolution
    mask_yaml_content = f"""\
# Auto-generated by draw_keepout.py — do not edit directly.
# Edit config/keepout_zones.yaml and re-run scripts/draw_keepout.py instead.
image: keepout_mask.pgm
mode: scale
resolution: {map_res}
origin: [{ox}, {oy}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""
    with open(MASK_YAML, 'w') as f:
        f.write(mask_yaml_content)

    print(f'keepout_mask.pgm  →  {map_w}×{map_h}px  origin=({ox},{oy})  res={map_res}m')
    if drawn:
        print(f'{len(drawn)} zone(s) drawn:')
        for d in drawn:
            print(d)
    else:
        print('Mask cleared (no active zones).')

    print()
    print('Done. Restart bringup (or just nav2) for the new mask to take effect.')
    print('  use_keepout:=true  must be passed to bringup.launch.py')


if __name__ == '__main__':
    main()
