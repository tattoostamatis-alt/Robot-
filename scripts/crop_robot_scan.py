#!/usr/bin/env python3
"""Strip the floor/table patch out of a phone LiDAR scan of the robot itself
(Scaniverse GLB export), keeping only the robot's own dark silhouette —
used to clean up config/robot_scan.glb for the map tab's "Σάρωμα" view.

The raw scan includes whatever surface the robot was sitting on when
scanned — a wide, roughly flat, light-coloured (wood-toned) patch several
times the robot's own footprint. Two independent signals separate it from
the robot, combined with AND so either one catches what the other misses:

  - radius: horizontal distance from the object's own (x,z) centre. The
    floor patch extends far past the robot's actual footprint.
  - brightness: the robot is dark (a bimodal split in per-vertex sampled
    texture colour landed the cut around brightness 105 on this scan —
    see the histogram in the session that wrote this script), the floor
    patch is light.

Neither alone was clean: radius-only left a light sliver where the floor
dips low and close to the object's own axis at the very base; a tighter
radius cut off jagged edges of the robot's own flared base instead of
floor. Brightness needs sampling the actual baseColorTexture (no per-vertex
colour in this GLB), via trimesh's TextureVisuals.to_color().

Connected-component filtering (keep only the largest mesh island) was tried
and abandoned: this scan, like the house scan in ply_to_map.py's neighbour
script, is topologically fragmented into ~1000 tiny disconnected patches —
the largest was under 4% of the robot's own visible faces, nowhere near
usable as "the main body."

Usage:
    crop_robot_scan.py robot_raw.glb config/robot_scan.glb
        [--radius 0.20] [--brightness 105]
"""

import argparse
import sys

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_glb')
    ap.add_argument('dst_glb')
    ap.add_argument('--radius', type=float, default=0.20,
                     help='max horizontal distance (m) from the object centre to keep')
    ap.add_argument('--brightness', type=float, default=105,
                     help='max mean RGB (0-255) to keep — the floor patch is lighter than this')
    args = ap.parse_args()

    scene = trimesh.load(args.src_glb)
    mesh = trimesh.util.concatenate(scene.dump()) if isinstance(scene, trimesh.Scene) else scene

    v = mesh.vertices
    cx, cz = np.median(v[:, 0]), np.median(v[:, 2])
    r = np.hypot(v[:, 0] - cx, v[:, 2] - cz)
    brightness = mesh.visual.to_color().vertex_colors[:, :3].mean(axis=1)

    face_r_max = r[mesh.faces].max(axis=1)
    face_bright = brightness[mesh.faces].mean(axis=1)
    keep = (face_r_max < args.radius) & (face_bright < args.brightness)

    if not keep.any():
        print('nothing left after filtering — check --radius/--brightness', file=sys.stderr)
        return 1

    sub = mesh.submesh([keep], append=True)
    # trimesh's glTF exporter re-encodes the texture as PNG unless the PIL
    # Image's .format says otherwise (exchange/gltf/__init__.py checks
    # `img.format == "JPEG"`) — submesh() rebuilds the material without
    # carrying that over, so a JPEG source silently ballooned into an
    # uncompressed PNG here (6.3 -> 18 MB) without this line.
    tex = sub.visual.material.baseColorTexture
    if tex is not None:
        tex.format = 'JPEG'
    sub.export(args.dst_glb)
    print(f'{keep.sum()}/{len(mesh.faces)} faces kept')
    print(f'bounds: {sub.bounds.tolist()}')
    print(f'wrote {args.dst_glb}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
