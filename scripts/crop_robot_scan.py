#!/usr/bin/env python3
"""Strip the floor/table patch out of a phone LiDAR scan of the robot itself
(Scaniverse GLB export), keeping only the robot's own dark silhouette —
used to clean up config/robot_scan.glb for the map tab's "Σάρωμα" view.

The raw scan includes whatever surface the robot was sitting on when
scanned — a wide, roughly flat, light-coloured (wood-toned) patch several
times the robot's own footprint, AND showing through the gaps in the
robot's own base ring at low height. That second part is what makes this
harder than a plain crop: radius-from-centre and colour brightness/warmth
both put genuine floor pixels at the same range of values as the robot's
own base at low height (a light-toned floor patch sits at small radius
right in the gaps of the base ring; the base ring itself has bright
specular highlights and a lighter-toned top plate elsewhere) — no single
threshold on either signal alone was clean.

What actually separated them: HEIGHT. Per-2cm-band colour composition
(session that wrote this script) showed floor-toned pixels are 28-82% of
every band below y=0.10, then abruptly under 3% in every band above it —
a sharp, reliable cutoff. So the real fix is simpler than the two-signal
approach it replaced: keep only y > ~0.095, plus a generous radius cap as
a safety net against anything far off the object's own axis (a small
disconnected debris fragment from the same scan sat well outside this).
The lost bottom ~10cm of the base becomes a flat cut, which the map tab's
loadRobotModel() already treats as the new floor-contact point (it shifts
the model so its lowest remaining point sits on the floor) — it reads as
"resting on the ground", not "floating".

Connected-component filtering (keep only the largest mesh island) was tried
and abandoned: this scan, like the house scan in ply_to_map.py's neighbour
script, is topologically fragmented into ~1000 tiny disconnected patches —
the largest was under 4% of the robot's own visible faces, nowhere near
usable as "the main body."

Usage:
    crop_robot_scan.py robot_raw.glb config/robot_scan.glb
        [--min-height 0.095] [--radius 0.30]
"""

import argparse
import sys

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_glb')
    ap.add_argument('dst_glb')
    ap.add_argument('--min-height', type=float, default=0.095,
                     help='drop everything below this local Y (m) — where the floor shows through the base')
    ap.add_argument('--radius', type=float, default=0.30,
                     help='max horizontal distance (m) from the object centre — catches off-axis debris')
    args = ap.parse_args()

    scene = trimesh.load(args.src_glb)
    mesh = trimesh.util.concatenate(scene.dump()) if isinstance(scene, trimesh.Scene) else scene

    v = mesh.vertices
    cx, cz = np.median(v[:, 0]), np.median(v[:, 2])
    r = np.hypot(v[:, 0] - cx, v[:, 2] - cz)

    face_r_max = r[mesh.faces].max(axis=1)
    face_y_min = v[mesh.faces][:, :, 1].min(axis=1)
    keep = (face_y_min > args.min_height) & (face_r_max < args.radius)

    if not keep.any():
        print('nothing left after filtering — check --min-height/--radius', file=sys.stderr)
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
