#!/usr/bin/env python3
"""Turn the RoArm-M3 URDF + STLs into one small JSON the dashboard can draw.

    scripts/build_arm_model.py            # writes config/arm_model.json

Why this exists: the dashboard page is self-contained (no CDN, no three.js, no
external fetches — see the 3D cloud tab, which draws a point cloud on a 2D
canvas for the same reason). The shipped meshes are 2.8 MB of binary STL across
eight links, which is far too much to inline and pointless besides: at the size
the arm is drawn on screen, a few hundred triangles per link is
indistinguishable from forty thousand.

So this does the heavy work ONCE, offline:

  * parses the joint chain out of roarm_m3.xacro (origins, axes, limits) —
    which is the part that must stay exact, because it decides where the links
    actually go;
  * loads each binary STL and reduces it to a triangle budget;
  * quantises coordinates to millimetres as integers, which roughly halves the
    JSON again and is well below what a 400 px viewport can show.

Re-run it if the arm is remodelled. The output is committed so the dashboard
never depends on roarm_description being present at runtime.
"""
import json
import os
import re
import struct
import sys

import numpy as np

PKG = os.path.expanduser('~/robot_ws/src/home_robot')
DESC = os.path.expanduser('~/robot_ws/src/roarm_description')
XACRO = os.path.join(DESC, 'urdf', 'roarm_m3', 'roarm_m3.xacro')
MESH_DIR = os.path.join(DESC, 'meshes', 'roarm_m3')
OUT = os.path.join(PKG, 'config', 'arm_model.json')

# Triangles kept per link. 420 is where the silhouette stops changing visibly
# at a few hundred pixels; the gripper gets more because its fingers are the
# part you actually watch when deciding whether a grasp looks right.
BUDGET = {'gripper_link': 700, 'gripper_left_link': 700}
DEFAULT_BUDGET = 420


def load_stl(path):
    """Binary STL -> (N,3,3) float array of triangle vertices, in metres."""
    with open(path, 'rb') as f:
        head = f.read(84)
        if head[:5] == b'solid' and b'facet' in head:
            raise ValueError(f'{path} is ASCII STL; only binary is handled')
        n = struct.unpack('<I', head[80:84])[0]
        data = np.frombuffer(f.read(n * 50), dtype=np.uint8)
    if len(data) < n * 50:
        raise ValueError(f'{path} truncated')
    tris = np.zeros((n, 3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            off = 12 + (i * 3 + j) * 4
            idx = np.arange(n) * 50 + off
            tris[:, i, j] = data[np.stack([idx + k for k in range(4)], 1)] \
                .copy().view(np.float32).ravel()
    return tris


def reduce_tris(tris, budget):
    """Keep the `budget` largest triangles.

    Not a real decimation — no edge collapse, no re-meshing. For a shaded
    silhouette at this size, dropping the smallest facets (which are the
    fillets and screw holes) preserves the shape people recognise, and it is
    twenty lines instead of a dependency.
    """
    if len(tris) <= budget:
        return tris
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    area = np.linalg.norm(np.cross(e1, e2), axis=1)
    keep = np.argsort(area)[-budget:]
    return tris[keep]


def parse_joints():
    src = open(XACRO).read()
    joints = []
    for m in re.finditer(r'<joint name="([^"]+)" type="(\w+)">(.*?)</joint>',
                         src, re.S):
        name, typ, body = m.groups()
        o = re.search(r'<origin xyz="([^"]+)"\s+rpy="([^"]+)"', body)
        a = re.search(r'<axis xyz="([^"]+)"', body)
        lim = re.search(r'<limit[^>]*lower="([-\d.]+)"[^>]*upper="([-\d.]+)"', body)
        parent = re.search(r'<parent link="([^"]+)"', body)
        child = re.search(r'<child link="([^"]+)"', body)
        joints.append({
            'name': name,
            'type': typ,
            'parent': parent.group(1) if parent else None,
            'child': child.group(1) if child else None,
            'xyz': [float(v) for v in o.group(1).split()] if o else [0, 0, 0],
            'rpy': [float(v) for v in o.group(2).split()] if o else [0, 0, 0],
            'axis': [float(v) for v in a.group(1).split()] if a else [0, 0, 1],
            'limit': [float(lim.group(1)), float(lim.group(2))] if lim else None,
        })
    return joints


def main():
    if not os.path.isdir(MESH_DIR):
        sys.exit(f'no meshes at {MESH_DIR} — is roarm_description checked out?')

    joints = parse_joints()
    links = {}
    total_in = total_out = 0
    for fn in sorted(os.listdir(MESH_DIR)):
        if not fn.endswith('.stl'):
            continue
        link = fn[:-4]
        tris = load_stl(os.path.join(MESH_DIR, fn))
        total_in += len(tris)
        kept = reduce_tris(tris, BUDGET.get(link, DEFAULT_BUDGET))
        total_out += len(kept)
        # ‼️ The STLs are ALREADY in millimetres — measured: base_link is 98.5
        # units across, and the real part is ~98 mm. This used to multiply by
        # 1000 "to convert metres to millimetres", which made the file
        # MICROmetres: link2 came out 246815, i.e. a 247-metre forearm. The
        # dashboard divided by 1000 and drew a 247 m arm, which orthographic
        # projection hid as an abstract grey blob and perspective turned into
        # spikes across the canvas.
        #
        # Integers, because the viewport is a few hundred pixels across a
        # ~0.4 m arm and sub-millimetre precision cannot be displayed.
        links[link] = np.round(kept).astype(int).reshape(-1).tolist()
        print(f'  {link:20s} {len(tris):6d} -> {len(kept):5d} triangles')

    model = {
        'units': 'mm',
        'joints': joints,
        # Flat [x,y,z, x,y,z, ...] per triangle, three vertices each.
        'links': links,
    }
    with open(OUT, 'w') as f:
        json.dump(model, f, separators=(',', ':'))
    size = os.path.getsize(OUT)
    print(f'\n{total_in} -> {total_out} triangles')
    print(f'wrote {OUT}  ({size / 1024:.0f} kB)')


if __name__ == '__main__':
    main()
