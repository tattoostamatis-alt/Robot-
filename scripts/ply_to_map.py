#!/usr/bin/env python3
"""Convert a gravity-aligned point cloud (e.g. iPhone LiDAR scan exported from
Scaniverse/3D Scanner App) into a Nav2/AMCL-style occupancy grid (pgm+yaml),
loadable with switch_map_mode.sh localize <name>.

There is no supported path to import an external point cloud into RTAB-Map's
own SLAM graph — its map is built only from live sensor+odometry+loop-closure
data (see memory project_robot_rtabmap_3d). This script instead builds a plain
2D occupancy grid the same way map_server/slam_toolbox output looks, going
straight around RTAB-Map into the AMCL/Nav2 map format.

Method: slice two horizontal bands out of the cloud —
  - "floor band" near z=0: wherever the phone actually saw floor, that cell
    is traversable ("free"). Cells the phone never saw stay unknown (gray),
    same as any unexplored SLAM map.
  - "lidar band" centered on the robot's real C1 LiDAR height (z=0.16,
    ~/robot_ws/src/home_robot/description/robot_sim.urdf.xacro lidar_z) —
    wherever the phone saw a surface at roughly the height the robot's own
    LiDAR would hit it, that cell is "occupied". Occupied always wins over
    free where they disagree (never open a gap in a wall the way
    map_downsample.py's ANY-hit rule protects against for the opposite bug).
  - A small morphological closing (3x3) on the occupied mask patches
    single-cell scan holes in walls without closing real doorways, which at
    0.05 m/cell are tens of cells wide.

‼️ The output's origin/yaw is whatever arbitrary frame Scaniverse assigned
the scan — it is NOT aligned to the robot's existing map (e.g. malou3). Do
not expect global coordinates to match; treat this as a fresh, separate map
that needs its own initial pose (2D Pose Estimate in RViz, or AprilTag reloc)
after switching to it. Distances have not been checked against a tape
measure — verify a couple of known distances before trusting it for
autonomous nav, same caution as the original scan-to-map plan.

Usage:
    ply_to_map.py 1.ply maps/iphone_house.yaml [--resolution 0.05]
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import yaml
from PIL import Image
from scipy.ndimage import binary_closing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ply_path')
    ap.add_argument('dst_yaml')
    ap.add_argument('--resolution', type=float, default=0.05)
    ap.add_argument('--lidar-height', type=float, default=0.16,
                     help='C1 LiDAR height above floor, meters (robot_sim.urdf.xacro lidar_z)')
    ap.add_argument('--band', type=float, default=0.12,
                     help='half-thickness of the occupied-detection slice around --lidar-height')
    ap.add_argument('--floor-lo', type=float, default=-0.05)
    ap.add_argument('--floor-hi', type=float, default=0.08)
    ap.add_argument('--padding', type=float, default=0.3, help='meters of margin around the scanned floor extent')
    ap.add_argument('--min-occ-points', type=int, default=2, help='points/cell to call a cell occupied (noise filter)')
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply_path)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print(f'no points read from {args.ply_path}', file=sys.stderr)
        return 1

    floor_mask = (pts[:, 2] >= args.floor_lo) & (pts[:, 2] <= args.floor_hi)
    occ_mask = (pts[:, 2] >= args.lidar_height - args.band) & (pts[:, 2] <= args.lidar_height + args.band)

    floor_xy = pts[floor_mask][:, :2]
    if len(floor_xy) == 0:
        print('no points in the floor band — check --floor-lo/--floor-hi against the cloud\'s Z range', file=sys.stderr)
        return 1

    origin_x = floor_xy[:, 0].min() - args.padding
    origin_y = floor_xy[:, 1].min() - args.padding
    max_x = floor_xy[:, 0].max() + args.padding
    max_y = floor_xy[:, 1].max() + args.padding

    res = args.resolution
    w = int(np.ceil((max_x - origin_x) / res))
    h = int(np.ceil((max_y - origin_y) / res))

    def to_grid(xy):
        col = np.floor((xy[:, 0] - origin_x) / res).astype(int)
        row = np.floor((xy[:, 1] - origin_y) / res).astype(int)  # row 0 = origin_y (ymin)
        valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        return row[valid], col[valid]

    # grid[row, col], row 0 = ymin — flipped to image order (row 0 = ymax) at the end.
    free_count = np.zeros((h, w), dtype=np.int32)
    r, c = to_grid(floor_xy)
    np.add.at(free_count, (r, c), 1)

    occ_count = np.zeros((h, w), dtype=np.int32)
    occ_xy = pts[occ_mask][:, :2]
    r, c = to_grid(occ_xy)
    np.add.at(occ_count, (r, c), 1)

    occupied = occ_count >= args.min_occ_points
    occupied = binary_closing(occupied, structure=np.ones((3, 3), dtype=bool))
    free = (free_count >= 1) & ~occupied

    grid = np.full((h, w), 205, dtype=np.uint8)  # unknown
    grid[free] = 254
    grid[occupied] = 0

    image = grid[::-1, :]  # pgm row 0 = ymax, matches ROS map_server convention

    dst_dir = os.path.dirname(os.path.abspath(args.dst_yaml)) or '.'
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.dst_yaml))[0]
    pgm_name = base + '.pgm'
    Image.fromarray(image, mode='L').save(os.path.join(dst_dir, pgm_name))

    meta = {
        'image': pgm_name,
        'mode': 'trinary',
        'resolution': res,
        'origin': [round(float(origin_x), 3), round(float(origin_y), 3), 0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }
    with open(args.dst_yaml, 'w') as fh:
        yaml.safe_dump(meta, fh, default_flow_style=False)

    n_occ = int(occupied.sum())
    n_free = int(free.sum())
    n_unknown = h * w - n_occ - n_free
    print(f'{w}x{h} cells @ {res} m  ({w*res:.1f} x {h*res:.1f} m)')
    print(f'occupied={n_occ} free={n_free} unknown={n_unknown}')
    print(f'origin={meta["origin"]}')
    print(f'wrote {args.dst_yaml} and {pgm_name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
