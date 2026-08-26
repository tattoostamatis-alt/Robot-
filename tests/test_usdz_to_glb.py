"""scripts/usdz_to_glb.py's alignment maths, on a synthetic room.

The part that can be silently wrong is not the USD reading (that either throws
or produces a mesh you can look at) — it is putting the scan into the map's
frame. A scan dropped in at the wrong yaw or half a metre off still renders a
convincing 3D house; what breaks is the robot marker, the trail and
click-to-navigate, all of which are drawn into the mesh in map coordinates.

Reading the .usdz needs usd-core, which the robot does not have installed; the
maths below deliberately does not, so this suite runs everywhere.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_usdz_to_glb.py -q
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'usdz_to_glb.py'
_spec = importlib.util.spec_from_file_location('usdz_to_glb', _PATH)
u = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u)

RES = 0.05


def _wall_quads(segments, y0=0.0, y1=1.0):
    """USD-frame (x, y_up, z) triangles for vertical walls along the given
    map-frame (x, y) segments — the shape a RoomPlan wall mesh has."""
    verts, faces = [], []
    for (x1, y1_), (x2, y2_) in segments:
        # map (x, y) -> USD (x, ·, -y)
        a, b = (x1, -y1_), (x2, -y2_)
        quad = [(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]),
                (a[0], y1, a[1])]
        i = len(verts)
        verts += quad
        faces += [(i, i + 1, i + 2), (i, i + 2, i + 3)]
    return {'verts': np.array(verts, float), 'faces': np.array(faces, int),
            'uv': None, 'texture': None}


def _room(w=4.0, h=3.0):
    """A closed rectangular room plus one off-centre internal wall.

    ‼️ The stub wall is the point: a bare rectangle matches itself equally well
    turned 180° (and mirrored), so the search has no way to prefer the true
    pose and the test would be asserting a coin toss. Real flats are not
    symmetric, which is why the alignment is usable at all.
    """
    c = [(0, 0), (w, 0), (w, h), (0, h)]
    walls = [(c[i], c[(i + 1) % 4]) for i in range(4)]
    walls.append(((1.0, 0.0), (1.0, 1.8)))
    return _wall_quads(walls)


def _map_from(chunk, tmp_path, origin=(0.0, 0.0), pad=0.5):
    """Render a chunk's LiDAR-height slice into a pgm+yaml, the way a saved
    map of that same room would look."""
    cells = u._slice_occupancy(chunk['verts'], chunk['faces'], RES,
                               u.LIDAR_Z - u.BAND, u.LIDAR_Z + u.BAND)
    xy = (cells + 0.5) * RES
    lo = xy.min(axis=0) - pad
    hi = xy.max(axis=0) + pad
    w = int(np.ceil((hi[0] - lo[0]) / RES))
    h = int(np.ceil((hi[1] - lo[1]) / RES))
    grid = np.full((h, w), 254, np.uint8)                  # free
    col = np.floor((xy[:, 0] - lo[0]) / RES).astype(int)
    row = h - 1 - np.floor((xy[:, 1] - lo[1]) / RES).astype(int)
    keep = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    grid[row[keep], col[keep]] = 0                          # occupied
    Image.fromarray(grid).save(tmp_path / 'm.pgm')
    (tmp_path / 'm.yaml').write_text(yaml.safe_dump({
        'image': 'm.pgm', 'resolution': RES,
        'origin': [float(lo[0]), float(lo[1]), 0.0],
        'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196}))
    return str(tmp_path / 'm.yaml')


def _apply(chunk, yaw, dx, dy):
    return {**chunk, 'verts': u.apply_transform(chunk['verts'], yaw, dx, dy)}


# ── the slice ──────────────────────────────────────────────────────────────

def test_the_slice_is_taken_at_the_lidars_height():
    """A wall that stops below the C1's beam must not end up in the map."""
    tall = _wall_quads([((0, 0), (2, 0))], y0=0.0, y1=1.0)
    low = _wall_quads([((0, 0), (2, 0))], y0=0.0, y1=0.02)
    assert len(u._slice_occupancy(tall['verts'], tall['faces'], RES,
                                  u.LIDAR_Z - u.BAND, u.LIDAR_Z + u.BAND))
    assert not len(u._slice_occupancy(low['verts'], low['faces'], RES,
                                      u.LIDAR_Z - u.BAND, u.LIDAR_Z + u.BAND))


def test_the_slice_samples_across_a_triangle_not_just_its_corners():
    """Two triangles is all a 4 m wall is; corner-only sampling would put
    three cells on the map and call the alignment hopeless."""
    wall = _wall_quads([((0, 0), (4, 0))])
    cells = u._slice_occupancy(wall['verts'], wall['faces'], RES,
                               u.LIDAR_Z - u.BAND, u.LIDAR_Z + u.BAND)
    assert len(cells) >= 4.0 / RES * 0.8


# ── the transform ──────────────────────────────────────────────────────────

def test_the_transform_only_moves_the_floor_plan_never_the_height():
    verts = np.array([[1.0, 0.4, -2.0]])
    out = u.apply_transform(verts, np.radians(90), 3.0, -1.0)
    assert out[0][1] == pytest.approx(0.4), 'height changed'
    # map (1, 2) rotated 90° -> (-2, 1), then translated
    assert out[0][0] == pytest.approx(1.0)
    assert -out[0][2] == pytest.approx(0.0)


def test_the_transform_is_a_rigid_motion():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    out = u.apply_transform(v, 0.7, 2.0, -3.0)
    d0 = np.linalg.norm(v[1:] - v[0], axis=1)
    d1 = np.linalg.norm(out[1:] - out[0], axis=1)
    assert np.allclose(d0, d1), 'distances changed: not a rigid motion'


# ── the alignment ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('yaw_deg, dx, dy', [(0, 0, 0), (90, 1.5, -2.0),
                                             (181, 2.9, 4.5), (270, -1.0, 0.5)])
def test_a_scan_is_put_back_where_the_map_says_it_is(tmp_path, yaw_deg, dx, dy):
    """Build a map from the room, move the room somewhere else, and check the
    search brings it back — this is exactly what --align-to does for a scan
    whose RoomPlan origin has nothing to do with the robot's map."""
    room = _room()
    map_yaml = _map_from(room, tmp_path)
    moved = _apply(room, np.radians(yaw_deg), dx, dy)

    yaw, ox, oy, score = u.align_to_map([moved], map_yaml)
    assert score > 0.9, f'poor fit: {score}'

    # Every wall back within a couple of cells of where the map has it. Not
    # exact: the search is over whole 5 cm cells and 1° steps, so ~one cell of
    # residual is the resolution of the method, not a bug — and one cell is far
    # tighter than the robot marker needs to read as "in the right room".
    back = u.apply_transform(moved['verts'], yaw, ox, oy)
    err = np.linalg.norm(back - room['verts'], axis=1)
    assert err.max() < 2.5 * RES, f'off by {err.max():.3f} m'


def test_a_room_that_is_not_the_map_scores_low(tmp_path):
    """The guard that stops a scan of the wrong house being baked into the
    wrong map's frame."""
    map_yaml = _map_from(_room(4.0, 3.0), tmp_path)
    other = _wall_quads([((0, 0), (0.6, 0)), ((0.6, 0), (0.6, 0.6)),
                         ((0.6, 0.6), (0, 0.6)), ((0, 0.6), (0, 0))])
    _, _, _, score = u.align_to_map([other], map_yaml)
    assert score < 0.9
