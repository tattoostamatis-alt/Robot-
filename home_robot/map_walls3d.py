"""Turn a map's clean rectilinear walls into 3D wall footprints, in metres.

Backs the dashboard's "Τοίχοι 3D" tab. Reuses the EXACT polygons
map_straighten.rectangularize_polygons() computes rather than re-deriving
wall boxes from the rasterized pixels (the way scripts/map_to_world.py does
for its Gazebo export): this building is rotated ~25-30° off the pixel grid,
so merging horizontal pixel runs turns every wall into a staircase of dozens
of tiny boxes — exactly the noise the 2D straightening pass exists to
remove. Working from the polygons instead gives one box per real wall
segment, at whatever angle the building actually sits.
"""
import numpy as np

from home_robot.map_straighten import (DEFAULT_EPSILON, DEFAULT_WALL_PX,
                                        rectangularize_polygons)

DEFAULT_HEIGHT_M = 2.4


def _px_to_world(pt, h_px, resolution, origin_xy):
    """map_server convention: origin is the image's BOTTOM-LEFT, so pixel row
    0 (the image's top row) is the map's highest Y — see map_to_world.py.
    """
    ox, oy = origin_xy
    return (ox + pt[0] * resolution, oy + (h_px - 1 - pt[1]) * resolution)


def _edge_quad(p0, p1, half_thickness):
    d = p1 - p0
    length = float(np.hypot(d[0], d[1]))
    if length < 1e-6:
        return None
    n = np.array([-d[1], d[0]]) / length * half_thickness
    return [p0 + n, p1 + n, p1 - n, p0 - n]


def wall_footprints(gray: np.ndarray, resolution: float, origin_xy,
                     epsilon: float = DEFAULT_EPSILON,
                     wall_px: int = DEFAULT_WALL_PX,
                     height: float = DEFAULT_HEIGHT_M):
    """List of (corners, height): corners is a 4-point world-XY quad (metres),
    one per straight wall segment — every edge of the apartment's outer
    silhouette, plus every interior wall/doorpost rectangle.
    """
    h_px = gray.shape[0]
    polys_free, polys_wall, _ = rectangularize_polygons(gray, epsilon)
    half = wall_px / 2.0

    quads = []
    for poly in polys_free:
        n = len(poly)
        for i in range(n):
            p0 = np.asarray(poly[i], dtype=float)
            p1 = np.asarray(poly[(i + 1) % n], dtype=float)
            q = _edge_quad(p0, p1, half)
            if q is not None:
                quads.append(q)
    for poly in polys_wall:
        quads.append([np.asarray(p, dtype=float) for p in poly])

    return [([_px_to_world(p, h_px, resolution, origin_xy) for p in q], height)
            for q in quads]
