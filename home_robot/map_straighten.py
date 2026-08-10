"""Cosmetic wall-rectangularization for a saved ROS occupancy-grid map.

Shared by scripts/straighten_map_display.py (CLI) and the dashboard's map
save flow. SLAM occupancy grids are staircase-noisy AND generally rotated
(the building's real walls almost never line up with the pixel grid). A
plain cv2.approxPolyDP pass (the first version of this file) removes the
staircase but still leaves whatever odd angles the noisy contour happened to
produce — not what "Roborock/Dreame-style clean rooms" means. What those
apps actually do is find the ONE dominant wall direction for the whole
building and force every wall to be either parallel or perpendicular to it,
i.e. turn the free-space polygon into a proper rectilinear (Manhattan)
polygon. That is what rectangularize() below does; `straighten` is kept as
the public name because the dashboard and CLI already call it.

DISPLAY-ONLY by convention: pixel-level wall positions shift by several cm,
and closing small gaps can erase a real thin obstacle that happened to look
like noise — never point Nav2/AMCL at the output without a visual check.
"""
import math

import cv2
import numpy as np

DEFAULT_EPSILON = 4.0
DEFAULT_WALL_PX = 2


def _edge_angle_histogram(polygons, bins=90):
    """Length-weighted histogram of edge directions, folded into [0, 90)°
    since a building's walls come in perpendicular pairs — a wall running at
    170° is the same "grid" as one running at 80° or 260°.
    """
    hist = np.zeros(bins)
    for pts in polygons:
        n = len(pts)
        for i in range(n):
            p0, p1 = pts[i], pts[(i + 1) % n]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = math.hypot(dx, dy)
            if length < 3:      # sub-pixel noise edges would just add jitter
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 90
            hist[int(angle) % bins] += length
    return hist


def _dominant_angle(polygons) -> float:
    """The single grid angle the whole building is snapped to, in degrees."""
    hist = _edge_angle_histogram(polygons)
    if hist.sum() == 0:
        return 0.0
    # Circular smoothing so a peak split across the 0/89 wrap is not missed.
    tripled = np.concatenate([hist, hist, hist])
    kernel = np.array([1., 2., 3., 4., 3., 2., 1.])
    kernel /= kernel.sum()
    smoothed = np.convolve(tripled, kernel, mode='same')[90:180]
    return float(np.argmax(smoothed))


def _rotate(pts: np.ndarray, angle_deg: float, center: np.ndarray) -> np.ndarray:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    shifted = pts - center
    rot = np.stack([shifted[:, 0] * c - shifted[:, 1] * s,
                     shifted[:, 0] * s + shifted[:, 1] * c], axis=1)
    return rot + center


def _orthogonalize(pts: np.ndarray) -> np.ndarray:
    """Force a (pre-rotated, near-axis-aligned) polygon's edges to be exactly
    horizontal or vertical, like the rectilinear room outlines robot-vacuum
    apps draw. Walks the loop forcing each edge's free coordinate to match the
    next simplified vertex; the seam where the walk closes the loop is the one
    edge that may not land on a clean right angle, which is an acceptable
    trade for a display-only cosmetic pass.
    """
    if len(pts) < 3:
        return pts
    pts = pts.copy()

    def classify(p):
        n = len(p)
        return ['H' if abs(p[(i + 1) % n][0] - p[i][0])
                     >= abs(p[(i + 1) % n][1] - p[i][1]) else 'V'
                for i in range(n)]

    # Merge consecutive edges that share a direction — approxPolyDP sometimes
    # leaves a redundant vertex mid-wall, which would otherwise force a
    # zero-length "step" at the merge point.
    classes = classify(pts)
    changed = True
    while changed and len(pts) > 4:
        changed = False
        n = len(pts)
        for i in range(n):
            if classes[i] == classes[i - 1]:
                pts = np.delete(pts, i, axis=0)
                classes = classify(pts)
                changed = True
                break

    n = len(pts)
    out = [pts[0]]
    for i in range(n):
        p0, p1 = out[-1], pts[(i + 1) % n]
        out.append(np.array([p1[0], p0[1]]) if classes[i] == 'H'
                    else np.array([p0[0], p1[1]]))
    return np.array(out[1:])


def _bounding_rect(raw_pts: np.ndarray, theta: float, pivot: np.ndarray) -> np.ndarray:
    """Axis-aligned (in the rotated grid) bounding box of raw contour pixels,
    rotated back. Interior walls/door posts are usually just thin straight
    strips, and their approxPolyDP simplification is often only 2-3 points —
    too few for the vertex-walk in _orthogonalize to do anything sensible
    with. A rotated bounding box needs no vertex count at all and cannot
    produce the stray diagonal slivers the walk did on these.
    """
    rotated = _rotate(raw_pts, -theta, pivot)
    lo, hi = rotated.min(axis=0), rotated.max(axis=0)
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return _rotate(box, theta, pivot).astype(np.int32)


def rectangularize_polygons(gray: np.ndarray, epsilon: float = DEFAULT_EPSILON):
    """The pixel-space polygons straighten() rasterizes: (polys_free,
    polys_wall, polys_unknown), each a list of int32 Nx2 point arrays.

    Split out from straighten() so the "Τοίχοι 3D" tab can extrude the exact
    same clean rectilinear edges into wall boxes instead of re-deriving
    geometry from the rasterized image (which would re-introduce the
    staircase this whole module exists to remove — see map_walls3d.py).
    """
    free = (gray >= 250).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(free, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], [], []
    hierarchy = hierarchy[0]

    min_outer_area = 30.0
    min_hole_area = 8.0

    # Pass 1: simplify the outer (apartment) contours and classify holes as
    # real walls vs. unscanned pockets — BEFORE rotating anything, so the
    # majority-vote pixel sampling still lines up with the original image.
    # Holes keep their RAW pixel points rather than an approxPolyDP simplify:
    # interior walls/door posts are thin strips whose simplification is often
    # only 2-3 points, too few for the outer contour's vertex-walk to turn
    # into anything but a stray diagonal — see _bounding_rect below.
    outers, walls, unknowns = [], [], []
    for c, h in zip(contours, hierarchy):
        parent = h[3]
        area = cv2.contourArea(c)
        if parent == -1:
            approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2).astype(np.float64)
            if area > min_outer_area and len(approx) >= 3:
                outers.append(approx)
        elif area > min_hole_area:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
            covered = gray[mask > 0]
            raw = c.reshape(-1, 2).astype(np.float64)
            (walls if np.mean(covered == 0) >= 0.5 else unknowns).append(raw)

    if not outers:
        return [], [], []

    # Pass 2: one dominant angle for the whole building (outer contours only
    # — thin holes are too short to vote reliably), then rotate every
    # polygon onto that grid, square it up, and rotate back.
    theta = _dominant_angle(outers)
    pivot = np.mean(np.concatenate(outers), axis=0)

    def rectify_outer(poly):
        rotated = _rotate(poly, -theta, pivot)
        squared = _orthogonalize(rotated)
        return _rotate(squared, theta, pivot).astype(np.int32)

    polys_free = [rectify_outer(p) for p in outers]
    polys_wall = [_bounding_rect(p, theta, pivot) for p in walls]
    polys_unknown = [_bounding_rect(p, theta, pivot) for p in unknowns]
    return polys_free, polys_wall, polys_unknown


def straighten(gray: np.ndarray, epsilon: float = DEFAULT_EPSILON,
                wall_px: int = DEFAULT_WALL_PX) -> np.ndarray:
    """Return a rectangularized copy of a trinary map image
    (0=occupied, 254=free, 205=unknown): every wall forced parallel or
    perpendicular to the building's single dominant grid angle.

    Works on the FREE-space region's boundary rather than the (mostly
    1px-thin) wall pixels directly: a filled polygon has real area, so
    approxPolyDP simplifies it cleanly without erasing thin walls the way
    naive erosion/opening does. The outer contour is the apartment's outline;
    holes in it are interior walls/doorway posts.
    """
    canvas = np.full_like(gray, 205)  # default unknown
    polys_free, polys_wall, polys_unknown = rectangularize_polygons(gray, epsilon)
    if not polys_free:
        return canvas

    cv2.drawContours(canvas, polys_free, -1, 254, thickness=cv2.FILLED)
    cv2.drawContours(canvas, polys_unknown, -1, 205, thickness=cv2.FILLED)
    cv2.drawContours(canvas, polys_wall, -1, 0, thickness=cv2.FILLED)
    # Stroke the outer silhouette and real walls so rooms read as walled.
    cv2.drawContours(canvas, polys_free, -1, 0, thickness=wall_px)
    cv2.drawContours(canvas, polys_wall, -1, 0, thickness=wall_px)
    return canvas


def encode_png(gray: np.ndarray, upscale: int = 1) -> bytes:
    if upscale > 1:
        gray = cv2.resize(gray, (gray.shape[1] * upscale, gray.shape[0] * upscale),
                           interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode('.png', gray)
    if not ok:
        raise RuntimeError('png encode failed')
    return buf.tobytes()
