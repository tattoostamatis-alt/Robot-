"""Shape-only room segmentation: distance-transform watershed over the free
space of an occupancy grid. Used by scripts/auto_rooms.py (offline, one map
at a time) and by web_dashboard_node.py's click-to-place-a-room tool (the
same algorithm run on demand, so a click lands on the same room boundary the
CLI would have drawn) — one implementation, not two that can drift apart.

See scripts/auto_rooms.py's module docstring for the algorithm writeup.
"""
import numpy as np
from scipy import ndimage

# 0.90 / 2.0 m^2 were chosen by running this over maps/malou2.pgm (the flat
# it actually drives) and looking at the preview — see auto_rooms.py.
DOOR_M = 0.90
MIN_AREA_M2 = 2.0

# Reused for room 1, 2, 3... Distinct hues rather than a gradient, because
# these are labels and not a scale. Kept clear of the greys the map is drawn
# in so a room is never confused with a wall.
PALETTE = [
    (204, 68, 255), (255, 204, 0), (68, 136, 255), (68, 204, 102),
    (255, 68, 68), (0, 214, 214), (255, 136, 0), (170, 102, 255),
    (102, 204, 170), (255, 102, 170), (140, 180, 60), (90, 150, 255),
]


def segment(img, res, door_m=DOOR_M, min_area_m2=MIN_AREA_M2, close_px=2):
    """Label the rooms in a grayscale occupancy-grid image (map_server's PGM
    convention: >=250 free, dark occupied). Returns (labels, count); label 0
    means 'not a room' (wall, doorway sliver, or below min_area_m2).
    """
    free = img >= 250

    # Speckle inside a room would punch holes in the distance transform and
    # split one room into several. A binary close is enough; the walls are
    # thick relative to a couple of pixels.
    if close_px:
        free = ndimage.binary_closing(free, np.ones((close_px, close_px), bool))

    # Distance (in metres) from each free cell to the nearest wall.
    dist = ndimage.distance_transform_edt(free) * res

    # A doorway is at most `door_m` wide, so its centre line sits at most
    # door_m/2 from a wall. Anything deeper than that is room interior.
    cores = dist > (door_m / 2.0)
    lab, n = ndimage.label(cores)
    if n == 0:
        return np.zeros_like(img, np.int32), 0

    # Grow the seeds back over all free space: every free cell takes the label
    # of the nearest seed. This is the watershed, done with one EDT rather than
    # an explicit flood, and it puts the boundary in the doorway — the far side
    # of a door is nearer the room beyond it than the room behind.
    _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    grown = lab[iy, ix]
    grown[~free] = 0

    # Only NOW drop the small ones. Filtering the seeds instead measured the
    # cores — much smaller than the rooms they become.
    cell_area = res * res
    out = np.zeros_like(grown)
    next_id = 1
    for i in range(1, n + 1):
        m = grown == i
        if m.sum() * cell_area >= min_area_m2:
            out[m] = next_id
            next_id += 1
    return out, next_id - 1


def colourise(labels, count):
    """RGBA image: room colour where labelled, fully transparent elsewhere."""
    h, w = labels.shape
    out = np.zeros((h, w, 4), np.uint8)
    for i in range(1, count + 1):
        r, g, b = PALETTE[(i - 1) % len(PALETTE)]
        m = labels == i
        out[m] = (r, g, b, 255)
    return out
