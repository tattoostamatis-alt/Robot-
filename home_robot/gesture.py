"""Pointing-gesture geometry — turn a skeleton into a place on the floor.

The camera gives 2D keypoints (pose_node) and an aligned depth image, so both
the shoulder and the wrist can be lifted into the camera's optical frame. The
line through them, extended forward until it meets the floor, is where the
person is pointing.

Kept free of ROS so the geometry can be unit-tested without a camera, in the
same spirit as obstacle_prediction.py and servo_filter.py.

Two things make this harder than "cast a ray":

  * **Depth on a wrist is unreliable.** A hand is small, often motion-blurred,
    and frequently has a wall metres behind it inside the same depth patch. A
    single bad wrist depth swings the floor intersection by metres, so the ray
    is only accepted when both endpoints have plausible, mutually consistent
    depth, and the result is confirmed across several frames.
  * **People gesture while talking.** An arm that merely moves is not a point.
    The arm must be reasonably straight (elbow not bent) and held away from the
    torso before the ray means anything.
"""

import math

__all__ = [
    'arm_straightness', 'is_pointing', 'floor_intersection',
    'GestureStabilizer', 'PointingCandidate',
]

# A pointing arm is nearly straight. At the elbow, shoulder->elbow and
# elbow->wrist subtend an angle near 180 degrees; a bent "talking" arm is much
# tighter. 140 degrees admits a relaxed point without admitting a folded arm.
_STRAIGHT_MIN_DEG = 140.0

# Below this the "arm" is a few noisy pixels — usually a partly occluded person.
_MIN_ARM_PIXELS = 40.0

# Depth sanity for a human limb in a room, metres.
_MIN_DEPTH = 0.3
_MAX_DEPTH = 6.0

# Shoulder and wrist belong to one arm: they cannot be far apart in depth. A
# fully extended arm pointing straight at the camera is ~0.7 m of reach, so
# anything past 1.2 m means one of the two depth samples hit the background.
_MAX_ARM_DEPTH_SPAN = 1.2


def _angle_deg(ax, ay, bx, by):
    """Angle between vectors a and b, in degrees. 0 if either is degenerate."""
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = (ax * bx + ay * by) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def arm_straightness(shoulder, elbow, wrist):
    """Elbow angle in degrees: 180 is a fully extended arm, 0 is folded shut.

    Points are 2D pixels (x, y). Pixel space is enough — a bent elbow reads as
    bent from any viewpoint that can see the arm at all.
    """
    sx, sy = shoulder
    ex, ey = elbow
    wx, wy = wrist
    # Angle at the elbow, between elbow->shoulder and elbow->wrist.
    return _angle_deg(sx - ex, sy - ey, wx - ex, wy - ey)


def is_pointing(shoulder, elbow, wrist, min_straight_deg=_STRAIGHT_MIN_DEG,
                min_pixels=_MIN_ARM_PIXELS):
    """True when this arm is extended enough to read as a deliberate point."""
    if arm_straightness(shoulder, elbow, wrist) < min_straight_deg:
        return False
    # Reject skeletons so small the keypoints are noise.
    return math.dist(shoulder, wrist) >= min_pixels


def depths_are_consistent(shoulder_depth, wrist_depth,
                          min_depth=_MIN_DEPTH, max_depth=_MAX_DEPTH,
                          max_span=_MAX_ARM_DEPTH_SPAN):
    """Both endpoints in range, and close enough together to be one arm.

    This is the guard that catches a wrist depth sample that actually landed on
    the wall behind the hand — the single most common way a pointing ray ends
    up aimed at nothing.
    """
    for d in (shoulder_depth, wrist_depth):
        if not (min_depth <= d <= max_depth):
            return False
    return abs(shoulder_depth - wrist_depth) <= max_span


def floor_intersection(shoulder_map, wrist_map, floor_z=0.0,
                       max_distance=8.0, min_distance=0.3):
    """Where the shoulder->wrist ray, extended forward, meets the floor.

    Both points are (x, y, z) in a gravity-aligned frame (``map``), so the floor
    is the plane z == floor_z. Returns (x, y) or None.

    None means the gesture does not designate a floor location: the arm is level
    or rising (pointing at a wall, a shelf, or the ceiling), the ray runs
    backwards, or the target is implausibly far. Callers must treat None as "no
    gesture", never as "the origin".
    """
    sx, sy, sz = shoulder_map
    wx, wy, wz = wrist_map

    dz = wz - sz
    # Pointing level or upward never reaches the floor in front of the person.
    # Require a real downward component, not just numerical noise.
    if dz > -1e-3:
        return None

    # Parameter along the ray where z hits the floor. Anchored at the wrist so
    # t is measured from the hand, which is what the person is aiming with.
    t = (floor_z - wz) / dz
    if t <= 0:
        return None

    x = wx + t * (wx - sx)
    y = wy + t * (wy - sy)

    reach = math.dist((wx, wy), (x, y))
    if reach < min_distance or reach > max_distance:
        return None
    return x, y


class PointingCandidate:
    """A floor point the stabilizer is currently accumulating evidence for."""

    __slots__ = ('x', 'y', 'hits')

    def __init__(self, x, y):
        self.x, self.y, self.hits = x, y, 1

    def merge(self, x, y):
        """Fold a new sighting in as a running mean, so the accepted point is
        the centre of the cluster rather than whichever frame arrived last."""
        n = self.hits + 1
        self.x += (x - self.x) / n
        self.y += (y - self.y) / n
        self.hits = n

    def as_tuple(self):
        return self.x, self.y


class GestureStabilizer:
    """Turns a stream of noisy per-frame floor points into one confirmed point.

    A gesture counts only when several consecutive frames agree to within
    ``radius`` metres. Anything else — a hand swinging past, one frame of bad
    depth — never reaches the confirmed state.

    ``confirm_hits`` frames must agree. After confirming, the same gesture is
    not re-emitted until the person drops their arm (``reset()`` via frames with
    no point) — otherwise holding a point would fire goals continuously.
    """

    def __init__(self, confirm_hits=5, radius=0.6, miss_tolerance=3):
        self.confirm_hits = confirm_hits
        self.radius = radius
        self.miss_tolerance = miss_tolerance
        self._candidate = None
        self._misses = 0
        self._latched = False

    @property
    def candidate(self):
        """The point being accumulated, or None. For live UI feedback."""
        return self._candidate

    @property
    def progress(self):
        """0.0-1.0 towards confirmation — drives the dashboard's progress ring."""
        if self._candidate is None:
            return 0.0
        return min(1.0, self._candidate.hits / float(self.confirm_hits))

    def update(self, point):
        """Feed one frame's floor point (or None). Returns (x, y) once, on the
        frame the gesture becomes confirmed; None on every other frame."""
        if point is None:
            # Tolerate brief dropouts — a keypoint flickering out for a frame or
            # two should not discard evidence already gathered.
            self._misses += 1
            if self._misses > self.miss_tolerance:
                self._candidate = None
                self._latched = False
            return None

        self._misses = 0
        x, y = point

        if self._candidate is None:
            self._candidate = PointingCandidate(x, y)
            return None

        if math.dist((x, y), self._candidate.as_tuple()) > self.radius:
            # Aimed somewhere else entirely: start over from this frame, and
            # allow a new confirmation since this is a different gesture.
            self._candidate = PointingCandidate(x, y)
            self._latched = False
            return None

        self._candidate.merge(x, y)
        if self._candidate.hits >= self.confirm_hits and not self._latched:
            self._latched = True
            return self._candidate.as_tuple()
        return None

    def reset(self):
        self._candidate = None
        self._misses = 0
        self._latched = False
