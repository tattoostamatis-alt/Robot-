"""Has someone fallen? — pure geometry over COCO-17 keypoints.

Consumes what pose_node.py already publishes on `pose_detections`, so nothing
new runs on the GPU. No ROS, no numpy-heavy work, no I/O: the decision is a
function of one skeleton plus how long it has looked that way, which is what
makes it testable without a robot or a person on the floor.

The hard part is not spotting a horizontal body — it is NOT crying wolf. A
person lying on the sofa, sitting on the floor with the kids, doing yoga, or
bending down to pick something up all produce a low and/or tilted torso. Three
independent things have to agree before this reports a fall:

  1. **The torso is horizontal.** Angle of the shoulder-midpoint → hip-midpoint
     vector against the image vertical. Standing and sitting are both upright;
     only lying down is not.
  2. **The body is low in the frame.** A person on a bed or sofa is horizontal
     too. The camera sits on a robot ~0.2 m off the floor looking slightly up,
     so a body on the FLOOR fills the lower part of the frame — furniture-height
     bodies do not.
  3. **It stays that way.** `hold_s` of continuous detection. Bending to tie a
     shoe is horizontal for a second; a fall is not something you get up from
     inside a few seconds, and if you do, no alert is the right answer.

Deliberately NOT attempted: detecting the fall *event* (the fast downward
motion). That needs reliable frame-to-frame tracking of the same person at a
stable frame rate, and the detector here runs at a few Hz with no identity
across frames. Post-hoc "someone is on the floor and has not got up" is both
easier to get right and the thing that actually matters for an empty house.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_fall_detect.py -q
"""
import math

# COCO-17 indices, the same order pose_node emits.
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16

# A keypoint below this confidence is treated as absent. Ultralytics reports
# occluded joints with low visibility rather than dropping them.
MIN_KP_CONF = 0.35

# Torso within this many degrees of horizontal counts as "lying". 45 is the
# natural midpoint between standing (0) and lying (90); 55 leans towards
# missing a borderline fall rather than reporting a seated person.
LYING_ANGLE_DEG = 55.0

# The body's centre must sit in the lowest fraction of the frame. 0.55 keeps a
# person on the floor while rejecting one on a sofa, measured against the D435
# mounted low and tilted slightly up.
FLOOR_BAND = 0.55

# A wide, short box is another lying signal, independent of the keypoints —
# useful when hips or shoulders are occluded by furniture.
WIDE_BOX_RATIO = 1.25


def _mid(kps, a, b):
    """Midpoint of two keypoints, or None if either is missing."""
    ka, kb = kps[a], kps[b]
    if ka['v'] < MIN_KP_CONF or kb['v'] < MIN_KP_CONF:
        return None
    return ((ka['x'] + kb['x']) / 2.0, (ka['y'] + kb['y']) / 2.0)


def torso_angle_deg(person):
    """Angle of the torso away from vertical, 0 = upright, 90 = flat.

    None when shoulders or hips are not visible enough to say.
    """
    kps = person.get('keypoints') or []
    if len(kps) < 17:
        return None
    sh = _mid(kps, L_SHOULDER, R_SHOULDER)
    hip = _mid(kps, L_HIP, R_HIP)
    if sh is None or hip is None:
        return None
    dx = hip[0] - sh[0]
    dy = hip[1] - sh[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    # atan2(|dx|, |dy|): purely vertical -> 0, purely horizontal -> 90.
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def box_is_wide(person):
    """Bounding box wider than tall — a lying body seen side-on."""
    w = person.get('x2', 0) - person.get('x1', 0)
    h = person.get('y2', 0) - person.get('y1', 0)
    if h <= 0:
        return False
    return (w / h) >= WIDE_BOX_RATIO


def looks_fallen(person, frame_h, *, floor_band=FLOOR_BAND,
                 lying_angle=LYING_ANGLE_DEG):
    """Does this ONE skeleton look like someone on the floor, right now?

    Time is not considered here — that is the caller's job via FallMonitor.
    """
    if not person or frame_h <= 0:
        return False

    angle = torso_angle_deg(person)
    # Fall back to the box when the torso keypoints are occluded, which is
    # common precisely when someone is down behind furniture.
    horizontal = (angle is not None and angle >= lying_angle) or \
                 (angle is None and box_is_wide(person))
    if not horizontal:
        return False

    # Low in the frame. Uses the box centre rather than a keypoint so it still
    # works when the skeleton is partial.
    cy = (person.get('y1', 0) + person.get('y2', 0)) / 2.0
    return (cy / frame_h) >= (1.0 - floor_band)


class FallMonitor:
    """Turns per-frame judgements into one alert, with hysteresis.

    Fires once when someone has looked fallen continuously for `hold_s`, and
    will not fire again until the situation clears for `clear_s` — otherwise a
    person who stays down (which is the whole point) would re-alert on every
    frame for as long as they lie there.
    """

    def __init__(self, hold_s=6.0, clear_s=10.0, clock=None):
        import time as _t
        self._clock = clock or _t.monotonic
        self.hold_s = float(hold_s)
        self.clear_s = float(clear_s)
        self._since = None        # when the current run of "fallen" started
        self._alerted = False
        self._clear_since = None

    @property
    def alerted(self):
        return self._alerted

    def update(self, fallen_now, now=None):
        """Feed one frame's verdict. True on the frame an alert should fire."""
        now = self._clock() if now is None else now

        if fallen_now:
            self._clear_since = None
            if self._since is None:
                self._since = now
            if not self._alerted and (now - self._since) >= self.hold_s:
                self._alerted = True
                return True
            return False

        # Not fallen. Require the all-clear to persist too, so one frame with
        # an occluded torso does not reset the timer and restart the countdown.
        self._since = None
        if self._alerted:
            if self._clear_since is None:
                self._clear_since = now
            elif (now - self._clear_since) >= self.clear_s:
                self._alerted = False
                self._clear_since = None
        return False
