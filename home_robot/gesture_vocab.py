"""A vocabulary of hand gestures, from the skeleton the camera already gives.

pose_node publishes 17 COCO keypoints per person. ‼️ The skeleton ENDS AT THE
WRIST — there are no fingers in it. Thumbs-up, a peace sign, counting on
fingers: none of these are possible with this model, and pretending otherwise
would produce a feature that looks right in code and never works in the room.
What COCO-17 does support is arm and body POSES, which have the advantage of
reading from across the room where fingers would be a handful of pixels.

Six gestures, chosen because they are visually distinct from one another at
3 metres and hard to strike by accident:

    point_floor    arm extended, angled down       "go there"
    hand_up        one wrist above the head        "stop"
    both_hands_up  both wrists above the head      "emergency"
    wave           hand up, moving side to side    "come here"
    arms_crossed   wrists across the chest         "cancel"
    t_pose         both arms straight out sideways "dock"

Pure, so the geometry is testable without a camera.

‼️ Image coordinates: y grows DOWNWARD. "Above the head" therefore means a
SMALLER y than the nose. Getting this backwards is the single easiest mistake
here, and it fails silently — the gesture simply never fires.

Everything is measured in units of SHOULDER WIDTH, so a person at 2 m and the
same person at 4 m produce the same numbers.
"""

import math

__all__ = ['GESTURES', 'GestureClassifier', 'classify_pose', 'shoulder_width']

# The gesture names this module can ever emit. The dashboard builds its editor
# from this list, so a gesture missing here simply cannot be bound.
GESTURES = ('point_floor', 'hand_up', 'both_hands_up', 'wave',
            'arms_crossed', 't_pose')

# Greek names, for the UI and for anything the robot says.
GESTURE_EL = {
    'point_floor':   'δείχνω στο πάτωμα',
    'hand_up':       'σηκωμένο χέρι',
    'both_hands_up': 'δύο χέρια ψηλά',
    'wave':          'χαιρετώ',
    'arms_crossed':  'σταυρωμένα χέρια',
    't_pose':        'χέρια ανοιχτά',
}

# A keypoint below this confidence is treated as missing rather than trusted.
_MIN_CONF = 0.5


def _pt(kps, name):
    kp = kps.get(name)
    if kp is None or kp.get('v', 0.0) < _MIN_CONF:
        return None
    return float(kp['x']), float(kp['y'])


def shoulder_width(kps):
    """Distance between the shoulders, the unit everything else is measured in.

    None when either shoulder is missing — without a scale nothing below can be
    judged, and guessing one would make every threshold distance-dependent.
    """
    left = _pt(kps, 'left_shoulder')
    right = _pt(kps, 'right_shoulder')
    if left is None or right is None:
        return None
    w = math.dist(left, right)
    return w if w > 1.0 else None


def _arm_straight(kps, side, min_deg=150.0):
    s = _pt(kps, f'{side}_shoulder')
    e = _pt(kps, f'{side}_elbow')
    w = _pt(kps, f'{side}_wrist')
    if not (s and e and w):
        return False
    ax, ay = s[0] - e[0], s[1] - e[1]
    bx, by = w[0] - e[0], w[1] - e[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na < 1e-6 or nb < 1e-6:
        return False
    cos = (ax * bx + ay * by) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) >= min_deg


def _head_y(kps):
    """Top-of-head reference. Nose if visible, else the higher eye or ear."""
    for name in ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'):
        p = _pt(kps, name)
        if p:
            return p[1]
    return None


def classify_pose(kps):
    """The gesture a single skeleton is holding, or None.

    Static poses only — `wave` needs motion and is added by GestureClassifier.
    Order matters: the two-armed poses are tested before the one-armed ones so
    that both-hands-up is never reported as hand_up.
    """
    width = shoulder_width(kps)
    if width is None:
        return None
    head = _head_y(kps)
    lw, rw = _pt(kps, 'left_wrist'), _pt(kps, 'right_wrist')
    ls, rs = _pt(kps, 'left_shoulder'), _pt(kps, 'right_shoulder')

    # ‼️ y grows downward, so "above the head" is a SMALLER y.
    left_up = bool(lw and head is not None and lw[1] < head)
    right_up = bool(rw and head is not None and rw[1] < head)

    if left_up and right_up:
        return 'both_hands_up'

    # T-pose: both wrists near shoulder height and far out to the sides.
    if lw and rw and ls and rs:
        near_shoulder_h = (abs(lw[1] - ls[1]) < 0.35 * width
                           and abs(rw[1] - rs[1]) < 0.35 * width)
        spread = abs(lw[0] - rw[0]) > 2.2 * width
        if near_shoulder_h and spread and _arm_straight(kps, 'left') \
                and _arm_straight(kps, 'right'):
            return 't_pose'

    # Arms crossed: each wrist has passed the OPPOSITE shoulder horizontally,
    # around chest height. Testing both wrists is what separates this from a
    # single arm resting across the body.
    if lw and rw and ls and rs:
        mid_x = (ls[0] + rs[0]) / 2.0
        chest_y = (ls[1] + rs[1]) / 2.0
        crossed = ((lw[0] - mid_x) * (ls[0] - mid_x) < 0
                   and (rw[0] - mid_x) * (rs[0] - mid_x) < 0)
        chest_h = (abs(lw[1] - chest_y) < 0.9 * width
                   and abs(rw[1] - chest_y) < 0.9 * width)
        if crossed and chest_h:
            return 'arms_crossed'

    if left_up or right_up:
        return 'hand_up'

    # Pointing at the floor: a straight arm angled down and away from the body.
    for side, wrist, shoulder in (('left', lw, ls), ('right', rw, rs)):
        if not (wrist and shoulder and _arm_straight(kps, side, 140.0)):
            continue
        below = wrist[1] > shoulder[1] + 0.2 * width      # hand below shoulder
        out = abs(wrist[0] - shoulder[0]) > 0.35 * width  # away from the torso
        if below and out:
            return 'point_floor'
    return None


class GestureClassifier:
    """Adds motion and hold-time to the static poses.

    Two things a single frame cannot give:

      * **wave** is hand_up plus lateral movement. Without the movement test a
        raised hand held still would read as a wave and vice versa.
      * **holding**. Every gesture must persist for `hold_frames` before it
        counts, and then it fires ONCE. Without that, a hand passing through a
        pose on its way somewhere else triggers a command.
    """

    def __init__(self, hold_frames=6, wave_frames=8, wave_min_swing=0.45,
                 release_frames=4):
        self.hold_frames = hold_frames
        self.wave_frames = wave_frames
        self.wave_min_swing = wave_min_swing
        self.release_frames = release_frames
        self._current = None
        self._count = 0
        self._idle = 0
        self._fired = None
        self._wrist_x = []          # recent wrist x, in shoulder widths

    def _track_wave(self, kps, pose):
        """Return True when the raised hand is swinging side to side."""
        width = shoulder_width(kps)
        if pose != 'hand_up' or width is None:
            self._wrist_x.clear()
            return False
        head = _head_y(kps)
        best = None
        for side in ('left', 'right'):
            w = _pt(kps, f'{side}_wrist')
            if w and head is not None and w[1] < head:
                best = w
                break
        if best is None:
            self._wrist_x.clear()
            return False
        shoulder = _pt(kps, 'left_shoulder') or _pt(kps, 'right_shoulder')
        ref_x = shoulder[0] if shoulder else best[0]
        self._wrist_x.append((best[0] - ref_x) / width)
        del self._wrist_x[:-self.wave_frames]
        if len(self._wrist_x) < self.wave_frames:
            return False
        return (max(self._wrist_x) - min(self._wrist_x)) >= self.wave_min_swing

    def update(self, kps):
        """Feed one skeleton. Returns a gesture name on the frame it is
        confirmed, otherwise None."""
        pose = classify_pose(kps) if kps else None
        if pose == 'hand_up' and self._track_wave(kps, pose):
            pose = 'wave'
        elif pose != 'hand_up':
            self._wrist_x.clear()

        if pose is None:
            self._idle += 1
            if self._idle >= self.release_frames:
                # Released: the same gesture may fire again next time.
                self._current, self._count, self._fired = None, 0, None
            return None

        self._idle = 0
        if pose != self._current:
            self._current, self._count = pose, 1
            return None
        self._count += 1
        if self._count >= self.hold_frames and self._fired != pose:
            self._fired = pose
            return pose
        return None

    @property
    def progress(self):
        """0..1 towards confirming the pose being held — for the UI ring."""
        if self._current is None:
            return 0.0
        return min(1.0, self._count / float(self.hold_frames))

    @property
    def holding(self):
        return self._current

    def reset(self):
        self._current, self._count, self._idle, self._fired = None, 0, 0, None
        self._wrist_x.clear()
