"""Finger gestures from 21 hand landmarks.

The body skeleton (COCO-17) stops at the wrist, so no arrangement of it can
tell a thumbs-up from a fist. This module works on the 21-point hand landmark
model instead — the MediaPipe topology, run as ONNX so it needs nothing beyond
onnxruntime.

‼️ MediaPipe the PYTHON PACKAGE cannot be used here. Installing it pulls
numpy>=2 and OpenCV 5, and ROS's cv_bridge is compiled against numpy 1.26 and
OpenCV 4: cv_bridge then dies with `KeyError: 16` on every image, which takes
the whole perception stack with it. The ONNX export of the same model has none
of those dependencies.

Landmark indices (MediaPipe topology):

    0            wrist
    1  2  3  4   thumb   (CMC, MCP, IP, TIP)
    5  6  7  8   index   (MCP, PIP, DIP, TIP)
    9 10 11 12   middle
   13 14 15 16   ring
   17 18 19 20   pinky

Pure, so every gesture is testable from a list of coordinates.

Coordinates are image-space with y growing DOWNWARD, the same convention as
the body skeleton — "up" is a SMALLER y.
"""

import math

__all__ = ['HAND_GESTURES', 'HAND_GESTURE_EL', 'fingers_extended',
           'classify_hand', 'HandGestureClassifier']

HAND_GESTURES = ('fist', 'point', 'victory', 'three', 'open_palm',
                 'thumbs_up', 'thumbs_down', 'ok')

HAND_GESTURE_EL = {
    'fist':        'γροθιά',
    'point':       'ένα δάχτυλο',
    'victory':     'δύο δάχτυλα',
    'three':       'τρία δάχτυλα',
    'open_palm':   'ανοιχτή παλάμη',
    'thumbs_up':   'αντίχειρας πάνω',
    'thumbs_down': 'αντίχειρας κάτω',
    'ok':          'ΟΚ',
}

WRIST = 0
_TIPS = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
_PIPS = {'thumb': 2, 'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}
_FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def _dist(a, b):
    return math.dist(a[:2], b[:2])


def hand_scale(landmarks):
    """A size unit for the hand: wrist to middle-finger knuckle.

    None if degenerate. Everything else is measured against this so a hand at
    30 cm and the same hand at 1 m give the same answers.
    """
    if len(landmarks) < 21:
        return None
    s = _dist(landmarks[WRIST], landmarks[9])
    return s if s > 1e-6 else None


def fingers_extended(landmarks):
    """{finger: bool} — which fingers are straightened out.

    A finger is extended when its TIP is further from the wrist than its middle
    joint. This works whatever way the hand is rotated, which a "tip is above
    the joint" test does not: it would call every finger curled the moment
    somebody turns their hand sideways.
    """
    out = {}
    if len(landmarks) < 21:
        return {f: False for f in _FINGERS}
    wrist = landmarks[WRIST]
    for finger in _FINGERS:
        tip = landmarks[_TIPS[finger]]
        pip = landmarks[_PIPS[finger]]
        out[finger] = _dist(wrist, tip) > _dist(wrist, pip) * 1.10
    return out


def _thumb_index_pinch(landmarks, scale):
    """True when the thumb and index tips are touching — the OK sign."""
    return _dist(landmarks[_TIPS['thumb']], landmarks[_TIPS['index']]) < 0.35 * scale


def classify_hand(landmarks):
    """The finger gesture a hand is making, or None.

    Order matters: `ok` is tested before `three`, because an OK sign has the
    middle, ring and pinky extended and would otherwise read as three fingers.
    """
    scale = hand_scale(landmarks)
    if scale is None:
        return None

    ext = fingers_extended(landmarks)
    thumb = ext['thumb']
    others = [ext['index'], ext['middle'], ext['ring'], ext['pinky']]
    n_others = sum(others)

    # OK: thumb and index pinched, the rest of the hand open.
    if _thumb_index_pinch(landmarks, scale) and ext['middle'] and ext['ring']:
        return 'ok'

    if n_others == 0:
        if not thumb:
            return 'fist'
        # Thumb alone: which way is it pointing? y grows DOWNWARD, so a thumb
        # tip ABOVE the knuckle has the smaller y.
        tip_y = landmarks[_TIPS['thumb']][1]
        mcp_y = landmarks[1][1]
        if tip_y < mcp_y - 0.25 * scale:
            return 'thumbs_up'
        if tip_y > mcp_y + 0.25 * scale:
            return 'thumbs_down'
        return 'fist'

    if n_others == 1 and ext['index']:
        return 'point'
    if n_others == 2 and ext['index'] and ext['middle']:
        return 'victory'
    if n_others == 3 and ext['index'] and ext['middle'] and ext['ring']:
        return 'three'
    if n_others == 4:
        return 'open_palm'
    return None


class HandGestureClassifier:
    """Hold-to-confirm, mirroring GestureClassifier for the body poses.

    Fingers flicker far more than arms do — a hand rotating through a pose on
    its way somewhere else passes through two or three of these — so holding is
    even more important here than for body gestures.
    """

    def __init__(self, hold_frames=6, release_frames=4):
        self.hold_frames = hold_frames
        self.release_frames = release_frames
        self._current = None
        self._count = 0
        self._idle = 0
        self._fired = None

    def update(self, landmarks):
        gesture = classify_hand(landmarks) if landmarks else None
        if gesture is None:
            self._idle += 1
            if self._idle >= self.release_frames:
                self._current, self._count, self._fired = None, 0, None
            return None

        self._idle = 0
        if gesture != self._current:
            self._current, self._count = gesture, 1
            return None
        self._count += 1
        if self._count >= self.hold_frames and self._fired != gesture:
            self._fired = gesture
            return gesture
        return None

    @property
    def progress(self):
        if self._current is None:
            return 0.0
        return min(1.0, self._count / float(self.hold_frames))

    @property
    def holding(self):
        return self._current

    def reset(self):
        self._current, self._count, self._idle, self._fired = None, 0, 0, None
