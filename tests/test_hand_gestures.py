"""Unit tests for finger gestures (home_robot/hand_gestures.py).

Hands are built pointing UP the image, so a straightened finger has a SMALLER
y than its knuckle — the same downward-y convention as the body skeleton.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.hand_gestures import (  # noqa: E402
    HAND_GESTURES, HAND_GESTURE_EL, HandGestureClassifier, classify_hand,
    fingers_extended, hand_scale,
)

# Column x for each finger, so they do not sit on top of each other.
_X = {'thumb': 60, 'index': 95, 'middle': 105, 'ring': 115, 'pinky': 125}
_BASE = {'thumb': 1, 'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17}
_TIPS = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}


def hand(extended=(), thumb_dir='side', pinch=False, flip=False):
    """A right hand with the wrist at (100, 200) and fingers pointing UP.

    `extended` names the straight fingers; the rest are curled, modelled by
    bringing the tip back to the knuckle so it sits CLOSER to the wrist than
    the middle joint.

    `flip` mirrors the whole hand about the wrist — an upside-down hand. That
    is what a real thumbs-down is, and it keeps every wrist distance identical,
    so finger extension is unaffected while the direction reverses.
    """
    lm = [[0.0, 0.0, 0.0] for _ in range(21)]
    lm[0] = [100.0, 200.0, 0.0]                 # wrist

    for finger in ('index', 'middle', 'ring', 'pinky'):
        base = _BASE[finger]
        x = _X[finger]
        mcp_y, pip_y = 150.0, 130.0
        lm[base] = [x, mcp_y, 0.0]
        lm[base + 1] = [x, pip_y, 0.0]
        if finger in extended:
            lm[base + 2] = [x, pip_y - 20, 0.0]
            lm[base + 3] = [x, pip_y - 40, 0.0]     # TIP, far from the wrist
        else:
            lm[base + 2] = [x, mcp_y + 5, 0.0]
            lm[base + 3] = [x, mcp_y + 12, 0.0]

    # The thumb sits off to the side and needs its own geometry: it has to stay
    # far from the wrist to read as extended whichever way it points.
    lm[1] = [60.0, 150.0, 0.0]                  # CMC/MCP
    lm[2] = [45.0, 140.0, 0.0]                  # IP  (the "pip" for extension)
    if 'thumb' in extended:
        if thumb_dir == 'up':
            lm[3], lm[4] = [35.0, 125.0, 0.0], [30.0, 110.0, 0.0]
        else:                                    # sideways: level with the MCP
            lm[3], lm[4] = [30.0, 150.0, 0.0], [20.0, 150.0, 0.0]
    else:
        lm[3], lm[4] = [62.0, 158.0, 0.0], [64.0, 164.0, 0.0]

    if pinch:
        lm[4] = list(lm[_TIPS['index']])         # thumb tip onto the index tip
        lm[4][0] -= 3

    if flip:
        wy = lm[0][1]
        lm = [[p[0], 2 * wy - p[1], p[2]] for p in lm]
    return lm


# ── scale and finger extension ────────────────────────────────────────────────

def test_hand_scale():
    # wrist (100,200) to middle MCP (105,150).
    assert hand_scale(hand()) == pytest.approx(50.25, abs=0.1)


def test_short_landmark_list_has_no_scale():
    assert hand_scale([[0, 0, 0]] * 5) is None


def test_extension_detects_straight_fingers():
    ext = fingers_extended(hand(extended=('index', 'middle')))
    assert ext['index'] and ext['middle']
    assert not ext['ring'] and not ext['pinky']


def test_extension_of_a_closed_hand():
    ext = fingers_extended(hand())
    assert not any(ext.values())


def test_extension_on_a_truncated_hand_is_all_false():
    assert not any(fingers_extended([[0, 0, 0]] * 3).values())


# ── the gestures ──────────────────────────────────────────────────────────────

def test_fist():
    assert classify_hand(hand()) == 'fist'


def test_point():
    assert classify_hand(hand(extended=('index',))) == 'point'


def test_victory():
    assert classify_hand(hand(extended=('index', 'middle'))) == 'victory'


def test_three():
    assert classify_hand(hand(extended=('index', 'middle', 'ring'))) == 'three'


def test_open_palm():
    got = classify_hand(hand(extended=('thumb', 'index', 'middle',
                                       'ring', 'pinky')))
    assert got == 'open_palm'


def test_open_palm_without_the_thumb_still_counts():
    got = classify_hand(hand(extended=('index', 'middle', 'ring', 'pinky')))
    assert got == 'open_palm'


def test_thumbs_up():
    assert classify_hand(hand(extended=('thumb',), thumb_dir='up')) == 'thumbs_up'


def test_thumbs_down():
    """A real thumbs-down is an inverted hand, which is what flip models."""
    got = classify_hand(hand(extended=('thumb',), thumb_dir='up', flip=True))
    assert got == 'thumbs_down'


def test_sideways_thumb_is_not_up_or_down():
    """A thumb sticking out sideways from a closed hand is not a verdict."""
    assert classify_hand(hand(extended=('thumb',), thumb_dir='side')) == 'fist'


def test_ok_sign():
    # The index IS extended in an OK sign — curved round to meet the thumb.
    got = classify_hand(hand(extended=('index', 'middle', 'ring', 'pinky'),
                             pinch=True))
    assert got == 'ok'


def test_ok_is_tested_before_open_palm():
    """‼️ An OK sign has four fingers extended and would otherwise read as an
    open palm."""
    got = classify_hand(hand(extended=('index', 'middle', 'ring', 'pinky'),
                             pinch=True))
    assert got != 'open_palm'


def test_unrecognised_combination_is_none():
    # Ring finger alone: not in the vocabulary, and must not be guessed at.
    assert classify_hand(hand(extended=('ring',))) is None


def test_degenerate_input_is_safe():
    assert classify_hand([]) is None
    assert classify_hand([[0, 0, 0]] * 21) is None


# ── vocabulary contract ───────────────────────────────────────────────────────

def test_every_gesture_has_a_greek_name():
    for g in HAND_GESTURES:
        assert HAND_GESTURE_EL.get(g), g


def test_no_stray_greek_names():
    assert set(HAND_GESTURE_EL) == set(HAND_GESTURES)


def test_classify_only_returns_known_names():
    for ext in ((), ('index',), ('index', 'middle'), ('ring',),
                ('thumb', 'index', 'middle', 'ring', 'pinky')):
        got = classify_hand(hand(extended=ext))
        assert got is None or got in HAND_GESTURES


# ── holding ───────────────────────────────────────────────────────────────────

VICTORY = hand(extended=('index', 'middle'))


def test_a_finger_gesture_must_be_held():
    """Fingers flicker more than arms; a hand passing through a pose on its way
    elsewhere must not fire a command."""
    c = HandGestureClassifier(hold_frames=5)
    for _ in range(4):
        assert c.update(VICTORY) is None
    assert c.update(VICTORY) == 'victory'


def test_it_fires_once():
    c = HandGestureClassifier(hold_frames=3)
    fired = [c.update(VICTORY) for _ in range(15)]
    assert [f for f in fired if f] == ['victory']


def test_releasing_allows_another_fire():
    c = HandGestureClassifier(hold_frames=3, release_frames=2)
    assert [f for f in (c.update(VICTORY) for _ in range(5)) if f] == ['victory']
    for _ in range(3):
        c.update(None)
    assert [f for f in (c.update(VICTORY) for _ in range(5)) if f] == ['victory']


def test_changing_gesture_restarts_the_hold():
    c = HandGestureClassifier(hold_frames=4)
    c.update(VICTORY)
    c.update(VICTORY)
    fist = hand()
    assert c.update(fist) is None
    assert c.update(fist) is None
    assert c.update(fist) is None
    assert c.update(fist) == 'fist'


def test_progress_and_holding():
    c = HandGestureClassifier(hold_frames=4)
    assert c.progress == 0.0 and c.holding is None
    c.update(VICTORY)
    assert c.progress == 0.25 and c.holding == 'victory'


def test_none_input_is_safe():
    assert HandGestureClassifier().update(None) is None


def test_reset():
    c = HandGestureClassifier(hold_frames=2)
    c.update(VICTORY)
    c.reset()
    assert c.holding is None and c.progress == 0.0
