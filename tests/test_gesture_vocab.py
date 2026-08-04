"""Unit tests for the gesture vocabulary (home_robot/gesture_vocab.py).

‼️ Image coordinates throughout: y grows DOWNWARD. A raised hand has a SMALLER
y than the head. Every skeleton below is built with that in mind.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.gesture_vocab import (  # noqa: E402
    GESTURE_EL, GESTURES, GestureClassifier, classify_pose, shoulder_width,
)


def kp(x, y, v=1.0):
    return {'x': x, 'y': y, 'v': v}


def skeleton(**over):
    """A person facing the camera, shoulders 100px apart, arms hanging down."""
    s = {
        'nose':           kp(300, 100),
        'left_eye':       kp(290, 95),  'right_eye':  kp(310, 95),
        'left_shoulder':  kp(250, 160), 'right_shoulder': kp(350, 160),
        'left_elbow':     kp(245, 240), 'right_elbow':    kp(355, 240),
        'left_wrist':     kp(240, 320), 'right_wrist':    kp(360, 320),
        'left_hip':       kp(265, 330), 'right_hip':      kp(335, 330),
    }
    s.update(over)
    return s


# ── scale ─────────────────────────────────────────────────────────────────────

def test_shoulder_width_is_the_unit():
    assert shoulder_width(skeleton()) == 100.0


def test_missing_shoulder_has_no_scale():
    """Without a scale every threshold would depend on how far away the person
    stands, so the honest answer is None."""
    assert shoulder_width(skeleton(left_shoulder=kp(250, 160, v=0.0))) is None


def test_no_scale_means_no_gesture():
    assert classify_pose(skeleton(left_shoulder=kp(0, 0, v=0.0))) is None


def test_empty_skeleton_is_safe():
    assert classify_pose({}) is None


# ── static poses ──────────────────────────────────────────────────────────────

def test_arms_down_is_no_gesture():
    """The resting pose must never be a command."""
    assert classify_pose(skeleton()) is None


def test_hand_up():
    s = skeleton(left_wrist=kp(240, 40), left_elbow=kp(245, 100))
    assert classify_pose(s) == 'hand_up'


def test_hand_up_needs_to_be_above_the_head():
    """‼️ y grows downward. A wrist at shoulder height is NOT raised."""
    s = skeleton(left_wrist=kp(240, 150), left_elbow=kp(245, 155))
    assert classify_pose(s) != 'hand_up'


def test_both_hands_up():
    s = skeleton(left_wrist=kp(240, 40), left_elbow=kp(245, 100),
                 right_wrist=kp(360, 40), right_elbow=kp(355, 100))
    assert classify_pose(s) == 'both_hands_up'


def test_both_hands_up_beats_hand_up():
    """Order matters: two hands must never be reported as one."""
    s = skeleton(left_wrist=kp(240, 40), left_elbow=kp(245, 100),
                 right_wrist=kp(360, 40), right_elbow=kp(355, 100))
    assert classify_pose(s) == 'both_hands_up'


def test_t_pose():
    s = skeleton(left_wrist=kp(100, 160), left_elbow=kp(175, 160),
                 right_wrist=kp(500, 160), right_elbow=kp(425, 160))
    assert classify_pose(s) == 't_pose'


def test_t_pose_needs_real_spread():
    # Elbows out but wrists close to the body: not a T.
    s = skeleton(left_wrist=kp(225, 160), left_elbow=kp(238, 160),
                 right_wrist=kp(375, 160), right_elbow=kp(362, 160))
    assert classify_pose(s) != 't_pose'


def test_arms_crossed():
    # Each wrist has passed the midline to the opposite side, at chest height.
    s = skeleton(left_wrist=kp(340, 230), left_elbow=kp(260, 250),
                 right_wrist=kp(260, 230), right_elbow=kp(340, 250))
    assert classify_pose(s) == 'arms_crossed'


def test_one_arm_across_the_body_is_not_crossed():
    """Testing both wrists is what separates this from a resting arm."""
    s = skeleton(left_wrist=kp(340, 230), left_elbow=kp(260, 250))
    assert classify_pose(s) != 'arms_crossed'


def test_point_at_the_floor():
    s = skeleton(left_wrist=kp(120, 300), left_elbow=kp(185, 230))
    assert classify_pose(s) == 'point_floor'


def test_point_needs_the_arm_away_from_the_body():
    # Straight but hanging: that is standing, not pointing.
    s = skeleton(left_wrist=kp(248, 330), left_elbow=kp(249, 245))
    assert classify_pose(s) != 'point_floor'


def test_low_confidence_keypoints_are_ignored():
    s = skeleton(left_wrist=kp(240, 40, v=0.1), left_elbow=kp(245, 100, v=0.1))
    assert classify_pose(s) != 'hand_up'


# ── vocabulary contract ───────────────────────────────────────────────────────

def test_every_gesture_has_a_greek_name():
    for g in GESTURES:
        assert GESTURE_EL.get(g), g


def test_no_stray_names_in_the_greek_table():
    assert set(GESTURE_EL) == set(GESTURES)


def test_classify_only_ever_returns_known_names():
    poses = [
        skeleton(),
        skeleton(left_wrist=kp(240, 40), left_elbow=kp(245, 100)),
        skeleton(left_wrist=kp(120, 300), left_elbow=kp(185, 230)),
    ]
    for s in poses:
        got = classify_pose(s)
        assert got is None or got in GESTURES


# ── holding and firing ────────────────────────────────────────────────────────

HAND_UP = skeleton(left_wrist=kp(240, 40), left_elbow=kp(245, 100))


def test_a_gesture_must_be_held():
    """A hand passing through a pose on its way elsewhere must not fire."""
    c = GestureClassifier(hold_frames=6)
    for _ in range(5):
        assert c.update(HAND_UP) is None
    assert c.update(HAND_UP) == 'hand_up'


def test_it_fires_once_not_every_frame():
    c = GestureClassifier(hold_frames=3)
    fired = [c.update(HAND_UP) for _ in range(20)]
    assert [f for f in fired if f] == ['hand_up']


def test_releasing_allows_it_to_fire_again():
    c = GestureClassifier(hold_frames=3, release_frames=2)
    assert [f for f in (c.update(HAND_UP) for _ in range(5)) if f] == ['hand_up']
    for _ in range(3):
        c.update(skeleton())               # arms down = released
    assert [f for f in (c.update(HAND_UP) for _ in range(5)) if f] == ['hand_up']


def test_switching_gesture_restarts_the_count():
    c = GestureClassifier(hold_frames=4)
    c.update(HAND_UP)
    c.update(HAND_UP)
    both = skeleton(left_wrist=kp(240, 40), left_elbow=kp(245, 100),
                    right_wrist=kp(360, 40), right_elbow=kp(355, 100))
    assert c.update(both) is None          # count restarted
    assert c.update(both) is None
    assert c.update(both) is None
    assert c.update(both) == 'both_hands_up'


def test_progress_reports_the_hold():
    c = GestureClassifier(hold_frames=4)
    assert c.progress == 0.0
    c.update(HAND_UP)
    assert c.progress == 0.25
    c.update(HAND_UP)
    assert c.progress == 0.5


def test_holding_reports_the_current_pose():
    c = GestureClassifier(hold_frames=4)
    c.update(HAND_UP)
    assert c.holding == 'hand_up'


def test_none_input_is_safe():
    c = GestureClassifier()
    assert c.update(None) is None


def test_reset_clears_state():
    c = GestureClassifier(hold_frames=2)
    c.update(HAND_UP)
    c.reset()
    assert c.holding is None
    assert c.progress == 0.0


# ── wave ──────────────────────────────────────────────────────────────────────

def _waving(offset):
    return skeleton(left_wrist=kp(240 + offset, 40), left_elbow=kp(245, 100))


def test_a_still_raised_hand_is_not_a_wave():
    c = GestureClassifier(hold_frames=3, wave_frames=6)
    fired = [c.update(HAND_UP) for _ in range(20)]
    assert 'wave' not in [f for f in fired if f]
    assert 'hand_up' in [f for f in fired if f]


def test_a_swinging_hand_is_a_wave():
    c = GestureClassifier(hold_frames=3, wave_frames=6, wave_min_swing=0.4)
    fired = []
    for i in range(24):
        fired.append(c.update(_waving(-40 if i % 2 else 40)))
    assert 'wave' in [f for f in fired if f]


def test_wave_survives_a_single_dropped_frame():
    """One bad frame must not cancel a gesture — release needs release_frames."""
    c = GestureClassifier(hold_frames=3, wave_frames=6, release_frames=4)
    for i in range(6):
        c.update(_waving(-40 if i % 2 else 40))
    c.update(skeleton())
    assert c.holding is not None


def test_wave_clears_once_the_hand_is_really_down():
    c = GestureClassifier(hold_frames=3, wave_frames=6, release_frames=3)
    for i in range(6):
        c.update(_waving(-40 if i % 2 else 40))
    for _ in range(4):
        c.update(skeleton())
    assert c.holding is None
