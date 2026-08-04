"""Unit tests for gesture bindings (home_robot/gesture_bindings.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.gesture_bindings import (  # noqa: E402
    ACTION_EL, ACTIONS, DEFAULT_BINDINGS, GestureBindings, HOLD_MOTION,
    HOLD_SAFE, hold_frames_for, is_safe,
)
from home_robot.gesture_vocab import GESTURES  # noqa: E402
from home_robot.hand_gestures import HAND_GESTURES  # noqa: E402


# ── the safety split ──────────────────────────────────────────────────────────

def test_stopping_actions_are_safe():
    for action in ('stop', 'estop', 'cancel', 'stop_follow', 'none'):
        assert is_safe(action), action


def test_moving_actions_are_not_safe():
    """‼️ A false positive on these puts a moving robot near a person."""
    for action in ('goto_pointed', 'follow', 'dock', 'patrol'):
        assert not is_safe(action), action


def test_motion_actions_need_a_longer_hold():
    assert hold_frames_for('follow') == HOLD_MOTION
    assert hold_frames_for('stop') == HOLD_SAFE
    assert HOLD_MOTION > HOLD_SAFE


# ── defaults ──────────────────────────────────────────────────────────────────

def test_every_body_gesture_has_a_default():
    for g in GESTURES:
        assert g in DEFAULT_BINDINGS, g


def test_every_finger_gesture_has_a_default():
    for g in HAND_GESTURES:
        assert g in DEFAULT_BINDINGS, g


def test_defaults_only_use_known_actions():
    for gesture, action in DEFAULT_BINDINGS.items():
        assert action in ACTIONS, f'{gesture} -> {action}'


def test_the_obvious_stop_gestures_stop():
    """A raised palm means stop to everyone; it would be perverse to bind it
    to anything else by default."""
    assert DEFAULT_BINDINGS['hand_up'] == 'stop'
    assert DEFAULT_BINDINGS['open_palm'] == 'stop'
    assert DEFAULT_BINDINGS['both_hands_up'] == 'estop'


def test_every_action_has_a_greek_label():
    for a in ACTIONS:
        assert ACTION_EL.get(a), a


def test_no_stray_labels():
    assert set(ACTION_EL) == set(ACTIONS)


# ── binding and validation ────────────────────────────────────────────────────

def test_setting_a_binding():
    b = GestureBindings()
    assert b.set('victory', 'patrol')
    assert b.action_for('victory') == 'patrol'


def test_unknown_actions_are_rejected():
    """A typo must not silently disable the gesture."""
    b = GestureBindings()
    before = b.action_for('hand_up')
    assert not b.set('hand_up', 'teleport')
    assert b.action_for('hand_up') == before


def test_none_binding_yields_no_action():
    b = GestureBindings()
    b.set('victory', 'none')
    assert b.action_for('victory') is None


def test_unbound_gesture_yields_no_action():
    assert GestureBindings().action_for('never_heard_of_it') is None


# ── the motion master switch ──────────────────────────────────────────────────

def test_motion_actions_are_suppressed_when_motion_is_off():
    b = GestureBindings(motion_enabled=False)
    b.set('victory', 'follow')
    assert b.action_for('victory') is None


def test_safe_actions_still_work_when_motion_is_off():
    """Turning motion off must never disable STOP — that would be exactly
    backwards."""
    b = GestureBindings(motion_enabled=False)
    assert b.action_for('hand_up') == 'stop'
    assert b.action_for('both_hands_up') == 'estop'


def test_motion_actions_work_when_enabled():
    b = GestureBindings(motion_enabled=True)
    b.set('victory', 'follow')
    assert b.action_for('victory') == 'follow'


# ── hold times per binding ────────────────────────────────────────────────────

def test_hold_follows_the_bound_action():
    b = GestureBindings()
    b.set('victory', 'follow')
    b.set('three', 'stop')
    assert b.hold_for('victory') == HOLD_MOTION
    assert b.hold_for('three') == HOLD_SAFE


# ── persistence ───────────────────────────────────────────────────────────────

def test_round_trip():
    b = GestureBindings(motion_enabled=False)
    b.set('victory', 'dock')
    restored = GestureBindings.from_dict(b.to_dict())
    assert restored.motion_enabled is False
    assert restored.as_dict()['victory'] == 'dock'


def test_from_dict_tolerates_empty():
    b = GestureBindings.from_dict({})
    assert b.action_for('hand_up') == 'stop'      # defaults intact


def test_from_dict_ignores_bad_actions():
    b = GestureBindings.from_dict({'bindings': {'hand_up': 'nonsense'}})
    assert b.action_for('hand_up') == 'stop'


def test_partial_mapping_keeps_other_defaults():
    b = GestureBindings(mapping={'victory': 'patrol'})
    assert b.action_for('victory') == 'patrol'
    assert b.action_for('hand_up') == 'stop'
