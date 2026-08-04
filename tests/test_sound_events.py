"""Unit tests for household sound events (home_robot/sound_events.py)."""

import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.sound_events import (  # noqa: E402
    EVENTS, SoundEventGate, bearing_el, classify, describe,
)

# A stand-in for YAMNet's class map: only the names the module references, in a
# stable order. Using the real 521-entry CSV would tie the tests to a download.
CLASS_NAMES = sorted({n for _, _, names in EVENTS.values() for n in names} |
                     {'Speech', 'Conversation', 'Silence', 'Music', 'Wind'})
IDX = {n: i for i, n in enumerate(CLASS_NAMES)}


def _scores(**named):
    v = np.zeros(len(CLASS_NAMES))
    for name, score in named.items():
        v[IDX[name.replace('_', ' ')]] = score
    return v


def _score_named(mapping):
    v = np.zeros(len(CLASS_NAMES))
    for name, score in mapping.items():
        v[IDX[name]] = score
    return v


# ── classify ──────────────────────────────────────────────────────────────────

def test_silence_produces_nothing():
    events, speech = classify(np.zeros(len(CLASS_NAMES)), CLASS_NAMES)
    assert events == []
    assert speech == 0.0


def test_doorbell_is_detected():
    events, _ = classify(_score_named({'Doorbell': 0.9}), CLASS_NAMES)
    assert events and events[0][0] == 'doorbell'


def test_group_fires_on_any_member():
    """Glass/Shatter/Breaking are one household event; a sound splitting its
    confidence across near-synonyms must still fire."""
    for name in ('Glass', 'Shatter', 'Breaking', 'Smash, crash'):
        events, _ = classify(_score_named({name: 0.8}), CLASS_NAMES)
        assert 'glass' in [k for k, _ in events], name


def test_group_scores_by_its_strongest_member():
    events, _ = classify(_score_named({'Glass': 0.4, 'Shatter': 0.9}), CLASS_NAMES)
    assert dict(events)['glass'] == pytest.approx(0.9)


def test_below_threshold_is_ignored():
    events, _ = classify(_score_named({'Doorbell': 0.2}), CLASS_NAMES)
    assert events == []


def test_threshold_is_configurable():
    v = _score_named({'Doorbell': 0.2})
    assert classify(v, CLASS_NAMES, threshold=0.1)[0]


def test_results_are_sorted_by_score():
    events, _ = classify(_score_named({'Doorbell': 0.5, 'Alarm': 0.9}),
                         CLASS_NAMES)
    assert [k for k, _ in events] == ['alarm', 'doorbell']


def test_max_over_frames_not_mean():
    """Breaking glass lasts a fraction of a second. Averaging a 2s window would
    bury it under the silence around it."""
    frames = np.zeros((4, len(CLASS_NAMES)))
    frames[1, IDX['Shatter']] = 0.9        # one loud frame out of four
    events, _ = classify(frames, CLASS_NAMES)
    assert 'glass' in [k for k, _ in events]


def test_speech_is_reported_separately_not_as_an_event():
    events, speech = classify(_score_named({'Speech': 0.9}), CLASS_NAMES)
    assert events == []                     # never announced
    assert speech == pytest.approx(0.9)     # but tracked for presence


def test_quiet_speech_is_not_presence():
    _, speech = classify(_score_named({'Speech': 0.2}), CLASS_NAMES)
    assert speech == 0.0


def test_unknown_class_names_do_not_crash():
    # A different YAMNet export could rename or drop classes.
    events, _ = classify(np.zeros(3), ['Foo', 'Bar', 'Baz'])
    assert events == []


# ── bearing_el ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('angle,expected', [
    (0, 'μπροστά'), (10, 'μπροστά'), (350, 'μπροστά'),
    (90, 'αριστερά'), (270, 'δεξιά'), (180, 'πίσω'),
    (45, 'μπροστά αριστερά'), (315, 'μπροστά δεξιά'),
    (135, 'πίσω αριστερά'), (225, 'πίσω δεξιά'),
])
def test_bearing_words(angle, expected):
    assert bearing_el(angle) == expected


def test_bearing_wraps():
    assert bearing_el(360) == bearing_el(0)
    assert bearing_el(725) == bearing_el(5)


def test_no_angle_gives_no_bearing():
    """Without DoA the robot knows WHAT but not WHERE; inventing a direction is
    worse than omitting it."""
    assert bearing_el(None) is None
    assert bearing_el(float('nan')) is None


# ── describe ──────────────────────────────────────────────────────────────────

def test_describe_with_direction():
    assert describe('doorbell', 270) == 'Ακούω το κουδούνι της πόρτας, δεξιά.'


def test_describe_without_direction():
    assert describe('doorbell') == 'Ακούω το κουδούνι της πόρτας.'


def test_describe_unknown_event_does_not_crash():
    assert 'φλαμίνγκο' in describe('φλαμίνγκο')


def test_every_event_describes_cleanly():
    for key in EVENTS:
        text = describe(key, 0)
        assert text.startswith('Ακούω ') and text.endswith('.')


# ── SoundEventGate ────────────────────────────────────────────────────────────

def test_ordinary_event_needs_two_windows():
    g = SoundEventGate(min_repeats=2)
    assert g.update([('phone', 0.8)], now=1000.0) == []
    assert g.update([('phone', 0.8)], now=1001.0) == ['phone']


def test_critical_event_fires_immediately_when_confident():
    """Waiting for a second window of breaking glass is the wrong trade."""
    g = SoundEventGate(min_repeats=2)
    assert g.update([('glass', 0.9)], now=1000.0) == ['glass']


def test_alarm_and_baby_also_fire_immediately():
    for key in ('alarm', 'baby'):
        g = SoundEventGate(min_repeats=2)
        assert g.update([(key, 0.9)], now=1000.0) == [key]


def test_marginal_critical_event_still_needs_corroboration():
    """‼️ Regression from the first live run: the robot announced «Ακούω μωρό
    που κλαίει» off a single 0.53 window of ordinary room noise, because
    importance 1.0 skipped the repeat check regardless of score."""
    g = SoundEventGate(min_repeats=2, instant_min_score=0.75)
    assert g.update([('baby', 0.53)], now=1000.0) == []
    assert g.update([('baby', 0.53)], now=1001.0) == ['baby']


def test_instant_threshold_is_configurable():
    g = SoundEventGate(min_repeats=2, instant_min_score=0.5)
    assert g.update([('baby', 0.53)], now=1000.0) == ['baby']


def test_cooldown_suppresses_repeats():
    # A doorbell rings for seconds and YAMNet sees it in several windows.
    g = SoundEventGate(cooldown=20.0, min_repeats=1)
    assert g.update([('doorbell', 0.9)], now=1000.0) == ['doorbell']
    for t in (1001.0, 1005.0, 1019.0):
        assert g.update([('doorbell', 0.9)], now=t) == []
    assert g.update([('doorbell', 0.9)], now=1021.0) == ['doorbell']


def test_cooldown_is_per_event():
    """A doorbell must not suppress a smoke alarm."""
    g = SoundEventGate(cooldown=20.0, min_repeats=1)
    assert g.update([('doorbell', 0.9)], now=1000.0) == ['doorbell']
    assert g.update([('alarm', 0.9)], now=1001.0) == ['alarm']


def test_isolated_window_never_fires():
    # One-window misfires are the common failure; they must stay silent.
    g = SoundEventGate(min_repeats=2, repeat_window=3.0)
    assert g.update([('phone', 0.8)], now=1000.0) == []
    assert g.update([('phone', 0.8)], now=1100.0) == []   # too far apart
    assert g.update([('phone', 0.8)], now=1101.0) == ['phone']


def test_repeat_evidence_expires():
    g = SoundEventGate(min_repeats=3, repeat_window=2.0)
    g.update([('water', 0.8)], now=1000.0)
    g.update([('water', 0.8)], now=1000.5)
    g.update([('water', 0.8)], now=1010.0)                # old evidence gone
    assert g.update([('water', 0.8)], now=1010.5) == []


def test_more_important_event_is_announced_first():
    g = SoundEventGate(min_repeats=1)
    fired = g.update([('dog', 0.9), ('alarm', 0.9)], now=1000.0)
    assert fired[0] == 'alarm'


def test_reset_clears_state():
    g = SoundEventGate(cooldown=99.0, min_repeats=1)
    g.update([('doorbell', 0.9)], now=1000.0)
    g.reset()
    assert g.update([('doorbell', 0.9)], now=1001.0) == ['doorbell']


def test_empty_input_fires_nothing():
    assert SoundEventGate().update([], now=1000.0) == []
