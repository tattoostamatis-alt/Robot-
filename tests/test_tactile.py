"""Unit tests for tactile sensing (home_robot/tactile.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.tactile import (  # noqa: E402
    LOAD_KEYS, ContactDetector, describe_hardness, describe_touch,
    describe_weight, hardness, total_load, weight_estimate,
)


def fb(**kw):
    """A feedback frame; unnamed joints read zero."""
    f = {k: 0.0 for k in LOAD_KEYS}
    f.update(kw)
    return f


# ── total_load ────────────────────────────────────────────────────────────────

def test_sums_absolute_load():
    assert total_load(fb(tS=30, tE=-20)) == 50


def test_missing_keys_are_zero():
    assert total_load({'tS': 10}) == 10


def test_empty_feedback_is_zero():
    assert total_load(None) == 0.0
    assert total_load({}) == 0.0


def test_non_numeric_is_tolerated():
    assert total_load({'tS': None, 'tE': 5}) == 5


# ── contact ───────────────────────────────────────────────────────────────────

def test_first_sample_only_seeds_the_baseline():
    c = ContactDetector()
    assert c.update(fb(tS=20)) is False
    assert c.baseline == 20


def test_contact_needs_a_sustained_step():
    """A single noisy sample is not a touch."""
    c = ContactDetector(threshold=40, confirm=3)
    c.update(fb(tS=20))
    assert not c.update(fb(tS=100))
    assert not c.update(fb(tS=100))
    assert c.update(fb(tS=100))          # third in a row
    assert c.in_contact


def test_a_spike_that_does_not_persist_is_not_contact():
    c = ContactDetector(threshold=40, confirm=3)
    c.update(fb(tS=20))
    c.update(fb(tS=100))
    c.update(fb(tS=20))                  # back to normal
    c.update(fb(tS=100))
    assert not c.in_contact


def test_it_fires_once_not_every_sample():
    c = ContactDetector(threshold=40, confirm=2)
    c.update(fb(tS=20))
    fired = [c.update(fb(tS=100)) for _ in range(10)]
    assert sum(fired) == 1


def test_releasing_allows_another_contact():
    c = ContactDetector(threshold=40, confirm=2)
    c.update(fb(tS=20))
    for _ in range(3):
        c.update(fb(tS=100))
    assert c.in_contact
    for _ in range(3):
        c.update(fb(tS=20))              # released
    assert not c.in_contact
    c.update(fb(tS=100))
    assert c.update(fb(tS=100))          # fires again


def test_the_baseline_does_not_learn_during_contact():
    """‼️ Pressing on a table for a few seconds must not teach the arm that the
    table is normal — it would stop noticing it."""
    c = ContactDetector(threshold=40, confirm=2, alpha=0.5)
    c.update(fb(tS=20))
    for _ in range(20):
        c.update(fb(tS=200))
    assert c.baseline == pytest.approx(20, abs=1)
    assert c.in_contact


def test_the_baseline_follows_free_motion():
    """Holding the arm out horizontally loads the shoulder far more than
    holding it down, so the baseline has to drift with pose."""
    c = ContactDetector(threshold=100, alpha=0.5)
    c.update(fb(tS=20))
    for _ in range(20):
        c.update(fb(tS=60))
    assert c.baseline == pytest.approx(60, abs=2)
    assert not c.in_contact


def test_excess_reports_the_peak():
    c = ContactDetector(threshold=40, confirm=1)
    c.update(fb(tS=20))
    c.update(fb(tS=90))
    c.update(fb(tS=140))
    assert c.excess == pytest.approx(120, abs=1)


def test_reset():
    c = ContactDetector()
    c.update(fb(tS=20))
    c.reset()
    assert c.baseline is None and not c.in_contact


# ── hardness ──────────────────────────────────────────────────────────────────

def test_a_hard_surface_has_a_steep_slope():
    table = [(0, 0), (1, 40), (2, 82), (3, 120)]
    assert hardness(table) > 30


def test_a_soft_surface_has_a_shallow_slope():
    cushion = [(0, 0), (2, 4), (4, 9), (6, 13)]
    assert hardness(cushion) < 8


def test_labels_follow_the_slope():
    assert describe_hardness(hardness([(0, 0), (1, 40), (2, 82)])) == 'σκληρό'
    assert describe_hardness(hardness([(0, 0), (4, 9), (8, 17)])) == 'μαλακό'


def test_no_travel_is_unknown_not_infinitely_hard():
    """‼️ Dividing by zero travel would report everything as maximally hard —
    exactly the wrong way to fail."""
    assert hardness([(0, 0), (0.1, 500)]) is None


def test_too_few_samples_is_unknown():
    assert hardness([(0, 0)]) is None
    assert hardness([]) is None
    assert hardness(None) is None


def test_unknown_hardness_has_a_label():
    assert describe_hardness(None) == 'άγνωστο'


def test_slope_is_robust_to_one_noisy_sample():
    clean = [(0, 0), (1, 40), (2, 80), (3, 120)]
    noisy = [(0, 0), (1, 40), (2, 200), (3, 120)]
    assert abs(hardness(clean) - hardness(noisy)) < hardness(clean)


# ── weight ────────────────────────────────────────────────────────────────────

def test_weight_is_the_difference_at_the_lifting_joints():
    held = fb(tS=120, tE=80, tG=200)
    empty = fb(tS=40, tE=30, tG=200)
    # The gripper is excluded: it only pinches and says nothing about mass.
    assert weight_estimate(held, empty) == pytest.approx(130)


def test_an_empty_hand_weighs_nothing():
    assert weight_estimate(fb(tS=40, tE=30), fb(tS=40, tE=30)) == 0


def test_missing_frames_give_no_estimate():
    assert weight_estimate(None, fb()) is None
    assert weight_estimate(fb(), None) is None


def test_weight_labels():
    assert describe_weight(5) == 'πολύ ελαφρύ'
    assert describe_weight(40) == 'ελαφρύ'
    assert describe_weight(100) == 'μέτριο'
    assert describe_weight(400) == 'βαρύ'
    assert describe_weight(None) == 'άγνωστο'


# ── the sentence ──────────────────────────────────────────────────────────────

def test_describe_combines_both():
    text = describe_touch(slope=50, weight=200)
    assert 'σκληρό' in text and 'βαρύ' in text


def test_describe_with_only_one():
    assert 'μαλακό' in describe_touch(slope=3)
    assert 'ελαφρύ' in describe_touch(weight=40)


def test_describe_with_nothing_admits_it():
    assert describe_touch() == 'Δεν κατάλαβα τι άγγιξα.'
