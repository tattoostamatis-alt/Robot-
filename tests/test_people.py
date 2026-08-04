"""Unit tests for person profiles and identity fusion (home_robot/people.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.people import (  # noqa: E402
    HEAD_TO_CROWN, Person, PersonRegistry, fuse,
)


def settled(name, height, n=20, jitter=0.005):
    """A person with a settled height, measured n times with slight jitter."""
    p = Person(name)
    for i in range(n):
        p.observe_height(height + (jitter if i % 2 else -jitter))
    return p


# ── height statistics ─────────────────────────────────────────────────────────

def test_height_needs_several_samples():
    p = Person('max')
    for _ in range(3):
        p.observe_height(1.30)
    assert p.height is None            # not enough yet
    for _ in range(10):
        p.observe_height(1.30)
    assert p.height == pytest.approx(1.30, abs=0.01)


def test_implausible_heights_are_rejected():
    """The two failures that actually happen: depth landing on the wall behind
    the person, and a head keypoint that jumped to the floor."""
    p = Person('max')
    assert not p.observe_height(0.2)
    assert not p.observe_height(4.5)
    assert p.samples == 0


def test_a_toddler_and_a_tall_adult_both_fit():
    assert Person('a').observe_height(0.75)
    assert Person('b').observe_height(2.05)


def test_standard_deviation_tracks_spread():
    tight = settled('a', 1.75, jitter=0.005)
    loose = settled('b', 1.75, jitter=0.06)
    assert loose.height_sd > tight.height_sd


def test_crown_offset_is_a_sane_constant():
    assert 0.05 < HEAD_TO_CROWN < 0.2


# ── the veto ──────────────────────────────────────────────────────────────────

def test_height_vetoes_an_impossible_match():
    """Very good at saying 'whoever that is, it is not the eight-year-old'."""
    max_ = settled('max', 1.28)
    assert max_.height_conflicts(1.82)


def test_height_does_not_veto_a_plausible_match():
    dad = settled('dad', 1.80)
    assert not dad.height_conflicts(1.83)


def test_a_slouch_cannot_veto():
    """‼️ Posture, shoes and a slouch move a real person several centimetres.
    A tight standard deviation must not make the veto hair-trigger."""
    dad = settled('dad', 1.80, jitter=0.002)   # very consistent
    assert not dad.height_conflicts(1.87)      # 7 cm: still the same person


def test_unsettled_height_never_vetoes():
    p = Person('new')
    p.observe_height(1.75)
    assert not p.height_conflicts(1.20)


def test_missing_measurement_never_vetoes():
    assert not settled('dad', 1.80).height_conflicts(None)


# ── fusion ────────────────────────────────────────────────────────────────────

def registry_with(*people):
    r = PersonRegistry()
    for p in people:
        r._people[p.name] = p
    return r


def test_face_and_voice_agreeing_is_the_strongest():
    r = registry_with(settled('max', 1.28))
    name, conf, why = fuse('max', 'max', 1.28, r)
    assert name == 'max' and conf > 0.9
    assert 'φωνή' in why and 'πρόσωπο' in why


def test_face_alone_is_good():
    r = registry_with(settled('max', 1.28))
    name, conf, _ = fuse('max', None, None, r)
    assert name == 'max' and 0.7 < conf < 0.95


def test_voice_alone_is_good():
    r = registry_with(settled('max', 1.28))
    name, conf, _ = fuse(None, 'max', None, r)
    assert name == 'max' and 0.7 < conf < 0.95


def test_nothing_is_unknown():
    name, conf, why = fuse(None, None, 1.7, PersonRegistry())
    assert name is None and conf == 0.0
    assert 'άγνωστ' in why


def test_face_wins_a_disagreement():
    """A voice can come out of a television; a face in front of the camera
    cannot."""
    r = registry_with(settled('max', 1.28), settled('dad', 1.80))
    name, conf, why = fuse('dad', 'max', 1.80, r)
    assert name == 'dad'
    assert conf < 0.7                       # but held with less confidence
    assert 'διαφων' in why


def test_height_vetoes_the_face_in_a_disagreement():
    r = registry_with(settled('max', 1.28), settled('dad', 1.80))
    # The face says dad, but this person is 1.28 m — so it is not dad.
    name, _, why = fuse('dad', 'max', 1.28, r)
    assert name == 'max'
    assert 'φωνή' in why


def test_height_can_veto_everything():
    r = registry_with(settled('max', 1.28))
    name, conf, why = fuse('max', 'max', 1.85, r)
    assert name is None and conf == 0.0
    assert 'ύψος' in why


def test_height_alone_never_identifies():
    """‼️ Two adults are often within a few centimetres. Height is a veto, not
    an identifier."""
    r = registry_with(settled('dad', 1.80))
    name, _, _ = fuse(None, None, 1.80, r)
    assert name is None


def test_unknown_person_cannot_be_vetoed():
    # A face match against somebody with no learned height still stands.
    r = PersonRegistry()
    r.add('guest')
    name, _, _ = fuse('guest', None, 1.9, r)
    assert name == 'guest'


# ── the registry ──────────────────────────────────────────────────────────────

def test_add_and_list():
    r = PersonRegistry()
    r.add('max'); r.add('dad')
    assert r.names() == ['dad', 'max']
    assert 'max' in r


def test_add_is_idempotent():
    r = PersonRegistry()
    r.add('max'); r.mark_face('max')
    r.add('max')
    assert r.get('max').has_face          # not reset


def test_blank_names_are_rejected():
    r = PersonRegistry()
    assert r.add('  ') is None
    assert len(r) == 0


def test_remove():
    r = PersonRegistry()
    r.add('max')
    assert r.remove('max')
    assert not r.remove('max')


def test_marking_face_and_voice():
    r = PersonRegistry()
    r.mark_face('max')
    r.mark_voice('max')
    p = r.get('max')
    assert p.has_face and p.has_voice


def test_observe_records_height_and_time():
    r = PersonRegistry()
    r.add('max')
    for _ in range(10):
        r.observe('max', height=1.30, now=1000.0)
    assert r.get('max').last_seen == 1000.0
    assert r.get('max').height == pytest.approx(1.30, abs=0.01)


def test_observe_unknown_person_does_nothing():
    assert not PersonRegistry().observe('ghost', height=1.7)


def test_summary_shows_the_gaps():
    r = PersonRegistry()
    r.mark_face('max')
    row = [x for x in r.summary() if x['name'] == 'max'][0]
    assert row['face'] and not row['voice']
    assert row['height'] is None
    assert not row['complete']


def test_summary_marks_a_complete_profile():
    r = PersonRegistry()
    r.mark_face('max'); r.mark_voice('max')
    for _ in range(12):
        r.observe('max', height=1.30)
    row = r.summary()[0]
    assert row['complete']
    assert row['height'] == pytest.approx(1.30, abs=0.01)


# ── persistence ───────────────────────────────────────────────────────────────

def test_round_trip():
    r = PersonRegistry()
    r.mark_face('max'); r.mark_voice('max')
    for _ in range(12):
        r.observe('max', height=1.30)
    restored = PersonRegistry.from_dict(r.to_dict())
    p = restored.get('max')
    assert p.has_face and p.has_voice
    assert p.height == pytest.approx(1.30, abs=0.01)
    assert p.samples == 12


def test_from_dict_tolerates_empty():
    assert len(PersonRegistry.from_dict({})) == 0
    assert len(PersonRegistry.from_dict(None)) == 0


def test_round_trip_preserves_the_veto():
    r = PersonRegistry()
    for _ in range(20):
        r.observe('max', height=1.28) if 'max' in r else r.add('max')
    for _ in range(20):
        r.observe('max', height=1.28)
    restored = PersonRegistry.from_dict(r.to_dict())
    assert restored.get('max').height_conflicts(1.85)
