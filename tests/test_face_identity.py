"""Unit tests for face identity (home_robot/face_identity.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.face_identity import (  # noqa: E402
    REFERENCE_LANDMARKS, FaceGallery, PresenceTracker, cosine_similarity,
)


def vec(*xs):
    return list(xs)


MAX = vec(1.0, 0.0, 0.0, 0.0)
MAX2 = vec(0.9, 0.3, 0.1, 0.0)          # same person, different frame (~0.94)
DAD = vec(0.0, 1.0, 0.0, 0.0)
STRANGER = vec(0.0, 0.0, 1.0, 0.0)


# ── cosine_similarity ─────────────────────────────────────────────────────────

def test_identical_vectors_are_one():
    assert cosine_similarity(MAX, MAX) == pytest.approx(1.0)


def test_orthogonal_vectors_are_zero():
    assert cosine_similarity(MAX, DAD) == pytest.approx(0.0)


def test_scale_does_not_matter():
    assert cosine_similarity(MAX, [5.0, 0, 0, 0]) == pytest.approx(1.0)


def test_zero_vector_is_zero_not_a_crash():
    assert cosine_similarity(MAX, [0.0, 0, 0, 0]) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_similar_vectors_score_high():
    assert cosine_similarity(MAX, MAX2) > 0.9


# ── reference landmarks ───────────────────────────────────────────────────────

def test_reference_template_has_five_points():
    """SFace is trained on crops aligned to exactly these five points; a
    different count means the alignment silently stops matching training."""
    assert len(REFERENCE_LANDMARKS) == 5
    for x, y in REFERENCE_LANDMARKS:
        assert 0 < x < 112 and 0 < y < 112


def test_eyes_are_above_the_mouth_in_the_template():
    left_eye, right_eye, nose, left_mouth, right_mouth = REFERENCE_LANDMARKS
    assert left_eye[1] < nose[1] < left_mouth[1]
    assert right_eye[1] < nose[1] < right_mouth[1]


# ── FaceGallery ───────────────────────────────────────────────────────────────

def test_empty_gallery_matches_nobody():
    name, score = FaceGallery().match(MAX)
    assert name is None
    assert score == 0.0


def test_enrolled_person_is_recognised():
    g = FaceGallery()
    g.enrol('max', MAX)
    name, score = g.match(MAX2)
    assert name == 'max'
    assert score > 0.9


def test_a_stranger_is_not_guessed_at():
    """‼️ Below threshold the answer is None. Calling a stranger by a family
    member's name is worse than saying nothing."""
    g = FaceGallery()
    g.enrol('max', MAX)
    g.enrol('mpampas', DAD)
    name, score = g.match(STRANGER)
    assert name is None
    assert score < g.threshold


def test_the_right_person_wins_between_two():
    g = FaceGallery()
    g.enrol('max', MAX)
    g.enrol('mpampas', DAD)
    assert g.match(DAD)[0] == 'mpampas'
    assert g.match(MAX2)[0] == 'max'


def test_multiple_embeddings_improve_a_match():
    """One embedding is one angle under one light. A second sighting should let
    a frame that missed the first still land."""
    profile = vec(0.3, 0.95, 0.0, 0.0)       # ~0.30 against MAX: below threshold
    g = FaceGallery()
    g.enrol('max', MAX)
    assert g.match(profile)[0] != 'max'          # too far from the frontal one
    g.enrol('max', profile)
    assert g.match(profile)[0] == 'max'


def test_gallery_is_bounded_per_person():
    g = FaceGallery(max_per_person=3)
    for i in range(10):
        g.enrol('max', vec(1.0, i * 0.01, 0, 0))
    assert len(g.to_dict()['people']['max']) == 3


def test_enrolling_needs_a_name_and_a_vector():
    g = FaceGallery()
    assert g.enrol('', MAX) == 0
    assert g.enrol('max', []) == 0
    assert len(g) == 0


def test_names_are_listed_sorted():
    g = FaceGallery()
    g.enrol('mpampas', DAD)
    g.enrol('max', MAX)
    assert g.names() == ['max', 'mpampas']


def test_forget_removes_a_person():
    g = FaceGallery()
    g.enrol('max', MAX)
    assert g.forget('max')
    assert g.match(MAX)[0] is None
    assert not g.forget('max')


def test_threshold_is_configurable():
    strict = FaceGallery(threshold=0.99)     # MAX2 scores ~0.94
    strict.enrol('max', MAX)
    assert strict.match(MAX2)[0] is None         # 0.9-ish is not enough now


def test_gallery_round_trip():
    g = FaceGallery()
    g.enrol('max', MAX)
    g.enrol('mpampas', DAD)
    restored = FaceGallery.from_dict(g.to_dict())
    assert restored.names() == g.names()
    assert restored.match(MAX2)[0] == 'max'


def test_from_dict_tolerates_empty():
    assert len(FaceGallery.from_dict({})) == 0


# ── PresenceTracker ───────────────────────────────────────────────────────────

def test_one_frame_is_not_presence():
    p = PresenceTracker(confirm=3)
    assert p.update(['max'], now=1000.0) == []
    assert p.present() == []


def test_confirmed_after_enough_frames():
    p = PresenceTracker(confirm=3)
    p.update(['max'], 1000.0)
    p.update(['max'], 1001.0)
    assert p.update(['max'], 1002.0) == ['max']
    assert p.present() == ['max']


def test_arrival_is_reported_once():
    p = PresenceTracker(confirm=2)
    p.update(['max'], 1000.0)
    assert p.update(['max'], 1001.0) == ['max']
    for t in range(1002, 1010):
        assert p.update(['max'], float(t)) == []      # still here, not news


def test_a_missed_frame_decays_rather_than_resets():
    p = PresenceTracker(confirm=3)
    p.update(['max'], 1000.0)
    p.update(['max'], 1001.0)
    p.update([], 1002.0)                              # looked away
    p.update(['max'], 1003.0)
    assert p.update(['max'], 1004.0) == ['max']


def test_presence_expires_after_a_long_absence():
    p = PresenceTracker(confirm=2, forget_after=25.0)
    p.update(['max'], 1000.0)
    p.update(['max'], 1001.0)
    assert p.present() == ['max']
    assert p.present(now=1100.0) == []


def test_returning_after_expiry_is_a_new_arrival():
    p = PresenceTracker(confirm=2, forget_after=25.0)
    p.update(['max'], 1000.0)
    p.update(['max'], 1001.0)
    p.update([], 1100.0)                              # expired
    p.update(['max'], 1101.0)
    assert p.update(['max'], 1102.0) == ['max']


def test_two_people_tracked_independently():
    p = PresenceTracker(confirm=2)
    p.update(['max', 'mpampas'], 1000.0)
    arrived = p.update(['max', 'mpampas'], 1001.0)
    assert sorted(arrived) == ['max', 'mpampas']
    assert p.present() == ['max', 'mpampas']


def test_unknown_names_are_ignored():
    p = PresenceTracker(confirm=1)
    assert p.update([None, ''], 1000.0) == []


def test_last_seen_is_recorded():
    p = PresenceTracker(confirm=1)
    p.update(['max'], 1234.0)
    assert p.last_seen('max') == 1234.0
    assert p.last_seen('nobody') is None


def test_reset_clears_everything():
    p = PresenceTracker(confirm=1)
    p.update(['max'], 1000.0)
    p.reset()
    assert p.present() == []
