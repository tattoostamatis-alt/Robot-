"""Fusing voice identity + direction + faces (home_robot/speaker_fusion.py).

The three signals existed separately for months and nothing combined them.
The interesting cases here are the ones where the honest answer is "I cannot
tell" — two people standing close together, a voice from behind the robot, a
name that has gone stale — because a confident wrong answer ("Μιλάει ο Μαξ"
pointed at the wrong person) is worse than no answer at all.

Robot-free.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_speaker_fusion.py -q
"""
import pytest

from home_robot.speaker_fusion import (
    CAMERA_HFOV_DEG, SpeakerTracker, bearing_to_column, match_face, wrap180,
)

W = 640


def _face(cx, w=90):
    return {'x1': int(cx - w / 2), 'y1': 100,
            'x2': int(cx + w / 2), 'y2': 100 + w, 'score': 0.9}


# ── angle handling ──────────────────────────────────────────────────────────

def test_angles_wrap_the_short_way():
    """350° is 10° to the LEFT. Read as 350 it would point off to the right."""
    assert wrap180(350) == pytest.approx(-10)
    assert wrap180(10) == pytest.approx(10)
    assert wrap180(190) == pytest.approx(-170)
    assert wrap180(180) == pytest.approx(180)


def test_straight_ahead_is_the_middle_of_the_frame():
    assert bearing_to_column(0, W) == pytest.approx(W / 2)


def test_right_of_the_robot_is_a_higher_column():
    assert bearing_to_column(20, W) > W / 2
    assert bearing_to_column(-20, W) < W / 2


def test_a_voice_from_behind_maps_nowhere():
    """Must be None, not clamped to an edge — the speaker is not on screen."""
    for a in (90, 140, 180, -120):
        assert bearing_to_column(a, W) is None, f'{a}° should be out of frame'


def test_the_edge_of_the_field_of_view_is_the_boundary():
    half = CAMERA_HFOV_DEG / 2
    assert bearing_to_column(half - 1, W) is not None
    assert bearing_to_column(half + 1, W) is None


# ── matching a bearing to a face ────────────────────────────────────────────

def test_the_face_in_the_direction_of_the_voice_is_picked():
    left, right = _face(160), _face(480)
    assert match_face(-20, [left, right], W) is left
    assert match_face(20, [left, right], W) is right


def test_two_faces_too_close_together_give_no_answer():
    """‼️ Two people side by side is the normal case in a living room.

    A 4-mic array cannot resolve them, so naming one is a coin flip presented
    as fact.
    """
    assert match_face(0, [_face(300), _face(340)], W) is None


def test_a_face_nowhere_near_the_voice_is_not_matched():
    assert match_face(-25, [_face(600)], W) is None


def test_no_faces_means_no_match():
    assert match_face(0, [], W) is None


def test_a_voice_from_behind_never_matches_a_face_in_front():
    assert match_face(150, [_face(320)], W) is None


# ── the tracker ─────────────────────────────────────────────────────────────

class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def test_a_name_and_a_direction_together_identify_the_speaker():
    clk = Clock()
    tr = SpeakerTracker(clock=clk)
    tr.set_name('Δήμος')
    tr.set_angle(20)
    tr.set_faces([_face(160), _face(480)], W)

    s = tr.snapshot()
    assert s['name'] == 'Δήμος'
    assert s['identified'] is True
    assert s['face'] is not None


def test_unknown_is_not_a_person():
    """The diarizer says 'unknown' when it cannot match — not a name."""
    tr = SpeakerTracker(clock=Clock())
    tr.set_name('unknown')
    tr.set_name('')
    assert tr.snapshot()['name'] is None


def test_a_stale_name_is_dropped():
    """Someone who spoke ten minutes ago is not speaking now."""
    clk = Clock()
    tr = SpeakerTracker(name_ttl=12.0, clock=clk)
    tr.set_name('Δήμος')
    clk.t += 5
    assert tr.snapshot()['name'] == 'Δήμος'
    clk.t += 30
    assert tr.snapshot()['name'] is None


def test_a_stale_bearing_is_dropped():
    clk = Clock()
    tr = SpeakerTracker(angle_ttl=4.0, clock=clk)
    tr.set_angle(30)
    clk.t += 10
    s = tr.snapshot()
    assert s['angle'] is None and s['identified'] is False


def test_stale_faces_cannot_identify_anyone():
    """The person may have walked away since the last frame."""
    clk = Clock()
    tr = SpeakerTracker(face_ttl=2.0, clock=clk)
    tr.set_name('Δήμος')
    tr.set_angle(20)
    tr.set_faces([_face(480)], W)
    clk.t += 30
    s = tr.snapshot()
    assert s['faces_visible'] == 0
    assert s['identified'] is False


def test_a_name_without_a_face_is_still_reported():
    """Someone talking from another room is worth knowing about."""
    tr = SpeakerTracker(clock=Clock())
    tr.set_name('Δήμος')
    s = tr.snapshot()
    assert s['name'] == 'Δήμος' and s['identified'] is False


# ── the line that reaches the LLM ───────────────────────────────────────────

def test_nothing_known_produces_no_line():
    """This goes into every prompt — an empty one costs prefill for nothing."""
    assert SpeakerTracker(clock=Clock()).describe() is None


def test_the_description_names_the_side():
    clk = Clock()
    tr = SpeakerTracker(clock=clk)
    tr.set_name('Δήμος')
    tr.set_angle(40)
    line = tr.describe()
    assert 'Δήμος' in line and 'δεξιά' in line

    tr.set_angle(-40)
    assert 'αριστερά' in tr.describe()

    tr.set_angle(5)
    assert 'μπροστά' in tr.describe()
