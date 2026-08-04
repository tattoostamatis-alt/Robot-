"""Unit tests for echolocation (home_robot/echo.py)."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.echo import (  # noqa: E402
    SAMPLE_RATE, EchoSignature, chirp, compare, energy_envelope,
    first_reflection, reflection_distance, rt60_estimate,
)


def decaying(rt60, seconds=1.2, rate=SAMPLE_RATE, echo_at=None):
    """Synthetic room response: a burst that decays exponentially.

    `rt60` is the seconds to fall 60 dB, which is what the estimator should
    recover. `echo_at` adds a discrete reflection that many seconds later —
    ‼️ deliberately QUIETER than the direct sound, or the "echo" becomes the
    loudest thing in the recording and there is nothing after the peak to find.
    """
    n = int(seconds * rate)
    t = np.arange(n) / rate
    # -60 dB over rt60 seconds => amplitude factor 10**(-3 * t / rt60)
    env = 10.0 ** (-3.0 * t / rt60)
    noise = np.random.RandomState(0).randn(n) * env
    if echo_at:
        i = int(echo_at * rate)
        if i < n:
            # 1.8x the local decay level: a clear bump, still well below the
            # direct sound at t=0.
            noise[i:i + 240] *= 1.8
    return noise


# ── the chirp ─────────────────────────────────────────────────────────────────

def test_chirp_length_and_range():
    c = chirp(seconds=0.05)
    assert len(c) == int(0.05 * SAMPLE_RATE)
    assert abs(c).max() <= 1.0


def test_chirp_fades_at_both_ends():
    """A hard edge rings louder than the room does."""
    c = chirp(seconds=0.05)
    # The very first and last samples sit where the window is near zero. A
    # wider slice includes the end of the ramp, which is already at full
    # amplitude — that is the fade working, not failing.
    assert abs(c[0]) < 0.05
    assert abs(c[-1]) < 0.05
    assert abs(c).max() > 0.5


def test_chirp_sweeps_upward():
    # Zero crossings should be denser at the end than at the start.
    c = chirp(seconds=0.1, f0=500, f1=5000)
    half = len(c) // 2
    early = np.sum(np.diff(np.sign(c[:half])) != 0)
    late = np.sum(np.diff(np.sign(c[half:])) != 0)
    assert late > early * 1.5


def test_zero_length_chirp_is_safe():
    assert len(chirp(seconds=0)) >= 1


# ── the envelope ──────────────────────────────────────────────────────────────

def test_envelope_peaks_at_zero_db():
    env = energy_envelope(decaying(0.4))
    assert max(env) == pytest.approx(0.0, abs=1e-6)


def test_envelope_decays():
    env = energy_envelope(decaying(0.4))
    assert env[0] > env[len(env) // 2] > env[-1]


def test_empty_input_is_empty():
    assert energy_envelope([]) == []
    assert energy_envelope(np.zeros(0)) == []


def test_silence_is_empty_not_a_crash():
    assert energy_envelope(np.zeros(1000)) == []


# ── RT60 ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('rt', [0.25, 0.5, 0.9])
def test_rt60_recovers_the_synthetic_decay(rt):
    est = rt60_estimate(energy_envelope(decaying(rt, seconds=2.0)))
    assert est is not None
    assert abs(est - rt) < rt * 0.35


def test_a_ringing_room_measures_longer_than_a_dead_one():
    live = rt60_estimate(energy_envelope(decaying(0.9, seconds=2.5)))
    dead = rt60_estimate(energy_envelope(decaying(0.2, seconds=2.5)))
    assert live > dead


def test_no_decay_in_the_window_is_unknown():
    """‼️ Extrapolating a 60 dB decay from a recording that never fell that far
    would invent a number."""
    flat = np.ones(4000) * 0.5
    assert rt60_estimate(energy_envelope(flat)) is None


def test_empty_envelope_is_unknown():
    assert rt60_estimate([]) is None


# ── reflections ───────────────────────────────────────────────────────────────

def test_a_discrete_echo_is_found():
    # ‼️ A short rt60, so the direct sound has decayed enough by 30 ms that a
    # 1.8x bump is a bump and not the new loudest thing in the recording.
    env = energy_envelope(decaying(0.25, echo_at=0.030))
    t = first_reflection(env)
    assert t is not None
    assert abs(t - 0.030) < 0.012


def test_smooth_decay_has_no_distinct_reflection():
    assert first_reflection(energy_envelope(decaying(0.25))) is None


def test_distance_is_there_and_back():
    assert reflection_distance(0.030) == pytest.approx(5.145, abs=0.01)


def test_no_reflection_no_distance():
    assert reflection_distance(None) is None


# ── signatures and comparison ─────────────────────────────────────────────────

def sig(rt, seconds=2.0, echo=None, room='saloni'):
    data = decaying(rt, seconds=seconds, echo_at=echo)
    env = energy_envelope(data)
    return EchoSignature(rt60_estimate(env), first_reflection(env), env, room)


def test_the_same_room_twice_reads_as_unchanged():
    a, b = sig(0.5), sig(0.5)
    changed, why = compare(a, b)
    assert not changed
    assert 'ίδιο' in why


def test_a_much_longer_reverb_is_a_change():
    changed, why = compare(sig(0.9), sig(0.4))
    assert changed
    assert 'περισσότερο' in why


def test_a_much_shorter_reverb_is_a_change():
    changed, why = compare(sig(0.3), sig(0.8))
    assert changed
    assert 'λιγότερο' in why


def test_no_baseline_is_not_a_change():
    """The first measurement in a room cannot be a change."""
    changed, why = compare(sig(0.5), None)
    assert not changed
    assert 'χωρίς' in why


def test_missing_current_is_not_a_change():
    assert compare(None, sig(0.5))[0] is False


def test_signature_round_trip():
    s = sig(0.25, echo=0.03)
    r = EchoSignature.from_dict(s.to_dict())
    assert r.rt60 == pytest.approx(s.rt60)
    assert r.room == s.room
    assert len(r.envelope) == len(s.envelope)


def test_signature_truncates_the_envelope():
    s = EchoSignature(envelope=list(range(1000)))
    assert len(s.envelope) == 200


def test_from_dict_tolerates_empty():
    s = EchoSignature.from_dict({})
    assert s.rt60 is None and s.envelope == []


def test_distance_property():
    s = EchoSignature(reflection_s=0.02)
    assert s.distance == pytest.approx(3.43, abs=0.01)


def test_a_late_wobble_is_not_a_wall():
    """‼️ From the first live run: an unbounded search called a fluctuation
    190 ms into the tail a reflection, and reported a wall 32.6 m away — in a
    flat. Anything past ~60 ms is tail noise, not a surface."""
    env = energy_envelope(decaying(0.6, seconds=2.0, echo_at=0.190))
    assert first_reflection(env) is None


def test_a_reflection_inside_a_room_is_still_found():
    env = energy_envelope(decaying(0.25, echo_at=0.030))
    assert first_reflection(env) is not None


def test_the_bound_is_configurable():
    env = energy_envelope(decaying(0.6, seconds=2.0, echo_at=0.190))
    assert first_reflection(env, max_gap_ms=400.0) is not None
