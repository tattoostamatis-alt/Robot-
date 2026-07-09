"""Tests for the barge-in / self-echo gate (see home_robot/voice_gate.py).

Two layers, both robot-free:

  * Unit tests of SpeakingGate with an injected fake clock — the actual
    suppress-while-speaking + release-tail logic.
  * Static wiring checks that the TTS node publishes `tts/speaking` and both
    listeners subscribe to it and consult the gate, so the three-node contract
    can't silently drift apart.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_voice_gate.py -q
"""
import os

import pytest

from home_robot.voice_gate import (
    TOPIC, STOP_TOPIC, SpeakingGate,
    wake_decision, IGNORE, SUPPRESS, BARGE_IN, WAKE,
)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = f'{PKG}/home_robot/nodes'


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


# ── SpeakingGate unit tests ───────────────────────────────────────────

def test_idle_gate_never_suppresses():
    clk = FakeClock()
    gate = SpeakingGate(release_tail=0.3, clock=clk)
    assert not gate.speaking
    assert not gate.suppressed()


def test_suppresses_while_speaking():
    clk = FakeClock()
    gate = SpeakingGate(release_tail=0.3, clock=clk)
    gate.set_speaking(True)
    assert gate.speaking
    assert gate.suppressed()
    # Time passing does not release while still speaking.
    clk.t = 100.0
    assert gate.suppressed()


def test_release_tail_holds_then_releases():
    clk = FakeClock(t=10.0)
    gate = SpeakingGate(release_tail=0.3, clock=clk)
    gate.set_speaking(True)
    gate.set_speaking(False)
    assert not gate.speaking
    # Still suppressed during the tail.
    clk.t = 10.2
    assert gate.suppressed()
    # Boundary is exclusive: exactly at release time it's open again.
    clk.t = 10.3
    assert not gate.suppressed()
    clk.t = 10.4
    assert not gate.suppressed()


def test_zero_tail_releases_immediately():
    clk = FakeClock(t=5.0)
    gate = SpeakingGate(release_tail=0.0, clock=clk)
    gate.set_speaking(True)
    gate.set_speaking(False)
    assert not gate.suppressed()


def test_negative_tail_clamped_to_zero():
    clk = FakeClock(t=5.0)
    gate = SpeakingGate(release_tail=-1.0, clock=clk)
    gate.set_speaking(True)
    gate.set_speaking(False)
    assert not gate.suppressed()


def test_redundant_false_does_not_rearm_tail():
    clk = FakeClock(t=0.0)
    gate = SpeakingGate(release_tail=0.3, clock=clk)
    gate.set_speaking(True)
    gate.set_speaking(False)   # arms tail until 0.3
    clk.t = 0.5                # tail already expired
    gate.set_speaking(False)   # must NOT re-arm from the True→False guard
    assert not gate.suppressed()


def test_speaking_again_reextends_suppression():
    clk = FakeClock(t=0.0)
    gate = SpeakingGate(release_tail=0.3, clock=clk)
    gate.set_speaking(True)
    gate.set_speaking(False)
    clk.t = 0.4                # released
    assert not gate.suppressed()
    gate.set_speaking(True)    # new utterance
    assert gate.suppressed()


def test_explicit_now_overrides_clock():
    clk = FakeClock(t=0.0)
    gate = SpeakingGate(release_tail=0.3, clock=clk)
    gate.set_speaking(True)
    gate.set_speaking(False)   # release at 0.3
    assert gate.suppressed(now=0.29)
    assert not gate.suppressed(now=0.31)


# ── wake_decision (pure) ──────────────────────────────────────────────

_KW = dict(suppress_on_tts=True, allow_barge_in=True, barge_in_threshold=0.7)


def test_below_threshold_ignored():
    assert wake_decision(0.4, 0.5, suppressed=False, speaking=False, **_KW) == IGNORE


def test_normal_wake_when_idle():
    assert wake_decision(0.6, 0.5, suppressed=False, speaking=False, **_KW) == WAKE


def test_self_echo_while_speaking_below_barge_threshold_suppressed():
    # Confident enough to wake, but not confident enough to barge in → drop it,
    # so a marginal self-echo spike can't interrupt the robot.
    assert wake_decision(0.6, 0.5, suppressed=True, speaking=True, **_KW) == SUPPRESS


def test_barge_in_when_speaking_and_above_barge_threshold():
    assert wake_decision(0.8, 0.5, suppressed=True, speaking=True, **_KW) == BARGE_IN


def test_reverb_tail_never_barges_in():
    # Suppressed but no longer actively speaking (tail) → always dropped, even
    # at a high score.
    assert wake_decision(0.9, 0.5, suppressed=True, speaking=False, **_KW) == SUPPRESS


def test_barge_in_disabled_suppresses_even_high_score():
    kw = dict(suppress_on_tts=True, allow_barge_in=False, barge_in_threshold=0.7)
    assert wake_decision(0.95, 0.5, suppressed=True, speaking=True, **kw) == SUPPRESS


def test_suppress_disabled_wakes_through_tts():
    kw = dict(suppress_on_tts=False, allow_barge_in=False, barge_in_threshold=0.7)
    assert wake_decision(0.6, 0.5, suppressed=True, speaking=True, **kw) == WAKE


# ── Three-node wiring contract ────────────────────────────────────────

def test_tts_publishes_speaking_topic():
    src = open(f'{NODES}/tts_node.py').read()
    assert 'SPEAKING_TOPIC' in src and 'create_publisher' in src, \
        'tts_node must publish the tts/speaking topic'
    assert '_set_speaking(True)' in src and '_set_speaking(False)' in src, \
        'tts_node must flip speaking around playback'


@pytest.mark.parametrize('node', ['wake_word_node.py', 'stt_node.py'])
def test_listeners_subscribe_and_gate(node):
    src = open(f'{NODES}/{node}').read()
    assert 'SpeakingGate' in src, f'{node} must use SpeakingGate'
    assert f'{SpeakingGate.suppressed.__name__}' in src, \
        f'{node} must consult gate.suppressed()'
    assert 'SPEAKING_TOPIC' in src and 'create_subscription' in src, \
        f'{node} must subscribe to tts/speaking'


def test_topic_name_is_stable():
    # The literal listeners/publisher agree on; guards an accidental rename.
    assert TOPIC == 'tts/speaking'


# ── Barge-in wiring contract ──────────────────────────────────────────

def test_stop_topic_name_is_stable():
    assert STOP_TOPIC == 'tts/stop'


def test_tts_honors_barge_in_stop():
    src = open(f'{NODES}/tts_node.py').read()
    assert 'STOP_TOPIC' in src and 'create_subscription' in src, \
        'tts_node must subscribe to tts/stop'
    assert 'sd.stop()' in src, 'tts_node must abort playback on barge-in'


def test_wake_word_publishes_barge_in_only_while_speaking():
    src = open(f'{NODES}/wake_word_node.py').read()
    assert 'allow_barge_in' in src, 'wake_word_node needs an allow_barge_in gate'
    assert 'STOP_TOPIC' in src, 'wake_word_node must publish tts/stop'
    # Barge-in keys off actual speaking, not the reverb tail.
    assert '_gate.speaking' in src
