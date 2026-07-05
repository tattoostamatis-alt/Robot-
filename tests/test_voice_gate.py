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

from home_robot.voice_gate import TOPIC, SpeakingGate

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
