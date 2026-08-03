"""The hardware VAD gate in stt_node, and above all how it FAILS.

Why the gate exists: the energy threshold cannot tell a voice from a fan
spike — both are simply "loud" — so a gust, a door or the base's own motors
can open a recording. The XVF3800's DSP has a real speech flag, published by
doa_node on /voice_activity, and it can.

Why most of this file is about failure: a gate on the listening path is the
most dangerous thing in this pipeline. A latched `False` reaching it would
make the robot silently deaf to every command, with no error anywhere — the
exact shape of the busy-lock bug that cost a live session. So the rule is that
the gate FAILS OPEN, and that is what is tested hardest here.

Contract:
  * start:  energy AND vad          — both must agree
  * keep:   vad may only EXTEND     — it can never cut a recording short
  * broken: gate disappears         — never blocks

Robot-free: ROS and the audio stack are stubbed, reusing the stt fixture.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_stt_hw_vad.py -q
"""
import os
import sys
import time

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_audio_pipeline_regressions import stt      # noqa: E402,F401
from test_safety_regressions import _ns              # noqa: E402


LOUD = 0.5          # well above the default energy threshold
QUIET = 0.0


def _chunk(node, amplitude):
    """A mic chunk of the given RMS, as Int16MultiArray-ish."""
    n = 512
    data = np.full(n, int(amplitude * 32767), dtype=np.int16)
    return _ns(data=data.tolist())


def _arm(node):
    """Put the node where it is waiting for the user to start speaking."""
    node._state = 'waiting_speech'
    node._wait_count = 0


def _say_vad(node, on, age=0.0):
    """Pretend /voice_activity last reported `on`, `age` seconds ago."""
    now = time.monotonic() - age
    node._vad_seen = True
    node._vad_on = on
    node._vad_at = now
    if on:
        node._vad_true_at = now


# ── fail open: the cases that must never make the robot deaf ────────────────

def test_no_vad_publisher_at_all_leaves_the_gate_wide_open(stt):
    """use_doa:=false, no ReSpeaker, doa_node not built — all the same thing.

    The topic never produces a message, and the robot must behave exactly as
    it did before the gate existed.
    """
    mod, node = stt
    assert node._vad_seen is False, 'fixture should start with no VAD seen'
    assert node._vad_gate_open() is True, (
        'the gate blocks when /voice_activity was never published — this would '
        'make the robot deaf wherever doa_node is not running')

    _arm(node)
    node._on_audio(_chunk(node, LOUD))
    assert node._state == 'recording', 'energy alone must still start a turn'


def test_a_dead_publisher_stops_gating_instead_of_latching_shut(stt):
    """doa_node crashes mid-session with the flag latched False.

    This is the dangerous one: transient_local means the last value sticks
    around, so a naive implementation keeps reading False for ever and the
    robot never records again.
    """
    mod, node = stt
    _say_vad(node, False, age=node.vad_stale_s + 5.0)

    assert node._vad_gate_open() is True, (
        'a stale VAD still gates — the robot would be permanently deaf after '
        'doa_node dies')

    _arm(node)
    node._on_audio(_chunk(node, LOUD))
    assert node._state == 'recording'


def test_the_stale_fallback_is_logged_once_not_every_chunk(stt):
    """It must say why it stopped trusting the VAD, without flooding."""
    mod, node = stt
    warnings = []
    # The stub builds a fresh logger per call, so patching the returned object
    # would be thrown away — replace get_logger itself.
    logger = _ns(info=lambda *a, **k: None, warn=warnings.append,
                 error=lambda *a, **k: None, debug=lambda *a, **k: None,
                 exception=lambda *a, **k: None)
    node.get_logger = lambda: logger
    _say_vad(node, False, age=node.vad_stale_s + 5.0)

    for _ in range(50):
        node._vad_gate_open()

    assert len(warnings) == 1, f'logged {len(warnings)} times, expected once'
    assert 'doa_node' in warnings[0], 'the warning does not say what to check'


def test_the_gate_can_be_switched_off_entirely(stt):
    mod, node = stt
    node.use_hw_vad = False
    _say_vad(node, False)                       # VAD actively says "no speech"

    assert node._vad_gate_open() is True
    _arm(node)
    node._on_audio(_chunk(node, LOUD))
    assert node._state == 'recording', 'use_hw_vad:=false did not disable it'


# ── the gate doing its job ──────────────────────────────────────────────────

def test_a_loud_noise_that_is_not_speech_does_not_start_a_recording(stt):
    """The whole point: a fan spike is loud, but the DSP knows it is not a voice."""
    mod, node = stt
    _say_vad(node, False)
    _arm(node)

    node._on_audio(_chunk(node, LOUD))

    assert node._state == 'waiting_speech', (
        'a loud non-speech burst opened a recording — the VAD gate is not '
        'being consulted on speech onset')


def test_a_loud_noise_that_IS_speech_starts_a_recording(stt):
    mod, node = stt
    _say_vad(node, True)
    _arm(node)

    node._on_audio(_chunk(node, LOUD))

    assert node._state == 'recording'


def test_silence_never_starts_a_recording_even_if_the_vad_is_high(stt):
    """The VAD is a second opinion, not a replacement for the energy check."""
    mod, node = stt
    _say_vad(node, True)
    _arm(node)

    node._on_audio(_chunk(node, QUIET))

    assert node._state == 'waiting_speech'


def test_the_flag_is_held_briefly_across_the_gaps_between_words(stt):
    """The DSP drops the flag between words at 10 Hz; a hard read would flap."""
    mod, node = stt
    _say_vad(node, True)
    # It just went low, but said "speech" a moment ago.
    node._vad_on = False
    node._vad_at = time.monotonic()
    node._vad_true_at = time.monotonic() - (node.vad_hold_s / 2)

    assert node._vad_gate_open() is True, 'the hold window is not applied'

    node._vad_true_at = time.monotonic() - (node.vad_hold_s * 3)
    assert node._vad_gate_open() is False, 'the hold window never expires'


# ── one-directional: it may extend, never cut ───────────────────────────────

def test_the_vad_keeps_a_recording_open_through_a_quiet_syllable(stt):
    mod, node = stt
    _say_vad(node, True)
    node._state = 'recording'
    node._record_buf = [np.zeros(512, dtype=np.float32)]
    node._sil_count = 3

    node._on_audio(_chunk(node, QUIET))       # energy dip, VAD still speech

    assert node._sil_count == 0, (
        'the silence counter was not reset while the DSP still heard speech')
    assert node._state == 'recording'


def test_the_vad_can_never_cut_a_recording_short(stt):
    """‼️ Truncating the user mid-sentence is worse than a slightly long clip.

    Even with the VAD flatly reporting "no speech", a loud chunk must keep the
    recording alive — the energy threshold and max_record_seconds are what end
    a turn.
    """
    mod, node = stt
    _say_vad(node, False)
    node._state = 'recording'
    node._record_buf = [np.zeros(512, dtype=np.float32)]
    node._sil_count = 0

    node._on_audio(_chunk(node, LOUD))

    assert node._state == 'recording', 'the VAD ended a recording'
    assert node._sil_count == 0, 'a VAD-driven silence count is creeping in'


# ── wiring ──────────────────────────────────────────────────────────────────

def test_the_subscription_matches_the_latched_publisher():
    """Durability must match doa_node or DDS silently never connects.

    An incompatible-QoS subscription is not an error — it simply never
    receives, so the gate would sit disabled while looking correctly wired.
    """
    import re
    src = open(f'{PKG}/home_robot/nodes/stt_node.py').read()
    m = re.search(r"'voice_activity'.*?\)\s*\)", src, re.S)
    assert m, 'stt_node does not subscribe to voice_activity'
    assert 'TRANSIENT_LOCAL' in m.group(0), (
        'the voice_activity subscription is not transient_local — it will '
        'never match doa_node latched publisher')


def test_doa_publishes_what_stt_subscribes_to():
    """Both ends of the topic, checked against each other."""
    import re
    doa = open(f'{PKG}/home_robot/nodes/doa_node.py').read()
    stt_src = open(f'{PKG}/home_robot/nodes/stt_node.py').read()
    assert re.search(r"create_publisher\(\s*\n?\s*Bool,\s*'voice_activity'", doa), \
        'doa_node no longer publishes voice_activity'
    assert "'voice_activity'" in stt_src
