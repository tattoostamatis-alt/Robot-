"""Tests for the offline TTS fallback (see home_robot/tts_fallback.py).

The failure this guards against is the misleading one: edge-tts fails, the
robot goes silent, and the user reads a working pipeline as a broken robot.
So the tests cover both halves — the fallback itself, and the wiring that
decides it actually gets used.

Model-free by default: PiperFallback never loads anything until synthesize()
is called, so absence/failure paths are testable with no 61 MB download. The
one test that needs the real voice skips when it is missing.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_tts_fallback.py -q
"""
import os

import numpy as np
import pytest

from home_robot.tts_fallback import DEFAULT_MODEL, PiperFallback

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTS_NODE = f'{PKG}/home_robot/nodes/tts_node.py'


# ── availability, without loading anything ────────────────────────────
def test_missing_model_is_unavailable():
    assert not PiperFallback('/nonexistent/voice.onnx').available()


def test_empty_path_is_unavailable():
    assert not PiperFallback('').available()


def test_missing_model_synthesizes_to_none():
    """Must return None, not raise — the caller is already in an error path."""
    assert PiperFallback('/nonexistent/voice.onnx').synthesize('γεια') is None


def test_available_does_not_load_the_model(tmp_path):
    """available() is called at startup on every boot; it must stay cheap."""
    fake = tmp_path / 'voice.onnx'
    fake.write_bytes(b'not a real model')
    fb = PiperFallback(str(fake))
    assert fb.available()
    assert fb._voice is None          # nothing loaded yet


def test_broken_model_fails_once_then_stays_failed(tmp_path):
    """A corrupt model must not be retried on every single utterance."""
    fake = tmp_path / 'voice.onnx'
    fake.write_bytes(b'not a real model')
    fb = PiperFallback(str(fake))
    assert fb.synthesize('γεια') is None
    assert fb._failed
    assert fb.synthesize('ξανά') is None


def test_synthesis_error_returns_none_not_raises(monkeypatch, tmp_path):
    fake = tmp_path / 'voice.onnx'
    fake.write_bytes(b'x')
    fb = PiperFallback(str(fake))

    class Boom:
        def synthesize(self, text):
            raise RuntimeError('onnx exploded')

    monkeypatch.setattr(fb, '_load', lambda: Boom())
    assert fb.synthesize('γεια') is None


def test_empty_chunk_list_returns_none(monkeypatch, tmp_path):
    fake = tmp_path / 'voice.onnx'
    fake.write_bytes(b'x')
    fb = PiperFallback(str(fake))

    class Silent:
        def synthesize(self, text):
            return iter(())

    monkeypatch.setattr(fb, '_load', lambda: Silent())
    assert fb.synthesize('γεια') is None


# ── the real voice ────────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.isfile(DEFAULT_MODEL),
                    reason='Greek piper voice not downloaded')
def test_real_voice_speaks_greek():
    fb = PiperFallback()
    out = fb.synthesize('Δεν έχω μπαταρία, δουλεύω με powerbank.')
    assert out is not None
    audio, rate = out
    assert rate > 0
    assert audio.dtype == np.float32
    # long enough to be that sentence, and actually audible
    assert len(audio) / rate > 0.8
    assert np.abs(audio).max() > 0.01


@pytest.mark.skipif(not os.path.isfile(DEFAULT_MODEL),
                    reason='Greek piper voice not downloaded')
def test_real_voice_stays_in_range():
    """Out-of-range samples clip audibly through sounddevice."""
    audio, _ = PiperFallback().synthesize('Πηγαίνω στην κουζίνα.')
    assert np.abs(audio).max() <= 1.0


# ── wiring: the fallback must actually be reachable ───────────────────
@pytest.fixture(scope='module')
def node_src():
    with open(TTS_NODE) as f:
        return f.read()


def test_node_falls_back_when_edge_tts_raises(node_src):
    """_synthesize raises RuntimeError after its 3 retries; that has to be
    caught, or the fallback can never run."""
    body = node_src.split('def _synthesize_and_play')[1].split('async def _synthesize')[0]
    assert 'except Exception' in body
    assert 'self._fallback.synthesize(text)' in body


def test_node_reraises_when_no_fallback(node_src):
    """With no local voice the original error must still surface, not be
    swallowed into silence."""
    body = node_src.split('def _synthesize_and_play')[1].split('async def _synthesize')[0]
    assert 'raise' in body


def test_playback_uses_per_utterance_rate(node_src):
    """The local voice is 16 kHz; playing it at the edge-tts constant of
    24 kHz would speed it up and raise the pitch."""
    body = node_src.split('def _synthesize_and_play')[1].split('async def _synthesize')[0]
    assert 'samplerate=rate' in body
    assert 'len(audio) / rate' in body


def test_fallback_is_configurable(node_src):
    assert "declare_parameter('fallback_model'" in node_src


def test_startup_warns_when_fallback_missing(node_src):
    """Finding out mid-outage is exactly too late."""
    assert 'No local TTS fallback' in node_src
