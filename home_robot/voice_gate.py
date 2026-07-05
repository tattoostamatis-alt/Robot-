"""Shared barge-in / self-echo gate for the voice pipeline.

The TTS node publishes `tts/speaking` (std_msgs/Bool): True while it is
synthesising/playing a response, False once playback finishes. The always-on
listeners (wake_word_node, stt_node) feed those messages into a `SpeakingGate`
and consult :meth:`SpeakingGate.suppressed` before acting on microphone audio,
so the robot does not trigger its own wake word or start recording on the sound
of its own voice (the "TTS self-echo" problem).

A release *tail* keeps the gate closed for a short while after playback ends,
because room reverb and the tail end of the audio device buffer keep leaking
the robot's voice into the mic for a beat after `sd.wait()` returns.

This module is deliberately dependency-free (no ROS, no audio libraries) so the
gate logic can be unit-tested without a robot or a running graph — see
`tests/test_voice_gate.py`.
"""

import time

TOPIC = 'tts/speaking'


class SpeakingGate:
    """Tracks whether TTS is (or was just) speaking.

    Parameters
    ----------
    release_tail:
        Seconds to keep :meth:`suppressed` True after speaking stops.
    clock:
        Monotonic clock callable, injectable for tests. Must be monotonic —
        never pass ``time.time`` (wall clock can jump).
    """

    def __init__(self, release_tail: float = 0.3, clock=time.monotonic):
        self._clock = clock
        self._release_tail = max(0.0, float(release_tail))
        self._speaking = False
        self._release_at = 0.0

    def set_speaking(self, speaking: bool) -> None:
        """Record a `tts/speaking` message. Arms the release tail on True→False."""
        if speaking:
            self._speaking = True
        elif self._speaking:
            self._speaking = False
            self._release_at = self._clock() + self._release_tail

    @property
    def speaking(self) -> bool:
        """True iff TTS is playing right now (ignores the release tail)."""
        return self._speaking

    def suppressed(self, now: float | None = None) -> bool:
        """True while mic input should be ignored: speaking, or within the tail."""
        if self._speaking:
            return True
        now = self._clock() if now is None else now
        return now < self._release_at
