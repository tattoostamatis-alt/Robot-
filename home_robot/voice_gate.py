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
# Barge-in: a listener publishes Bool(True) here to abort TTS mid-utterance.
# tts_node subscribes and calls sd.stop() + drains its queue; wake_word_node
# publishes it when the wake word fires while the robot is speaking.
STOP_TOPIC = 'tts/stop'


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


# ── wake-word decision (pure) ────────────────────────────────────────────

# What to do with a wake-word detection, given the gate state. Kept as plain
# strings so the decision can be unit-tested without ROS.
IGNORE = 'ignore'        # below threshold — do nothing
SUPPRESS = 'suppress'    # self-echo / reverb tail — drop it
BARGE_IN = 'barge_in'    # user interrupted active TTS — abort TTS, start a turn
WAKE = 'wake'            # normal wake — start a turn


# Words that, coming out of the robot's own mouth, can be mistaken for the wake
# phrase by the detector. "Έι ρομπότ" is the phrase, but the model fires on the
# "ρομπότ" half often enough that any reply containing it is a self-echo risk.
# Accent-free: compared against strip_accents() output.
_SELF_WAKE_WORDS = ('ρομποτ', 'robot')


def utterance_is_risky(text: str) -> bool:
    """Is the robot about to say something that could wake itself?

    Barge-in's whole difficulty is that the hardware AEC does not fully remove
    the robot's own voice from the wake-word channel, so a reply is judged on
    the same audio as a real interruption. But most replies are harmless — the
    detector only false-fires on the ones that actually contain the wake word.

    Knowing WHICH utterance is dangerous costs nothing (the text is already in
    hand before playback) and lets barge-in stay permissive for the other 95%
    of replies instead of being switched off wholesale, which is what it had
    been since 2026-07-24.
    """
    if not text:
        return False
    from .stop_command import strip_accents
    flat = strip_accents(text)
    return any(w in flat for w in _SELF_WAKE_WORDS)


def wake_decision(score: float, threshold: float, *,
                  suppressed: bool, speaking: bool,
                  suppress_on_tts: bool, allow_barge_in: bool,
                  barge_in_threshold: float,
                  risky_utterance: bool = False) -> str:
    """Classify a wake-word detection.

    The subtlety is telling a real interruption apart from the robot's own voice
    leaking into the mic ("self-echo"). While TTS plays we only accept a wake as
    a genuine barge-in if it clears a *stricter* `barge_in_threshold` — self-echo
    false positives tend to be brief marginal spikes, whereas a person saying the
    whole "Έι ρομπότ" scores high and sustained. Anything during the reverb tail
    (suppressed but no longer speaking) is always dropped.

    `risky_utterance` says the reply now playing contains the wake word itself
    (see :func:`utterance_is_risky`). Then no score is trustworthy — the robot
    is literally saying the thing the detector listens for — so barge-in is
    refused outright for the duration rather than gambling on a threshold. This
    is what makes it safe to leave barge-in ON in general: the case that made it
    interrupt itself every few seconds is exactly this one.
    """
    if score < threshold:
        return IGNORE
    if suppress_on_tts and suppressed:
        if (allow_barge_in and speaking and not risky_utterance
                and score >= barge_in_threshold):
            return BARGE_IN
        return SUPPRESS
    return WAKE
