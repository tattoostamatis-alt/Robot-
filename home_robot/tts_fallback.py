"""Offline Greek speech for when edge-tts will not answer.

edge-tts is a free, unauthenticated Microsoft endpoint and it was the only
voice on this robot. Measured 2026-07-26: it fails ~20% of the time (throttling
— it closes the stream having sent no audio, or returns an empty one). The 3x
retry in tts_node recovers most of those, but not all, and nothing recovers a
dropped internet connection or an outage.

That failure mode is the most misleading one the robot has: STT heard, the LLM
understood and answered, and the only thing missing is sound — so it reads as
"the robot is broken" when the whole pipeline worked. A local voice removes the
network from the one step that decides whether the user perceives a reply.

Piper (~61 MB ONNX, runs on CPU in ~0.1 s per utterance) is deliberately the
*fallback*, not the default: edge-tts sounds markedly better and works most of
the time. This only has to be intelligible.

Loading is lazy — a robot whose network is fine never pays the 0.7 s model
load or the memory.
"""
import os
import threading

import numpy as np

# Downloaded by scripts/, kept out of the ROS package data on purpose.
DEFAULT_MODEL = os.path.expanduser(
    '~/robot_ws/src/home_robot/config/models/piper/el_GR-rapunzelina-low.onnx')


class PiperFallback:
    """Lazily-loaded local TTS. Never raises on construction.

    `available()` answers whether the model file exists without importing or
    loading anything, so tts_node can log the situation once at startup rather
    than discovering it mid-outage.
    """

    def __init__(self, model_path=DEFAULT_MODEL, logger=None):
        self.model_path = model_path
        self._log = logger
        self._voice = None
        self._lock = threading.Lock()
        self._failed = False        # a broken model should not be retried forever

    def available(self):
        return bool(self.model_path) and os.path.isfile(self.model_path)

    def _load(self):
        """Import and load on first real use. Returns the voice or None."""
        if self._voice is not None or self._failed:
            return self._voice
        with self._lock:
            if self._voice is not None or self._failed:
                return self._voice
            try:
                from piper import PiperVoice
                self._voice = PiperVoice.load(self.model_path)
            except Exception as e:                          # noqa: BLE001
                self._failed = True
                if self._log:
                    self._log.error(f'Piper fallback unusable: {e!r}')
            return self._voice

    def synthesize(self, text):
        """(float32 mono in [-1,1], sample_rate) or None if unavailable.

        Piper emits its own rate (16 kHz for this voice) rather than edge-tts's
        24 kHz, so the rate is returned alongside the audio — resampling would
        only add a dependency and a way to get pitch wrong.
        """
        if not self.available():
            return None
        voice = self._load()
        if voice is None:
            return None
        try:
            chunks = list(voice.synthesize(text))
        except Exception as e:                              # noqa: BLE001
            if self._log:
                self._log.error(f'Piper synthesis failed: {e!r}')
            return None
        if not chunks:
            return None
        audio = np.concatenate([c.audio_float_array for c in chunks])
        if audio.size == 0:
            return None
        return audio.astype(np.float32), int(chunks[0].sample_rate)
