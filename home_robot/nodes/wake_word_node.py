#!/usr/bin/env python3
"""Wake word detection — openWakeWord on NPU (VitisAI EP), always-on listener.

Runs a lightweight openWakeWord model continuously on the microphone
stream and publishes `wake_word` (std_msgs/String) with the triggered
model name whenever a wake word is detected. Downstream nodes (the
planned faster-whisper streaming STT node, the LLM bridge, etc.) can
stay idle until they see a message here, instead of running
continuously — see the RAM budget notes in project memory.

Default model is a custom-trained "hey_robot" model
(`config/models/hey_robot.onnx`, see `training/wake_word_hey_robot/`) for
the wake phrase "Έι ρομπότ" / "Hey robot" — replaced "Ρομπότ Μαξ" on
2026-07-25. Plain "ρομπότ" (which shows up in ordinary commands) and a
bare "έι"/"hey" are trained in as hard negatives, as is the retired
"Ρομπότ Μαξ", so only the two words together fire. Trained on synthetic
edge-tts speech with gain/reverb/noise augmentation — see
training/wake_word_hey_robot/README for evaluation results, known
limitations and how to improve it with real recordings.
`model_name` can be set to one of
openWakeWord's bundled pretrained English models (alexa, hey_jarvis,
hey_mycroft, hey_marvin, timer, weather) for pipeline testing instead.
`model_path` overrides both and points directly at a custom
.onnx/.tflite file (e.g. a newly retrained model before it's copied
into `config/models/`).

Mic capture targets the reSpeaker XVF3800 6-channel firmware:
  ch 0-3 = raw mics, ch 4 = ASR beam (beamformed+noise-suppressed),
  ch 5 = AEC reference.  mic_channel=4 is the right input for wake word.
For testing on a plain mono/stereo mic, set mic_channels=1, mic_channel=0.
"""

import os
import sys

# VitisAI EP setup — must happen before any onnxruntime import.
_VENV_SITE = '/home/dimi/ryzenai_venv/lib/python3.12/site-packages'
if os.path.isdir(_VENV_SITE):
    sys.path.insert(0, _VENV_SITE)
os.environ.setdefault('XILINX_XRT', '/opt/xilinx/xrt')
os.environ.setdefault('RYZEN_AI_INSTALLATION_PATH', '/home/dimi/ryzenai_venv')

import onnxruntime as _ort

_VAIP_CONFIG = '/home/dimi/ryzenai_venv/voe-4.0-linux_x86_64/vaip_config.json'
_orig_IS = _ort.InferenceSession

def _npu_session(path, sess_options=None, providers=None, provider_options=None, **kw):
    # CPU-only: VAIP compiler crashes on this graph (same pattern as YOLO NPU bug)
    return _orig_IS(path, sess_options=sess_options,
                    providers=['CPUExecutionProvider'], **kw)

# Patch ort globally so openWakeWord's model + utils + vad all use NPU.
import openwakeword.model as _oww_model
import openwakeword.utils as _oww_utils
import openwakeword.vad as _oww_vad
_oww_model.ort.InferenceSession = _npu_session
_oww_utils.ort.InferenceSession = _npu_session
_oww_vad.ort.InferenceSession   = _npu_session

import queue
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Int16MultiArray
import sounddevice as sd
from scipy.signal import butter, lfilter, lfilter_zi
from ament_index_python.packages import get_package_share_directory

from home_robot.voice_gate import (
    SpeakingGate, TOPIC as SPEAKING_TOPIC, STOP_TOPIC,
    wake_decision, IGNORE, SUPPRESS, BARGE_IN, WAKE,
)


def _find_device_by_name(name: str) -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and name.lower() in d['name'].lower():
            return i
    return None

import openwakeword
from openwakeword.model import Model


SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms @ 16kHz — openWakeWord's expected frame size


def _play_beep(freq=880, duration=0.35, sample_rate=44100):
    try:
        import sounddevice as sd
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        # Use pulse (index 7) — works with any PulseAudio output device
        sd.play(tone, samplerate=sample_rate, device=7)
        sd.wait()
    except Exception as e:
        try:
            import subprocess
            subprocess.Popen(['paplay', '/usr/share/sounds/freedesktop/stereo/bell.oga'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


class WakeWordNode(Node):
    def __init__(self):
        super().__init__('wake_word_node')

        self.declare_parameter('device_index', -1)
        self.declare_parameter('device_name', '')
        self.declare_parameter('mic_channels', 3)
        self.declare_parameter('mic_channel', 0)
        self.declare_parameter('model_name', 'hey_robot')
        self.declare_parameter('model_path', '')
        self.declare_parameter('threshold', 0.50)
        self.declare_parameter('cooldown', 1.5)
        self.declare_parameter('beep_on_wake', True)
        # Barge-in / self-echo gate: ignore wake detections while the robot is
        # speaking (+ a reverb tail) so its own TTS can't trigger the wake word.
        self.declare_parameter('suppress_on_tts', True)
        self.declare_parameter('tts_release_tail', 0.3)
        # Barge-in: if True, saying the wake word *while the robot is speaking*
        # aborts the TTS (publishes tts/stop) and starts a new turn, instead of
        # being suppressed as self-echo. Relies on the XVF3800's hardware AEC
        # (mic_channel=4 ASR beam) keeping the robot's own voice off the mic —
        # leave False on a plain mic without echo cancellation, or it will
        # interrupt itself. The reverb *tail* after speech is always suppressed.
        self.declare_parameter('allow_barge_in', False)
        # Stricter threshold to accept a wake as a real barge-in (vs the robot's
        # own voice leaking through imperfect AEC). Higher than `threshold` so a
        # marginal self-echo spike can't interrupt the robot mid-sentence.
        self.declare_parameter('barge_in_threshold', 0.70)
        # High-pass cutoff (Hz) applied to the mic channel before wake detection.
        # The XVF3800 ASR beam carries strong low-frequency noise on this rig
        # (mini-PC fan / mains hum, dominant 50-250 Hz) that the "max" model
        # scores as a wake at ~1.00 — measured 8-17 false triggers per 20 s of
        # silence on every channel. Speech formants live at 500-3000 Hz, so a
        # 2nd-order high-pass at 250 Hz removes the noise (0 false triggers in
        # 20 s) without touching the wake word. Set to 0 to disable.
        self.declare_parameter('highpass_hz', 250.0)

        device_index = self.get_parameter('device_index').value
        device_name = self.get_parameter('device_name').value
        self.mic_channels = self.get_parameter('mic_channels').value
        self.mic_channel = self.get_parameter('mic_channel').value

        if device_index < 0 and device_name:
            device_index = _find_device_by_name(device_name) or -1
        model_name = self.get_parameter('model_name').value
        model_path = self.get_parameter('model_path').value
        self.threshold = self.get_parameter('threshold').value
        self.cooldown = self.get_parameter('cooldown').value
        self.beep_on_wake = self.get_parameter('beep_on_wake').value
        self.suppress_on_tts = self.get_parameter('suppress_on_tts').value
        self.allow_barge_in = self.get_parameter('allow_barge_in').value
        self.barge_in_threshold = self.get_parameter('barge_in_threshold').value
        self._gate = SpeakingGate(
            release_tail=self.get_parameter('tts_release_tail').value)

        # Stateful high-pass filter for the mic channel (see highpass_hz above).
        # lfilter_zi seeds the filter state so the very first chunks aren't a
        # transient; the state is carried across callbacks for a continuous IIR.
        highpass_hz = self.get_parameter('highpass_hz').value
        if highpass_hz and highpass_hz > 0:
            self._hp_b, self._hp_a = butter(2, highpass_hz / (SAMPLE_RATE / 2),
                                            btype='high')
            # Keep the unit-step steady state as a template; scale it by the very
            # first sample so the filter starts already settled to the incoming
            # DC/noise level. Without this the IIR warm-up transient itself
            # false-triggers the model for the first ~1 s (2 spurious wakes seen).
            self._hp_zi_template = lfilter_zi(self._hp_b, self._hp_a)
            self._hp_zi = None
            self.get_logger().info(f'High-pass filter @ {highpass_hz:.0f} Hz on mic')
        else:
            self._hp_b = None

        if model_path:
            model_paths = [model_path]
        elif model_name == 'hey_robot':
            model_paths = [os.path.join(get_package_share_directory('home_robot'),
                                         'config', 'models', 'hey_robot.onnx')]
        else:
            model_paths = [p for p in openwakeword.get_pretrained_model_paths()
                           if model_name in os.path.basename(p)]
            if not model_paths:
                raise ValueError(f'No bundled openWakeWord model matches "{model_name}"')

        self._model = Model(wakeword_model_paths=model_paths, vad_threshold=0.5)

        self.wake_pub = self.create_publisher(String, 'wake_word', 10)
        self.stop_pub = self.create_publisher(Bool, STOP_TOPIC, 10)
        self.audio_pub = self.create_publisher(Int16MultiArray, 'mic/audio', 200)
        self.create_subscription(Bool, SPEAKING_TOPIC, self._on_tts_speaking, 10)

        self._audio_q = queue.Queue()
        self._last_trigger = {}

        threading.Thread(target=self._listen_loop, args=(device_index,), daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

        self.get_logger().info(
            f'Wake word node started — models: {list(self._model.models.keys())}, '
            f'threshold={self.threshold}')

    def _listen_loop(self, device_index):
        device = None if device_index < 0 else device_index
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=self.mic_channels, dtype='int16',
                            blocksize=CHUNK_SIZE, device=device,
                            callback=self._audio_callback):
            while rclpy.ok():
                time.sleep(0.1)

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, self.mic_channel].copy()
        # Wake detection runs on the high-pass-filtered signal (kills fan/mains
        # rumble that false-triggers the model). The STT stream on mic/audio
        # keeps the raw chunk — Whisper needs the full band for transcription.
        if self._hp_b is not None:
            fchunk = chunk.astype(np.float32)
            if self._hp_zi is None:                 # seed on the first chunk
                self._hp_zi = self._hp_zi_template * fchunk[0]
            filt, self._hp_zi = lfilter(self._hp_b, self._hp_a,
                                        fchunk, zi=self._hp_zi)
            self._audio_q.put(np.clip(filt, -32768, 32767).astype(np.int16))
        else:
            self._audio_q.put(chunk)
        msg = Int16MultiArray()
        msg.data = chunk.tolist()
        self.audio_pub.publish(msg)

    def _on_tts_speaking(self, msg: Bool):
        self._gate.set_speaking(msg.data)

    def _detect_loop(self):
        while rclpy.ok():
            chunk = self._audio_q.get()
            predictions = self._model.predict(chunk)
            now = time.monotonic()
            for name, score in predictions.items():
                decision = wake_decision(
                    score, self.threshold,
                    suppressed=self._gate.suppressed(now),
                    speaking=self._gate.speaking,
                    suppress_on_tts=self.suppress_on_tts,
                    allow_barge_in=self.allow_barge_in,
                    barge_in_threshold=self.barge_in_threshold)

                if decision == IGNORE:
                    continue
                if decision == SUPPRESS:
                    self.get_logger().debug(
                        f'Wake "{name}" ({score:.2f}) suppressed — TTS speaking/tail')
                    continue
                # WAKE or BARGE_IN both start a new turn; rate-limit either.
                if now - self._last_trigger.get(name, 0.0) < self.cooldown:
                    continue
                if decision == BARGE_IN:
                    self.get_logger().info(
                        f'Barge-in: "{name}" ({score:.2f}) while speaking → stopping TTS')
                    self.stop_pub.publish(Bool(data=True))
                self._last_trigger[name] = now
                self.get_logger().info(f'Wake word "{name}" detected (score={score:.2f})')
                if self.beep_on_wake:
                    threading.Thread(target=_play_beep, daemon=True).start()
                self.wake_pub.publish(String(data=name))

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = WakeWordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
