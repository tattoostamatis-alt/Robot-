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

Mic capture targets the reSpeaker XVF3800 6-channel firmware. Measured on this
unit 2026-07-26 (8 s of speech, all channels captured simultaneously):

  ch0  peak 0.470  SNR 39.5 dB   processed/AGC beam
  ch1  peak 0.157  SNR 25.1 dB   |
  ch2  peak 0.060  SNR 23.9 dB   | raw capsules — all cluster together
  ch3  peak 0.132  SNR 21.6 dB   |
  ch4  peak 0.077  SNR 21.2 dB   |
  ch5  peak 0.074  SNR 23.9 dB   AEC reference

An earlier version of this docstring claimed ch4 was the ASR beam; the numbers
say otherwise — ch0 is the only channel that stands apart. The two consumers
therefore read different channels: wake detection on `mic_channel` (4, flat
and false-trigger-free) and transcription on `stt_channel` (0, 18 dB cleaner).
For a plain mono/stereo mic set mic_channels=1, mic_channel=0.
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
    wake_decision, utterance_is_risky, IGNORE, SUPPRESS, BARGE_IN,
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
# ~2.4 s of audio. Deep enough to ride out a scheduling hiccup, shallow enough
# that a detector which has fallen behind is noticed rather than silently
# accumulating a minutes-long backlog. See _enqueue().
AUDIO_QUEUE_MAX = 30


# "I'm listening" chime, in the spirit of Hey Siri's two-note acknowledgement.
# The old cue was a single 880 Hz sine at 0.6 amplitude held for 350 ms, which
# the user found grating — a bare sine that starts and stops abruptly clicks at
# both ends and rings for far too long. This is two short notes a fifth apart
# (E6 -> B6), each 80/110 ms, at a third of the volume, each shaped by a
# raised-cosine envelope so it fades in and out instead of clicking. Total
# ~0.2 s, so it never gets in the way of the command you speak next.
CHIME_NOTES = [(1319.0, 0.08), (1976.0, 0.11)]  # (Hz, seconds)
CHIME_GAIN = 0.20


def _play_beep(sample_rate=44100):
    try:
        import sounddevice as sd
        parts = []
        for freq, duration in CHIME_NOTES:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            # Raised-cosine (Hann) envelope — zero at both ends, so no click.
            env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len(t)) / max(1, len(t) - 1)))
            parts.append(CHIME_GAIN * env * np.sin(2 * np.pi * freq * t))
        tone = np.concatenate(parts).astype(np.float32)
        # Use pulse (index 7) — works with any PulseAudio output device
        sd.play(tone, samplerate=sample_rate, device=7)
        sd.wait()
    except Exception:
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
        # Channel forwarded to STT. -1 = same as mic_channel (single-channel
        # mics); on the XVF3800 set it to the processed beam. See the channel
        # measurements in _audio_callback for why this differs from mic_channel.
        self.declare_parameter('stt_channel', -1)
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
        stt_ch = self.get_parameter('stt_channel').value
        self.stt_channel = (self.mic_channel if stt_ch is None or stt_ch < 0
                            else min(stt_ch, self.mic_channels - 1))

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
        # True while the reply being spoken contains the wake word itself.
        # Set from speech_response, cleared when playback ends.
        self._risky_utterance = False

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
        # The same text tts_node is about to speak, so a reply containing the
        # wake word can disable barge-in for its duration.
        self.create_subscription(String, 'speech_response',
                                 self._on_speech_response, 10)

        # ‼️ BOUNDED. This was an unbounded Queue fed from the realtime audio
        # callback and drained by the model. Whenever prediction fell behind
        # realtime — which on this box happens whenever flm/Whisper/YOLO are
        # loaded, see the swap-thrashing history — the backlog grew without
        # limit: memory climbed AND every detection was scored against audio
        # that got progressively older, so the wake word "stopped working" while
        # actually firing many seconds late. Dropping the OLDEST chunk is the
        # right trade: stale audio is worthless for wake detection.
        # diarization_node has used a bounded queue all along; this matches it.
        self._audio_q = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
        self._dropped_chunks = 0
        self._last_trigger = {}

        threading.Thread(target=self._listen_loop, args=(device_index,), daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

        self.get_logger().info(
            f'Wake word node started — models: {list(self._model.models.keys())}, '
            f'threshold={self.threshold}')

    def _listen_loop(self, device_index):
        """Own the mic, and KEEP owning it.

        ‼️ This used to be a bare `with sd.InputStream(...)`. If the device went
        away — the USB hub these peripherals hang off glitches, and the array
        has dropped out before — the stream raised, the `with` unwound, this
        thread ended, and nothing was ever published on mic/audio again. Wake
        word AND STT went permanently deaf with no error line anywhere, node
        still alive and apparently healthy. Reopen instead, and say so.
        """
        device = None if device_index < 0 else device_index
        backoff = 1.0
        while rclpy.ok():
            try:
                with sd.InputStream(samplerate=SAMPLE_RATE,
                                    channels=self.mic_channels, dtype='int16',
                                    blocksize=CHUNK_SIZE, device=device,
                                    callback=self._audio_callback):
                    self.get_logger().info('Microphone stream open')
                    backoff = 1.0
                    self._hp_zi = None       # re-seed the IIR on the new stream
                    while rclpy.ok():
                        time.sleep(0.1)
            except Exception as exc:                      # noqa: BLE001
                if not rclpy.ok():
                    return
                self.get_logger().error(
                    f'Microphone stream lost ({exc}) — reopening in {backoff:.0f}s')
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, self.mic_channel].copy()
        # Wake detection and STT read DIFFERENT channels on purpose — they want
        # opposite things from the array. Measured 2026-07-26 over 8 s of real
        # speech, all six channels at once:
        #
        #     ch0  peak 0.470  SNR 39.5 dB   <- processed/AGC beam
        #     ch1  peak 0.157  SNR 25.1 dB
        #     ch2  peak 0.060  SNR 23.9 dB
        #     ch3  peak 0.132  SNR 21.6 dB
        #     ch4  peak 0.077  SNR 21.2 dB   <- what both used to read
        #     ch5  peak 0.074  SNR 23.9 dB
        #
        # ch0 stands alone; ch1-5 cluster together like raw capsules. So the
        # module docstring had it backwards: ch0 is the processed beam, not ch4.
        # Whisper was being fed a raw capsule 18 dB noisier and 6x quieter than
        # what the array can produce, which is why transcriptions came back as
        # plausible-but-wrong Greek ("Πάση μπανταριά έχεις", "Σας ευχαριστώ").
        #
        # But ch0 must NOT drive wake detection: its AGC lifts the noise floor
        # during silence and the model false-fires on it — 9 triggers in 75 s at
        # 0.95-1.00 (2026-07-25), each one publishing tts/stop so the robot cut
        # off its own speech. ch4 stays flat and gave 0 false triggers in 30 s.
        # Hence: detect on ch4, transcribe on ch0.
        if self._hp_b is not None:
            fchunk = chunk.astype(np.float32)
            if self._hp_zi is None:                 # seed on the first chunk
                self._hp_zi = self._hp_zi_template * fchunk[0]
            filt, self._hp_zi = lfilter(self._hp_b, self._hp_a,
                                        fchunk, zi=self._hp_zi)
            self._enqueue(np.clip(filt, -32768, 32767).astype(np.int16))
        else:
            self._enqueue(chunk)
        # STT gets the full band (Whisper needs it) off the transcription channel.
        stt_chunk = (chunk if self.stt_channel == self.mic_channel
                     else indata[:, self.stt_channel].copy())
        msg = Int16MultiArray()
        msg.data = stt_chunk.tolist()
        self.audio_pub.publish(msg)

    def _enqueue(self, chunk):
        """Hand a chunk to the detector, dropping the OLDEST if it is behind.

        Never blocks: this runs on the realtime audio callback, where waiting
        for the model would stall capture and cost us the next chunk too.
        """
        try:
            self._audio_q.put_nowait(chunk)
        except queue.Full:
            try:
                self._audio_q.get_nowait()      # discard the stalest chunk
            except queue.Empty:
                pass
            self._dropped_chunks += 1
            if self._dropped_chunks % 100 == 1:
                self.get_logger().warn(
                    f'Wake detection behind realtime — dropped '
                    f'{self._dropped_chunks} chunk(s). CPU contention?')
            try:
                self._audio_q.put_nowait(chunk)
            except queue.Full:
                pass

    def _on_tts_speaking(self, msg: Bool):
        self._gate.set_speaking(msg.data)
        if not msg.data:
            # Playback finished — the risk belonged to that utterance only.
            # Cleared on the falling edge rather than when the next reply
            # arrives, so a long silence cannot leave the gate clamped shut.
            self._risky_utterance = False

    def _on_speech_response(self, msg: String):
        """The text the robot is about to say, straight from llm_bridge.

        Arrives before tts/speaking goes True (the TTS node has to synthesise
        first), so the flag is set by the time playback starts. Knowing the
        text is what lets barge-in stay on: only replies that actually contain
        the wake word are dangerous, and those are refused outright instead of
        being trusted to a threshold.
        """
        self._risky_utterance = utterance_is_risky(msg.data)
        if self._risky_utterance:
            self.get_logger().debug(
                'Reply contains the wake word — barge-in disabled for it')

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
                    barge_in_threshold=self.barge_in_threshold,
                    risky_utterance=self._risky_utterance)

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
