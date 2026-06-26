#!/usr/bin/env python3
"""Wake word detection — openWakeWord on NPU (VitisAI EP), always-on listener.

Runs a lightweight openWakeWord model continuously on the microphone
stream and publishes `wake_word` (std_msgs/String) with the triggered
model name whenever a wake word is detected. Downstream nodes (the
planned faster-whisper streaming STT node, the LLM bridge, etc.) can
stay idle until they see a message here, instead of running
continuously — see the RAM budget notes in project memory.

Default model is a custom-trained "max" model (`config/models/max.onnx`,
see `training/wake_word_max/`) for the wake word "Max"/"Μαξ", trained on
synthetic edge-tts speech with gain/reverb/noise augmentation (no real
recordings yet — see training/wake_word_max/README for evaluation results,
known limitations and how to improve it with real recordings).
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
    if os.path.isfile(_VAIP_CONFIG):
        return _orig_IS(path, sess_options=sess_options,
                        providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
                        provider_options=[{'config_file': _VAIP_CONFIG}, {}], **kw)
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
from std_msgs.msg import String, Int16MultiArray
import sounddevice as sd
from ament_index_python.packages import get_package_share_directory


def _find_device_by_name(name: str) -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and name.lower() in d['name'].lower():
            return i
    return None

import openwakeword
from openwakeword.model import Model


SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms @ 16kHz — openWakeWord's expected frame size


def _play_beep(freq=880, duration=0.15, sample_rate=16000):
    try:
        import sounddevice as sd
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sd.play(tone, samplerate=sample_rate)
    except Exception:
        pass


class WakeWordNode(Node):
    def __init__(self):
        super().__init__('wake_word_node')

        self.declare_parameter('device_index', -1)
        self.declare_parameter('device_name', 'reSpeaker')
        self.declare_parameter('mic_channels', 6)
        self.declare_parameter('mic_channel', 4)
        self.declare_parameter('model_name', 'max')
        self.declare_parameter('model_path', '')
        self.declare_parameter('threshold', 0.75)
        self.declare_parameter('cooldown', 1.5)
        self.declare_parameter('beep_on_wake', True)

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

        if model_path:
            model_paths = [model_path]
        elif model_name == 'max':
            model_paths = [os.path.join(get_package_share_directory('home_robot'),
                                         'config', 'models', 'max.onnx')]
        else:
            model_paths = [p for p in openwakeword.get_pretrained_model_paths()
                           if model_name in os.path.basename(p)]
            if not model_paths:
                raise ValueError(f'No bundled openWakeWord model matches "{model_name}"')

        self._model = Model(wakeword_model_paths=model_paths, vad_threshold=0.5)

        self.wake_pub = self.create_publisher(String, 'wake_word', 10)
        self.audio_pub = self.create_publisher(Int16MultiArray, 'mic/audio', 200)

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
        self._audio_q.put(chunk)
        msg = Int16MultiArray()
        msg.data = chunk.tolist()
        self.audio_pub.publish(msg)

    def _detect_loop(self):
        while rclpy.ok():
            chunk = self._audio_q.get()
            predictions = self._model.predict(chunk)
            now = time.monotonic()
            for name, score in predictions.items():
                if score < self.threshold:
                    continue
                if now - self._last_trigger.get(name, 0.0) < self.cooldown:
                    continue
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
