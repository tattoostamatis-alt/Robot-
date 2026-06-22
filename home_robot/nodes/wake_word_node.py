#!/usr/bin/env python3
"""Wake word detection — openWakeWord, always-on listener.

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

Mic capture follows the same conventions as stt_node.py for the
reSpeaker XVF3800 (2-channel firmware, channel 1 = "ASR" beam). For
testing on a plain mono/stereo laptop mic, `mic_channel: 0` works too.
"""

import os
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd
from ament_index_python.packages import get_package_share_directory

import openwakeword
from openwakeword.model import Model


SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms @ 16kHz — openWakeWord's expected frame size


class WakeWordNode(Node):
    def __init__(self):
        super().__init__('wake_word_node')

        self.declare_parameter('device_index', -1)
        self.declare_parameter('mic_channels', 2)
        self.declare_parameter('mic_channel', 1)
        self.declare_parameter('model_name', 'max')
        self.declare_parameter('model_path', '')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('cooldown', 1.5)

        device_index = self.get_parameter('device_index').value
        self.mic_channels = self.get_parameter('mic_channels').value
        self.mic_channel = self.get_parameter('mic_channel').value
        model_name = self.get_parameter('model_name').value
        model_path = self.get_parameter('model_path').value
        self.threshold = self.get_parameter('threshold').value
        self.cooldown = self.get_parameter('cooldown').value

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

        self._model = Model(wakeword_model_paths=model_paths)

        self.wake_pub = self.create_publisher(String, 'wake_word', 10)

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
        self._audio_q.put(indata[:, self.mic_channel].copy())

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
