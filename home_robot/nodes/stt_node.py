#!/usr/bin/env python3
"""Streaming speech-to-text — faster-whisper, triggered by wake_word.

Stays idle (no mic stream open) until a `wake_word` (std_msgs/String)
message arrives from `wake_word_node`. Then opens the mic, waits for the
user to start speaking (energy-based VAD, with a timeout in case the wake
word fired on a false positive), records until a period of silence, and
publishes the transcription on `speech_text` (std_msgs/String) for
downstream nodes (the planned LLM bridge, etc.).

The faster-whisper model itself is loaded once at startup in a background
thread and kept resident — only the mic stream is opened/closed per
utterance.

Mic capture follows the same conventions as voice_control.py / wake_word_node
for the reSpeaker XVF3800 (2-channel firmware, channel 1 = "ASR" beam). For
testing on a plain mono/stereo laptop mic, `mic_channel: 0` works too.
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd


SAMPLE_RATE = 16000
BLOCK_SIZE = 8000  # 0.5s chunks


class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')

        self.declare_parameter('model_size', 'base')
        self.declare_parameter('language', 'el')
        self.declare_parameter('device_index', -1)
        self.declare_parameter('mic_channels', 2)
        self.declare_parameter('mic_channel', 1)
        self.declare_parameter('energy_thresh', 0.01)
        self.declare_parameter('start_timeout', 5.0)
        self.declare_parameter('silence_limit', 1.5)
        self.declare_parameter('max_record_seconds', 12.0)

        model_size = self.get_parameter('model_size').value
        self.lang = self.get_parameter('language').value
        self.device_index = self.get_parameter('device_index').value
        self.mic_channels = self.get_parameter('mic_channels').value
        self.mic_channel = self.get_parameter('mic_channel').value
        self.energy_thresh = self.get_parameter('energy_thresh').value
        self.start_timeout = self.get_parameter('start_timeout').value
        self.silence_limit = self.get_parameter('silence_limit').value
        self.max_record_seconds = self.get_parameter('max_record_seconds').value

        self.text_pub = self.create_publisher(String, 'speech_text', 10)

        self._whisper = None
        self._busy = threading.Lock()

        threading.Thread(target=self._load_whisper, args=(model_size,), daemon=True).start()

        self.create_subscription(String, 'wake_word', self._on_wake_word, 10)

        self.get_logger().info('STT node started — waiting for wake_word')

    def _load_whisper(self, model_size):
        from faster_whisper import WhisperModel
        self._whisper = WhisperModel(model_size, device='cpu', compute_type='int8')
        self.get_logger().info(f'Whisper model "{model_size}" loaded')

    def _on_wake_word(self, msg: String):
        if self._whisper is None:
            self.get_logger().warn('Wake word detected but whisper model not loaded yet, ignoring')
            return
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already transcribing, ignoring wake word')
            return
        threading.Thread(target=self._record_and_transcribe, daemon=True).start()

    def _record_and_transcribe(self):
        try:
            audio = self._record_utterance()
            if audio is None:
                self.get_logger().info('No speech detected after wake word')
                return

            segments, _ = self._whisper.transcribe(audio, language=self.lang, beam_size=1)
            text = ' '.join(s.text for s in segments).strip()

            if not text:
                self.get_logger().info('Transcription empty')
                return

            self.get_logger().info(f'Heard: {text}')
            self.text_pub.publish(String(data=text))
        finally:
            self._busy.release()

    def _record_utterance(self):
        audio_q = []
        state = {'started': False, 'silence_chunks': 0, 'total_chunks': 0}
        done = threading.Event()

        start_timeout_chunks = max(1, int(self.start_timeout * SAMPLE_RATE / BLOCK_SIZE))
        silence_limit_chunks = max(1, int(self.silence_limit * SAMPLE_RATE / BLOCK_SIZE))
        max_chunks = max(1, int(self.max_record_seconds * SAMPLE_RATE / BLOCK_SIZE))

        def callback(indata, frames, time_info, status):
            chunk = indata[:, self.mic_channel].copy()
            energy = np.sqrt(np.mean(chunk ** 2))
            state['total_chunks'] += 1

            if not state['started']:
                if energy > self.energy_thresh:
                    state['started'] = True
                    audio_q.append(chunk)
                elif state['total_chunks'] >= start_timeout_chunks:
                    done.set()
                return

            audio_q.append(chunk)
            if energy < self.energy_thresh:
                state['silence_chunks'] += 1
            else:
                state['silence_chunks'] = 0

            if (state['silence_chunks'] >= silence_limit_chunks
                    or len(audio_q) >= max_chunks):
                done.set()

        device = None if self.device_index < 0 else self.device_index
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=self.mic_channels, dtype='float32',
                             blocksize=BLOCK_SIZE, device=device, callback=callback):
            done.wait()

        if not state['started']:
            return None
        return np.concatenate(audio_q).flatten()

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
