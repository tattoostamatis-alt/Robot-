#!/usr/bin/env python3
"""Streaming speech-to-text — faster-whisper, triggered by wake_word.

Stays idle until a `wake_word` (std_msgs/String) message arrives, then
switches to recording mode by consuming audio from the `mic/audio` topic
published by wake_word_node (Int16MultiArray, 16kHz, mono ASR beam).
Sharing the topic avoids opening a second ALSA stream on the same device.

State machine:  idle → waiting_speech → recording → (transcribe thread) → idle

A short pre-roll buffer captures audio that arrived just before the wake
word so the first syllable is not lost.
"""

import collections
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String, Int16MultiArray


SAMPLE_RATE = 16000
CHUNK_SIZE  = 1280   # must match wake_word_node CHUNK_SIZE (80ms @ 16kHz)


class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')

        self.declare_parameter('model_size',        'large-v3')
        self.declare_parameter('language',          'el')
        self.declare_parameter('energy_thresh',     0.065)
        self.declare_parameter('start_timeout',     5.0)
        self.declare_parameter('silence_limit',     1.5)
        self.declare_parameter('max_record_seconds', 12.0)
        self.declare_parameter('preroll_seconds',    0.5)
        self.declare_parameter('wakeword_flush_ms', 300)
        self.declare_parameter('calibrate_on_start', True)

        model_size            = self.get_parameter('model_size').value
        self.lang             = self.get_parameter('language').value
        self.energy_thresh    = self.get_parameter('energy_thresh').value
        self.calibrate_on_start = self.get_parameter('calibrate_on_start').value
        start_timeout         = self.get_parameter('start_timeout').value
        silence_limit         = self.get_parameter('silence_limit').value
        max_record_seconds    = self.get_parameter('max_record_seconds').value
        preroll_seconds       = self.get_parameter('preroll_seconds').value
        flush_ms              = self.get_parameter('wakeword_flush_ms').value

        self._start_timeout_chunks  = max(1, int(start_timeout * SAMPLE_RATE / CHUNK_SIZE))
        self._silence_limit_chunks  = max(1, int(silence_limit * SAMPLE_RATE / CHUNK_SIZE))
        self._max_chunks            = max(1, int(max_record_seconds * SAMPLE_RATE / CHUNK_SIZE))
        preroll_chunks              = max(1, int(preroll_seconds * SAMPLE_RATE / CHUNK_SIZE))
        self._flush_chunks          = max(1, int(flush_ms / 1000 * SAMPLE_RATE / CHUNK_SIZE))

        self._preroll    = collections.deque(maxlen=preroll_chunks)
        self._state      = 'idle'   # idle | flushing | waiting_speech | recording
        self._record_buf = []
        self._wait_count = 0
        self._sil_count  = 0
        self._flush_rem  = 0
        self._lock       = threading.Lock()
        self._busy       = threading.Lock()

        self._whisper = None
        threading.Thread(target=self._load_whisper, args=(model_size,), daemon=True).start()

        self.add_on_set_parameters_callback(self._on_param_change)

        self.text_pub = self.create_publisher(String, 'speech_text', 10)
        self.create_subscription(String,         'wake_word', self._on_wake_word, 10)
        self.create_subscription(Int16MultiArray, 'mic/audio', self._on_audio,    200)

        self.get_logger().info('STT node started — waiting for wake_word')

    def _on_param_change(self, params):
        for p in params:
            if p.name == 'energy_thresh':
                self.energy_thresh = p.value
                self.get_logger().info(f'energy_thresh updated to {p.value:.4f}')
        return SetParametersResult(successful=True)

    def _load_whisper(self, model_size):
        from faster_whisper import WhisperModel
        self._whisper = WhisperModel(model_size, device='cpu', compute_type='int8')
        self.get_logger().info(f'Whisper model "{model_size}" loaded')
        if self.calibrate_on_start:
            self._calibrate_energy_thresh()

    def _calibrate_energy_thresh(self):
        # Measure ambient RMS from /mic/audio topic over ~2s, set threshold = max(0.05, rms*3)
        import time
        chunks = []
        deadline = time.monotonic() + 2.5
        orig_state = self._state

        def _collect(msg):
            if time.monotonic() < deadline:
                chunks.append(
                    np.array(msg.data, dtype=np.int16).astype(np.float32) / 32768.0
                )

        sub = self.create_subscription(Int16MultiArray, 'mic/audio', _collect, 200)
        time.sleep(2.5)
        self.destroy_subscription(sub)

        if chunks:
            all_audio = np.concatenate(chunks)
            ambient_rms = float(np.sqrt(np.mean(all_audio ** 2)))
            new_thresh = max(0.05, ambient_rms * 1.4)
            self.energy_thresh = new_thresh
            self.get_logger().info(
                f'Ambient noise RMS={ambient_rms:.4f} → energy_thresh={new_thresh:.4f}')
        else:
            self.get_logger().warn('Calibration got no audio from /mic/audio, keeping default')

    def _on_wake_word(self, msg: String):
        if self._whisper is None:
            self.get_logger().warn('Whisper not loaded yet, ignoring wake word')
            return
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn('Already transcribing, ignoring wake word')
            return
        with self._lock:
            self._preroll.clear()
            self._state      = 'flushing'
            self._flush_rem  = self._flush_chunks
            self._record_buf = []
            self._wait_count = 0
            self._sil_count  = 0
        self.get_logger().info('Wake word received — listening for speech')

    def _on_audio(self, msg: Int16MultiArray):
        chunk = np.array(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

        with self._lock:
            if self._state == 'idle':
                self._preroll.append(chunk)
                return

            if self._state == 'flushing':
                self._flush_rem -= 1
                if self._flush_rem <= 0:
                    self._state = 'waiting_speech'
                return

            energy = float(np.sqrt(np.mean(chunk ** 2)))

            if self._state == 'waiting_speech':
                self._wait_count += 1
                if energy > self.energy_thresh:
                    self._state      = 'recording'
                    self._record_buf = list(self._preroll) + [chunk]
                    self._sil_count  = 0
                elif self._wait_count >= self._start_timeout_chunks:
                    self.get_logger().info('No speech detected after wake word')
                    self._state = 'idle'
                    self._busy.release()
                return

            if self._state == 'recording':
                self._record_buf.append(chunk)
                if energy < self.energy_thresh:
                    self._sil_count += 1
                else:
                    self._sil_count = 0

                if (self._sil_count  >= self._silence_limit_chunks or
                        len(self._record_buf) >= self._max_chunks):
                    audio = np.concatenate(self._record_buf)
                    self._state = 'idle'
                    threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: np.ndarray):
        try:
            self.get_logger().info(f'Transcribing {len(audio)/SAMPLE_RATE:.1f}s rms={float(np.sqrt(np.mean(audio**2))):.4f}')
            # beam_size=5 + a domain initial_prompt biases the decoder toward
            # the command vocabulary (room names / "πήγαινε στο…"), which
            # markedly improves Greek accuracy over greedy beam_size=1.
            segments, _ = self._whisper.transcribe(
                audio, language=self.lang, beam_size=5,
                no_speech_threshold=0.3, vad_filter=True,
                initial_prompt='Εντολές προς το ρομπότ Μαξ: πήγαινε στην κουζίνα, '
                               'στο σαλόνι, στον διάδρομο, στην τουαλέτα, στο δωμάτιο '
                               'του Μαξ, στο δωμάτιο του μπαμπά, πήγαινε στη βάση.')
            text = ' '.join(s.text for s in segments).strip()
            if text:
                self.get_logger().info(f'Heard: {text}')
                self.text_pub.publish(String(data=text))
            else:
                self.get_logger().info('Transcription empty')
        finally:
            self._busy.release()


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
