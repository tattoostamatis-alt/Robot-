#!/usr/bin/env python3
"""Voice control — reSpeaker XVF3800 + faster-whisper + Qwen3 → robot commands.

The XVF3800 (2-channel USB firmware, the default) presents itself as a
stereo capture device:
  channel 0 = "Conference" — AEC + beamformed/post-processed audio
  channel 1 = "ASR"        — auto-selected-beam output, tuned for speech
                              recognition (used here by default)
Find the device with `python3 -c "import sounddevice as sd; print(sd.query_devices())"`
and set the `device_index` parameter accordingly.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
import sounddevice as sd
import numpy as np
import queue
import threading
import json
import ollama


SAMPLE_RATE   = 16000
BLOCK_SIZE    = 8000   # 0.5s chunks
SILENCE_LIMIT = 2.0    # seconds of silence → end of utterance
ENERGY_THRESH = 0.01

# Wake word "Max" — Whisper (el) may transcribe it as Latin or Greek script
WAKE_WORDS = ('max', 'μαξ', 'μαχ')


SYSTEM_PROMPT = """Είσαι ο "Max", ο controller ενός home robot. Μετατρέπεις φωνητικές εντολές σε JSON actions.
Διαθέσιμες actions:
- {"action": "tidy", "room": "living_room|bedroom|kitchen|all"}
- {"action": "goto", "location": "living_room|bedroom|kitchen|dock"}
- {"action": "dock"}
- {"action": "stop"}
- {"action": "patrol"}
- {"action": "report_clutter"}
- {"action": "unknown"}  (αν δεν καταλαβαίνεις την εντολή)

Κάθε JSON πρέπει επιπλέον να έχει το πεδίο "reply": μια σύντομη, φιλική απάντηση
στα Ελληνικά προς τον χρήστη (π.χ. "Εντάξει, πάω στην κουζίνα").

Απάντα ΜΟΝΟ με valid JSON, τίποτα άλλο. Πάντα στα Ελληνικά."""


class VoiceControl(Node):
    def __init__(self):
        super().__init__('voice_control')

        self.declare_parameter('model_size', 'base')
        self.declare_parameter('language', 'el')
        self.declare_parameter('device_index', -1)
        # XVF3800 2-channel firmware: 0 = Conference (AEC), 1 = ASR beam.
        self.declare_parameter('mic_channels', 2)
        self.declare_parameter('mic_channel', 1)

        model_size   = self.get_parameter('model_size').value
        self.lang    = self.get_parameter('language').value
        device_index = self.get_parameter('device_index').value
        self.mic_channels = self.get_parameter('mic_channels').value
        self.mic_channel  = self.get_parameter('mic_channel').value

        self.cmd_pub      = self.create_publisher(String, 'voice_command',  10)
        self.response_pub = self.create_publisher(String, 'voice_response', 10)
        self.dock_pub     = self.create_publisher(Bool,   'dock',           10)
        self.goal_pub     = self.create_publisher(PoseStamped, 'goal_pose', 10)

        self._audio_q  = queue.Queue()
        self._whisper  = None

        threading.Thread(target=self._load_whisper, args=(model_size,), daemon=True).start()
        threading.Thread(target=self._listen_loop, args=(device_index,), daemon=True).start()

        self.get_logger().info('Voice control started — listening...')

    def _load_whisper(self, model_size):
        from faster_whisper import WhisperModel
        self._whisper = WhisperModel(model_size, device='cpu', compute_type='int8')
        self.get_logger().info(f'Whisper model "{model_size}" loaded')

    def _listen_loop(self, device_index):
        device = None if device_index < 0 else device_index
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=self.mic_channels, dtype='float32',
                            blocksize=BLOCK_SIZE, device=device,
                            callback=self._audio_callback):
            while rclpy.ok():
                self._process_speech()

    def _audio_callback(self, indata, frames, time, status):
        self._audio_q.put(indata[:, self.mic_channel].copy())

    def _process_speech(self):
        if self._whisper is None:
            return

        audio_chunks = []
        silence_chunks = 0
        silence_limit_chunks = int(SILENCE_LIMIT * SAMPLE_RATE / BLOCK_SIZE)

        # Wait for speech to start
        while True:
            chunk = self._audio_q.get()
            if np.sqrt(np.mean(chunk ** 2)) > ENERGY_THRESH:
                audio_chunks.append(chunk)
                break

        # Collect until silence
        while silence_chunks < silence_limit_chunks:
            chunk = self._audio_q.get()
            audio_chunks.append(chunk)
            if np.sqrt(np.mean(chunk ** 2)) < ENERGY_THRESH:
                silence_chunks += 1
            else:
                silence_chunks = 0

        audio = np.concatenate(audio_chunks).flatten()
        segments, _ = self._whisper.transcribe(audio, language=self.lang, beam_size=1)
        text = ' '.join(s.text for s in segments).strip()

        if not text:
            return

        self.get_logger().info(f'Heard: {text}')

        text_lower = text.lower()
        wake_idx = -1
        wake_word = ''
        for w in WAKE_WORDS:
            idx = text_lower.find(w)
            if idx != -1:
                wake_idx, wake_word = idx, w
                break

        if wake_idx == -1:
            return  # no wake word ("Max") — ignore

        command_text = (text[:wake_idx] + text[wake_idx + len(wake_word):]).strip(' ,.')
        if not command_text:
            self.get_logger().info('Wake word "Max" detected, αναμονή εντολής...')
            return

        self._handle_command(command_text)

    def _handle_command(self, text: str):
        try:
            response = ollama.chat(
                model='qwen3-vl:4b-instruct',
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': text},
                ],
                options={'temperature': 0.1},
                keep_alive='10m',  # ξύπνα/κράτα φορτωμένο το LLM μετά το wake word
            )
            raw = response.message.content.strip()
            cmd = json.loads(raw)
        except Exception as e:
            self.get_logger().warn(f'Command parse failed: {e}')
            return

        self.get_logger().info(f'Command: {cmd}')
        self.cmd_pub.publish(String(data=json.dumps(cmd)))

        reply = cmd.get('reply')
        if reply:
            self.get_logger().info(f'Max: {reply}')
            self.response_pub.publish(String(data=reply))

        action = cmd.get('action', '')
        if action == 'dock':
            self.dock_pub.publish(Bool(data=True))
        elif action == 'stop':
            self.dock_pub.publish(Bool(data=False))

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = VoiceControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
