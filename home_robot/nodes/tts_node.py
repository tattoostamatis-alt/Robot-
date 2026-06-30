#!/usr/bin/env python3
"""Streaming text-to-speech — edge-tts, triggered by speech_response.

Subscribes to `speech_response` (std_msgs/String, from llm_bridge_node) and
speaks each message with an edge-tts neural voice (default Greek, "Max"
persona). Messages are queued and played back sequentially on a background
thread so the ROS callback never blocks.

edge-tts returns MP3 audio; ffmpeg (already used for wake-word training data
in this project) decodes it to PCM via a subprocess pipe before playback
with sounddevice.
"""

import asyncio
import queue
import subprocess
import threading

import edge_tts
import numpy as np
import rclpy
import sounddevice as sd
from rclpy.node import Node
from std_msgs.msg import String

SAMPLE_RATE = 24000  # edge-tts default output rate


class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')

        self.declare_parameter('voice', 'el-GR-NestorasNeural')
        self.declare_parameter('rate', '+0%')
        self.declare_parameter('volume', '+0%')
        self.declare_parameter('device_index', 7)  # pulse — works with any PulseAudio output

        self.voice = self.get_parameter('voice').value
        self.rate = self.get_parameter('rate').value
        self.volume = self.get_parameter('volume').value
        self.device_index = self.get_parameter('device_index').value

        self._queue = queue.Queue()
        threading.Thread(target=self._playback_loop, daemon=True).start()

        self.create_subscription(String, 'speech_response', self._on_speech_response, 10)

        self.get_logger().info(f'TTS node started — voice={self.voice}')

    def _on_speech_response(self, msg: String):
        text = msg.data.strip()
        if text:
            self._queue.put(text)

    def _playback_loop(self):
        while True:
            text = self._queue.get()
            try:
                self._synthesize_and_play(text)
            except Exception as e:
                self.get_logger().error(f'TTS failed: {e}')

    def _synthesize_and_play(self, text):
        mp3 = asyncio.run(self._synthesize(text))
        pcm = self._decode_mp3(mp3)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        device = None if self.device_index < 0 else self.device_index
        self.get_logger().info(f'Speaking: {text}')
        sd.play(audio, samplerate=SAMPLE_RATE, device=device)
        sd.wait()

    async def _synthesize(self, text):
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, volume=self.volume)
        chunks = []
        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                chunks.append(chunk['data'])
        return b''.join(chunks)

    def _decode_mp3(self, mp3_bytes):
        proc = subprocess.run(
            ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1', 'pipe:1'],
            input=mp3_bytes, capture_output=True, check=True)
        return proc.stdout

    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
