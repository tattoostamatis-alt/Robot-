#!/usr/bin/env python3
"""Streaming text-to-speech — edge-tts, triggered by speech_response.

Subscribes to `speech_response` (std_msgs/String, from llm_bridge_node) and
speaks each message with an edge-tts neural voice (default Greek female,
"Max" persona). Messages are queued and played back sequentially on a background
thread so the ROS callback never blocks.

edge-tts returns MP3 audio; ffmpeg (already used for wake-word training data
in this project) decodes it to PCM via a subprocess pipe before playback
with sounddevice.
"""

import asyncio
import queue
import subprocess
import threading
import time

import edge_tts
import numpy as np
import rclpy
import sounddevice as sd
from rclpy.node import Node
from std_msgs.msg import Bool, String

from home_robot.voice_gate import TOPIC as SPEAKING_TOPIC, STOP_TOPIC

SAMPLE_RATE = 24000  # edge-tts default output rate


class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')

        # el-GR-AthinaNeural (female) by user preference 2026-07-25; the male
        # el-GR-NestorasNeural is the only other Greek edge-tts voice.
        self.declare_parameter('voice', 'el-GR-AthinaNeural')
        self.declare_parameter('rate', '+0%')
        self.declare_parameter('volume', '+0%')
        self.declare_parameter('device_index', 7)  # pulse — works with any PulseAudio output
        # Hold `tts/speaking` True this long after playback ends so the listeners
        # ride out the room-reverb tail before re-arming (see voice_gate.py).
        self.declare_parameter('speaking_tail', 0.3)

        self.voice = self.get_parameter('voice').value
        self.rate = self.get_parameter('rate').value
        self.volume = self.get_parameter('volume').value
        self.device_index = self.get_parameter('device_index').value
        self.speaking_tail = self.get_parameter('speaking_tail').value

        # State is re-published on every transition (and True again at the start
        # of each utterance); the listeners co-start at bringup so plain volatile
        # QoS is enough — no late-joiner catch-up needed.
        self.speaking_pub = self.create_publisher(Bool, SPEAKING_TOPIC, 10)
        self._speaking = False
        self._set_speaking(False)

        self._queue = queue.Queue()
        self._interrupted = False
        threading.Thread(target=self._playback_loop, daemon=True).start()

        self.create_subscription(String, 'speech_response', self._on_speech_response, 10)
        # Barge-in: abort the current utterance and drop anything queued.
        self.create_subscription(Bool, STOP_TOPIC, self._on_stop, 10)

        self.get_logger().info(f'TTS node started — voice={self.voice}')

    def _set_speaking(self, speaking: bool):
        # Publish on every call so late joiners get the current state, but only
        # log on an actual transition.
        if speaking != self._speaking:
            self.get_logger().debug(f'tts/speaking → {speaking}')
        self._speaking = speaking
        self.speaking_pub.publish(Bool(data=speaking))

    def _on_speech_response(self, msg: String):
        text = msg.data.strip()
        if text:
            self._queue.put(text)

    def _on_stop(self, msg: Bool):
        """Barge-in: abort the current utterance and drop anything queued."""
        if not msg.data:
            return
        self._interrupted = True
        # Only touch the device when something is actually playing. Calling
        # sd.stop() against an idle/paused stream is what used to race sd.play()
        # and leave PortAudio wedged (see _synthesize_and_play).
        if self._speaking:
            try:
                sd.stop()
            except Exception:
                pass
        dropped = 0
        try:
            while True:
                self._queue.get_nowait()
                dropped += 1
        except queue.Empty:
            pass
        self.get_logger().info(f'Barge-in: TTS interrupted (dropped {dropped} queued)')

    def _playback_loop(self):
        while True:
            text = self._queue.get()
            self._interrupted = False   # fresh utterance — clear any prior barge-in
            self._set_speaking(True)
            try:
                self._synthesize_and_play(text)
            except Exception as e:
                self.get_logger().error(f'TTS failed: {e}')
            finally:
                # Only re-arm the mic once the queue has drained, so a burst of
                # responses doesn't toggle the gate open in the gaps between them.
                if self._queue.empty():
                    time.sleep(self.speaking_tail)
                    if self._queue.empty():
                        self._set_speaking(False)

    def _synthesize_and_play(self, text):
        mp3 = asyncio.run(self._synthesize(text))
        pcm = self._decode_mp3(mp3)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        if self._interrupted:   # barge-in landed during synthesis — don't start playing
            return
        device = None if self.device_index < 0 else self.device_index
        self.get_logger().info(f'Speaking: {text}')
        sd.play(audio, samplerate=SAMPLE_RATE, device=device)

        # Bounded wait instead of sd.wait(). sd.wait() blocks forever, and a
        # sd.stop() arriving from the barge-in callback while sd.play() is still
        # starting up could leave it hung: the playback thread then never
        # returned, so the robot went permanently silent AND never published
        # tts/speaking=False — which also keeps the STT gate closed, so it
        # stopped hearing commands too. Both symptoms, no error logged, node
        # still alive. Seen for real on 2026-07-23 after four rapid barge-ins.
        # Polling to a deadline derived from the audio length cannot hang.
        deadline = time.monotonic() + len(audio) / SAMPLE_RATE + 2.0
        while time.monotonic() < deadline and not self._interrupted:
            time.sleep(0.05)
        try:
            sd.stop()
        except Exception:
            pass

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
