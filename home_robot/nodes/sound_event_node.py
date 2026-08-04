#!/usr/bin/env python3
"""Sound events — the robot notices what it HEARS, not just what is said to it.

YAMNet (521 AudioSet classes) over the microphone, filtered down to the dozen
household sounds worth reacting to: doorbell, breaking glass, smoke alarm, a
baby crying, running water, something falling. Combined with the XVF3800's
direction-of-arrival, the robot knows what it heard AND roughly where from.

Topics
  in   /mic/audio        (Int16MultiArray) — 16 kHz mono, from wake_word_node
       /doa/angle        (Float32)         — 0-359°, XVF3800
       /tts/speaking     (Bool)            — mute while the robot talks
  out  /sound_events     (String, JSON)    — feed for the dashboard/timeline
       /speech_response  (String)          — spoken, for important events only

‼️ It does NOT open the microphone. wake_word_node owns the ALSA stream and
republishes it on /mic/audio precisely so a second stream is never opened on the
same device — stt_node consumes it the same way. Opening the device here would
fight the wake word for it.

‼️ onnxruntime threads are capped. The default is every core, which on this
16-core box measured SLOWER than two threads and pinned 9.5 cores for the face
detector. YAMNet is ~5 ms per window at 2 threads; it must stay that way, since
this runs continuously.

The model (~15 MB ONNX) is pulled from HuggingFace on first run and cached, the
same way the faster-whisper models are. No weights are committed to the repo.
"""

import json
import threading
import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int16MultiArray, String

from home_robot.sound_events import (
    EVENTS, SoundEventGate, bearing_el, classify, describe,
)

_HF_REPO = 'zeropointnine/yamnet-onnx'
_SAMPLE_RATE = 16000
# YAMNet needs at least 0.96 s to produce a single frame; 1.5 s gives a few
# frames per window so a short transient is not clipped at a boundary.
_WINDOW_S = 1.5
_HOP_S = 0.5


class SoundEventNode(Node):
    def __init__(self):
        super().__init__('sound_event_node')

        self.declare_parameter('enabled', True)
        self.declare_parameter('threshold', 0.35)
        self.declare_parameter('speak', True)
        self.declare_parameter('speak_min_importance', 0.8)
        self.declare_parameter('cooldown', 20.0)
        # A critical event (glass, alarm, baby) skips the two-window
        # corroboration only above this score. Live run 2026-08-04: a single
        # 0.53 window of room noise announced a crying baby.
        self.declare_parameter('instant_min_score', 0.75)
        self.declare_parameter('doa_max_age', 3.0)      # s
        self.declare_parameter('num_threads', 2)

        self.enabled    = self.get_parameter('enabled').value
        self.threshold  = self.get_parameter('threshold').value
        self.speak      = self.get_parameter('speak').value
        self.speak_min  = self.get_parameter('speak_min_importance').value
        self.doa_max_age = self.get_parameter('doa_max_age').value

        self._session = None
        self._class_names = []
        self._buf = deque(maxlen=int(_SAMPLE_RATE * _WINDOW_S))
        self._lock = threading.Lock()
        self._angle = None
        self._angle_at = 0.0
        self._speaking = False
        self._feed = []
        self._last_speech_score = 0.0
        self._windows = 0

        self.gate = SoundEventGate(
            cooldown=self.get_parameter('cooldown').value,
            instant_min_score=self.get_parameter('instant_min_score').value)

        threading.Thread(target=self._load_model, daemon=True).start()

        self.create_subscription(Int16MultiArray, '/mic/audio', self._cb_audio, 50)
        self.create_subscription(Float32, '/doa/angle', self._cb_doa, 10)
        self.create_subscription(Bool, '/tts/speaking', self._cb_speaking, 10)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._events_pub = self.create_publisher(String, 'sound_events', latched)
        self._say_pub    = self.create_publisher(String, '/speech_response', 10)

        self.create_timer(_HOP_S, self._tick)
        self.get_logger().info('Sound event node starting — loading YAMNet…')

    # ── model ─────────────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            import csv
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            self.get_logger().error(f'onnxruntime/huggingface_hub missing: {exc}')
            return
        try:
            model_path = hf_hub_download(_HF_REPO, 'yamnet.onnx')
            map_path = hf_hub_download(_HF_REPO, 'yamnet_class_map.csv')
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().error(
                f'could not fetch YAMNet ({exc}) — sound events disabled. '
                f'It downloads once and caches; check the network.')
            return

        with open(map_path, encoding='utf-8') as fh:
            self._class_names = [row[2] for row in list(csv.reader(fh))[1:]]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.get_parameter('num_threads').value
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_path, opts, providers=['CPUExecutionProvider'])
        self.get_logger().info(
            f'YAMNet ready — {len(self._class_names)} classes, listening for '
            f'{len(EVENTS)} household events')

    # ── inputs ────────────────────────────────────────────────────────────────

    def _cb_audio(self, msg: Int16MultiArray):
        with self._lock:
            self._buf.extend(msg.data)

    def _cb_doa(self, msg: Float32):
        self._angle = float(msg.data)
        self._angle_at = time.time()

    def _cb_speaking(self, msg: Bool):
        self._speaking = bool(msg.data)

    def _current_angle(self):
        """The DoA angle, if it is recent enough to belong to this sound."""
        if self._angle is None:
            return None
        if (time.time() - self._angle_at) > self.doa_max_age:
            return None
        return self._angle

    # ── the loop ──────────────────────────────────────────────────────────────

    def _tick(self):
        if not self.enabled or self._session is None:
            return
        # Muted while the robot talks: its own TTS comes straight back through
        # the mic, and classifying our own output is noise at best.
        if self._speaking:
            return

        with self._lock:
            if len(self._buf) < int(_SAMPLE_RATE * 0.96):
                return
            samples = np.array(self._buf, dtype=np.float32)

        # int16 -> [-1, 1], which is what YAMNet's frontend expects.
        waveform = samples / 32768.0
        try:
            scores = self._session.run(None, {'waveform': waveform})[0]
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().error(f'YAMNet inference failed: {exc}',
                                    throttle_duration_sec=30.0)
            return

        self._windows += 1
        events, speech = classify(scores, self._class_names,
                                  threshold=self.threshold)
        self._last_speech_score = speech

        fired = self.gate.update(events)
        if fired:
            self._on_events(fired, dict(events))
        self._publish(events, fired)

    def _on_events(self, fired, scores):
        angle = self._current_angle()
        now = time.time()
        for key in fired:
            greek, importance, _ = EVENTS.get(key, (key, 0.0, ()))
            text = describe(key, angle)
            self._feed.append({
                'event': key, 'text': text, 'greek': greek,
                'score': scores.get(key), 'importance': importance,
                'angle': angle, 'bearing': bearing_el(angle), 'ts': now,
            })
            del self._feed[:-30]
            self.get_logger().info(f'Heard: {text} (score {scores.get(key)})')
            if self.speak and importance >= self.speak_min:
                self._say_pub.publish(String(data=text))

    def _publish(self, events, fired):
        self._events_pub.publish(String(data=json.dumps({
            'listening': self._session is not None and not self._speaking,
            'windows': self._windows,
            'speech': self._last_speech_score,
            'angle': self._current_angle(),
            'bearing': bearing_el(self._current_angle()),
            # Everything above threshold this window, so the UI can show what it
            # is nearly hearing, not only what cleared the gate.
            'candidates': [{'event': k, 'score': s,
                            'greek': EVENTS.get(k, (k,))[0]} for k, s in events],
            'fired': fired,
            'feed': self._feed[-10:],
        }, ensure_ascii=False)))


def main():
    rclpy.init()
    node = SoundEventNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
