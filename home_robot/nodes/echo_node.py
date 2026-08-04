#!/usr/bin/env python3
"""Echolocation — the robot chirps and listens to the room answer.

The robot has a speaker and a microphone array, so it can make an ACTIVE
acoustic measurement: play a short chirp, record what comes back, and read the
room out of the reverberation — in the dark, with no camera, through a closed
door.

‼️ It listens on the RAW beam, NOT the echo-cancelled one. This was the
opposite of my first assumption: the XVF3800's AEC exists so the robot can hear
OTHERS while it speaks, and it does that by removing the robot's own output —
which here IS the signal. On the processed channel every measurement came back
with no measurable decay, because the canceller had already taken it away.

Topics
  in   /echo/probe     (String)        — room name (or empty) to measure now
       /mic/audio_raw  (Int16MultiArray) — the UN-cancelled 16 kHz beam
  out  /echo           (String, JSON)  — the last measurement and any change
       /speech_response (String)       — spoken, only when something CHANGED

Signatures persist in ~/.ros/home_robot_echo.json, keyed by room.

‼️ MANUAL and opt-in. The chirp is audible and would be alarming coming out of
a robot at three in the morning, so nothing here fires on a timer. It runs when
asked, and it refuses while the robot is speaking or while somebody is talking.

‼️ The trustworthy output is CHANGE, not absolutes. The speaker, the microphone
and the robot's own body all appear identically in two measurements from the
same spot and cancel out; they are all baked into a single number. So the node
compares against a stored signature for the room and reports the difference —
an absolute "this room is 4 m across" is offered, but flagged as approximate.
"""

import json
import os
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Int16MultiArray, String

from home_robot.echo import (
    SAMPLE_RATE, EchoSignature, chirp, compare, energy_envelope,
    first_reflection, rt60_estimate,
)

_STATE_PATH = os.path.expanduser('~/.ros/home_robot_echo.json')
# How long to keep recording after the chirp. Long enough for a normal room to
# decay 60 dB; a cathedral would need more, and this is a flat.
_LISTEN_S = 1.2


class EchoNode(Node):
    def __init__(self):
        super().__init__('echo_node')

        self.declare_parameter('device_index', 7)   # pulse, same as tts_node
        self.declare_parameter('chirp_seconds', 0.06)
        self.declare_parameter('amplitude', 0.5)
        self.declare_parameter('speak_changes', True)

        self.speak_changes = self.get_parameter('speak_changes').value
        self.signatures = self._load()

        self._buf = []
        self._capturing = False
        self._lock = threading.Lock()
        self._busy = False
        self._speaking = False
        self._last = None
        self._verdict = None

        # ‼️ The RAW channel, not /mic/audio. That one is echo-cancelled, and
        # the canceller removes the robot's own chirp — which is the entire
        # signal here. Needs wake_word_node started with publish_raw:=true.
        # Measured 2026-08-04: on the processed channel RT60 came back None
        # every time, because there was nothing left to decay.
        self.create_subscription(Int16MultiArray, '/mic/audio_raw',
                                 self._cb_audio, 50)
        self.create_subscription(Bool, '/tts/speaking', self._cb_speaking, 10)
        self.create_subscription(String, '/echo/probe', self._cb_probe, 5)
        self.create_subscription(String, '/current_room', self._cb_room, 10)
        self._room = None

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(String, 'echo', latched)
        self._say_pub = self.create_publisher(String, '/speech_response', 10)

        self._publish()
        self.get_logger().info(
            f'Echo ready — {len(self.signatures)} rooms measured. '
            f'Publish a room name to /echo/probe to measure.')

    # ── state ─────────────────────────────────────────────────────────────────

    def _load(self):
        try:
            with open(_STATE_PATH, encoding='utf-8') as fh:
                raw = json.load(fh) or {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return {k: EchoSignature.from_dict(v) for k, v in raw.items()}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
            tmp = _STATE_PATH + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump({k: v.to_dict() for k, v in self.signatures.items()},
                          fh)
            os.replace(tmp, _STATE_PATH)
        except OSError as exc:
            self.get_logger().warn(f'could not save echo signatures: {exc}')

    # ── inputs ────────────────────────────────────────────────────────────────

    def _cb_room(self, msg: String):
        self._room = (msg.data or '').strip() or None

    def _cb_speaking(self, msg: Bool):
        self._speaking = bool(msg.data)

    def _cb_audio(self, msg: Int16MultiArray):
        with self._lock:
            if self._capturing:
                self._buf.extend(msg.data)

    def _cb_probe(self, msg: String):
        if self._busy:
            self.get_logger().warn('already measuring')
            return
        if self._speaking:
            self.get_logger().warn('not while the robot is speaking')
            return
        room = (msg.data or '').strip() or self._room or 'unknown'
        threading.Thread(target=self._measure, args=(room,),
                         daemon=True).start()

    # ── the measurement ───────────────────────────────────────────────────────

    def _measure(self, room):
        self._busy = True
        try:
            import sounddevice as sd
        except ImportError as exc:
            self.get_logger().error(f'sounddevice missing: {exc}')
            self._busy = False
            return

        sig = chirp(seconds=self.get_parameter('chirp_seconds').value,
                    amplitude=self.get_parameter('amplitude').value)
        with self._lock:
            self._buf = []
            self._capturing = True
        try:
            sd.play(sig, samplerate=SAMPLE_RATE,
                    device=self.get_parameter('device_index').value)
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().error(f'could not play the chirp: {exc}')
            with self._lock:
                self._capturing = False
            self._busy = False
            return

        time.sleep(_LISTEN_S)
        with self._lock:
            self._capturing = False
            samples = np.array(self._buf, dtype=np.float32) / 32768.0

        if samples.size < SAMPLE_RATE // 4:
            self.get_logger().warn(
                'no raw microphone audio — start wake_word_node with '
                'publish_raw:=true (mic/audio is echo-cancelled and useless here)')
            self._busy = False
            return

        env = energy_envelope(samples)
        now = EchoSignature(rt60_estimate(env), first_reflection(env), env, room)
        before = self.signatures.get(room)
        changed, why = compare(now, before)

        self._last = now
        self._verdict = {'room': room, 'changed': changed, 'why': why}
        # ‼️ Store only when NOT changed, or the first surprise becomes the new
        # normal and the change can never be reported twice. A changed room has
        # to be re-baselined deliberately.
        if before is None or not changed:
            self.signatures[room] = now
            self._save()

        rt = f'{now.rt60:.2f}s' if now.rt60 else '—'
        dist = f'{now.distance:.1f}m' if now.distance else '—'
        self.get_logger().info(f'{room}: RT60 {rt}, first reflection {dist} — {why}')
        if changed and self.speak_changes:
            self._say_pub.publish(String(data=f'Κάτι άλλαξε εδώ: {why}.'))
        self._publish()
        self._busy = False

    def _publish(self):
        last = self._last
        self._pub.publish(String(data=json.dumps({
            'rooms': sorted(self.signatures),
            'busy': self._busy,
            'last': ({
                'room': last.room,
                'rt60': round(last.rt60, 2) if last.rt60 else None,
                'distance': round(last.distance, 2) if last.distance else None,
                'envelope': [round(v, 1) for v in last.envelope[:120]],
            } if last else None),
            'verdict': self._verdict,
        }, ensure_ascii=False)))


def main():
    rclpy.init()
    node = EchoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
