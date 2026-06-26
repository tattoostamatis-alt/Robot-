#!/usr/bin/env python3
"""
diarization_node.py — Real-time speaker diarization using resemblyzer d-vectors.

Runs VAD (webrtcvad) continuously on the microphone. Each detected speech
segment is embedded with resemblyzer's VoiceEncoder and compared against enrolled
speaker profiles using cosine similarity. Publishes who is speaking.

Subscribes:  diarization/register (std_msgs/String) — name to enroll as next speaker
Publishes:   current_speaker      (std_msgs/String) — matched name or "unknown"

Speaker profiles persist across restarts in ~/.robot_speakers.npz.
New voices are auto-enrolled as Speaker_1, Speaker_2, … when `auto_enroll` is true.
To enroll a specific name: publish the name to diarization/register, then speak.
"""

import os
import sys
import threading
import queue
from collections import deque

# VitisAI EP setup before onnxruntime import.
_VENV_SITE = '/home/dimi/ryzenai_venv/lib/python3.12/site-packages'
if os.path.isdir(_VENV_SITE):
    sys.path.insert(0, _VENV_SITE)
os.environ.setdefault('XILINX_XRT', '/opt/xilinx/xrt')
os.environ.setdefault('RYZEN_AI_INSTALLATION_PATH', '/home/dimi/ryzenai_venv')

import numpy as np
import onnxruntime as ort
import sounddevice as sd
from resemblyzer import VoiceEncoder, preprocess_wav

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SAMPLE_RATE    = 16000
FRAME_MS       = 30
FRAME_SAMPLES  = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples per frame

_VAIP_CONFIG   = '/home/dimi/ryzenai_venv/voe-4.0-linux_x86_64/vaip_config.json'
_SILERO_MODEL  = '/home/dimi/.local/lib/python3.12/site-packages/openwakeword/resources/models/silero_vad.onnx'


class SileroVAD:
    """Silero VAD ONNX session with persistent LSTM state, running on NPU."""
    def __init__(self):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        if os.path.isfile(_VAIP_CONFIG):
            self._sess = ort.InferenceSession(
                _SILERO_MODEL, sess_options=opts,
                providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
                provider_options=[{'config_file': _VAIP_CONFIG}, {}])
        else:
            self._sess = ort.InferenceSession(
                _SILERO_MODEL, sess_options=opts,
                providers=['CPUExecutionProvider'])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def is_speech(self, frame_int16: np.ndarray, threshold: float = 0.5) -> bool:
        x = frame_int16.astype(np.float32) / 32768.0
        x = x[np.newaxis, :]  # (1, N)
        out, hn, cn = self._sess.run(None, {
            'input': x, 'sr': self._sr, 'h': self._h, 'c': self._c
        })
        self._h, self._c = hn, cn
        return float(out[0, 0]) >= threshold
PROFILES_FILE  = os.path.expanduser('~/.robot_speakers.npz')
MIN_FRAMES_FOR_EMBED = 17                        # ~0.5s — shorter segments give bad embeddings


class DiarizationNode(Node):
    def __init__(self):
        super().__init__('diarization_node')

        self.declare_parameter('device_index',         -1)
        self.declare_parameter('mic_channels',          1)
        self.declare_parameter('mic_channel',           0)
        self.declare_parameter('vad_aggressiveness',    2)    # 0-3; higher = stricter
        self.declare_parameter('min_speech_frames',     5)    # frames before segment starts
        self.declare_parameter('silence_frames',       15)    # frames of silence to end segment
        self.declare_parameter('max_segment_frames',  200)    # ~6s max
        self.declare_parameter('similarity_threshold', 0.75)
        self.declare_parameter('auto_enroll',          True)  # label new voices automatically

        self._dev_idx   = self.get_parameter('device_index').value
        self._n_ch      = self.get_parameter('mic_channels').value
        self._ch        = self.get_parameter('mic_channel').value
        self._sim_thr   = self.get_parameter('similarity_threshold').value
        self._auto_enr  = self.get_parameter('auto_enroll').value
        self._min_sp    = self.get_parameter('min_speech_frames').value
        self._sil_lim   = self.get_parameter('silence_frames').value
        self._max_seg   = self.get_parameter('max_segment_frames').value

        self._vad = SileroVAD()
        self.get_logger().info(f'VAD: Silero on {self._vad._sess.get_providers()[0]}')

        self.get_logger().info('Loading VoiceEncoder…')
        self._encoder = VoiceEncoder()

        self._profiles: dict[str, np.ndarray] = {}
        self._profiles_lock = threading.Lock()
        self._load_profiles()

        self._pending_enroll: str | None = None   # set by register callback
        self._auto_count = len(self._profiles) + 1

        self._pub = self.create_publisher(String, 'current_speaker', 10)
        self.create_subscription(String, 'diarization/register', self._register_cb, 10)

        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()

        names = ', '.join(self._profiles) or '(none)'
        self.get_logger().info(
            f'Diarization ready — enrolled: {names} | '
            f'auto_enroll={self._auto_enr} | threshold={self._sim_thr}'
        )

    # ── Profile persistence ──────────────────────────────────────────

    def _load_profiles(self):
        if os.path.exists(PROFILES_FILE):
            data = np.load(PROFILES_FILE, allow_pickle=True)
            with self._profiles_lock:
                self._profiles = {k: data[k] for k in data.files}
            self.get_logger().info(f'Loaded {len(self._profiles)} speaker profile(s)')

    def _save_profiles(self):
        with self._profiles_lock:
            np.savez(PROFILES_FILE, **self._profiles)

    # ── ROS callbacks ────────────────────────────────────────────────

    def _register_cb(self, msg: String):
        name = msg.data.strip()
        if name:
            self._pending_enroll = name
            self.get_logger().info(f'Ready to enroll next speaker as "{name}" — speak now')

    # ── Audio capture ────────────────────────────────────────────────

    def _capture_loop(self):
        dev = None if self._dev_idx == -1 else self._dev_idx
        try:
            with sd.InputStream(
                device=dev, samplerate=SAMPLE_RATE,
                channels=self._n_ch, dtype='int16',
                blocksize=FRAME_SAMPLES,
            ) as stream:
                while self._running:
                    data, _ = stream.read(FRAME_SAMPLES)
                    mono = data[:, min(self._ch, self._n_ch - 1)]
                    try:
                        self._audio_q.put_nowait(mono.copy())
                    except queue.Full:
                        pass  # drop oldest indirectly; queue drains in process_loop
        except Exception as e:
            self.get_logger().error(f'Audio capture failed: {e}')

    # ── VAD + segment detection ──────────────────────────────────────

    def _process_loop(self):
        vad_ring   = deque(maxlen=self._min_sp)
        audio_ring = deque(maxlen=self._min_sp)
        segment: list[np.ndarray] = []
        in_speech  = False
        silence_ct = 0

        while self._running:
            try:
                frame = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                is_speech = self._vad.is_speech(frame)
            except Exception:
                continue

            vad_ring.append(is_speech)
            audio_ring.append(frame)

            if not in_speech:
                if (len(vad_ring) == self._min_sp
                        and sum(vad_ring) >= self._min_sp):
                    in_speech  = True
                    segment    = list(audio_ring)   # include pre-roll
                    silence_ct = 0
            else:
                segment.append(frame)
                silence_ct = silence_ct + 1 if not is_speech else 0

                if silence_ct >= self._sil_lim or len(segment) >= self._max_seg:
                    if len(segment) >= MIN_FRAMES_FOR_EMBED:
                        self._identify_segment(segment)
                    segment    = []
                    in_speech  = False
                    silence_ct = 0

    # ── Speaker identification ───────────────────────────────────────

    def _identify_segment(self, frames: list[np.ndarray]):
        audio = np.concatenate(frames).astype(np.float32) / 32768.
        try:
            wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
            emb = self._encoder.embed_utterance(wav)
        except Exception as e:
            self.get_logger().warn(f'Embedding failed: {e}')
            return

        # Enroll mode: next segment gets labelled with the pending name
        pending = self._pending_enroll
        if pending:
            self._pending_enroll = None
            with self._profiles_lock:
                self._profiles[pending] = emb
            self._save_profiles()
            self.get_logger().info(f'Enrolled "{pending}"')
            self._pub.publish(String(data=pending))
            return

        # Compare against existing profiles
        with self._profiles_lock:
            profiles_snapshot = dict(self._profiles)

        if not profiles_snapshot:
            if self._auto_enr:
                name = f'Speaker_{self._auto_count}'
                self._auto_count += 1
                with self._profiles_lock:
                    self._profiles[name] = emb
                self._save_profiles()
                self.get_logger().info(f'First speaker auto-enrolled as "{name}"')
                self._pub.publish(String(data=name))
            return

        best_name, best_sim = 'unknown', 0.
        for name, profile_emb in profiles_snapshot.items():
            sim = float(
                np.dot(emb, profile_emb)
                / (np.linalg.norm(emb) * np.linalg.norm(profile_emb) + 1e-6)
            )
            if sim > best_sim:
                best_sim, best_name = sim, name

        if best_sim < self._sim_thr:
            if self._auto_enr:
                best_name = f'Speaker_{self._auto_count}'
                self._auto_count += 1
                with self._profiles_lock:
                    self._profiles[best_name] = emb
                self._save_profiles()
                self.get_logger().info(
                    f'New voice auto-enrolled as "{best_name}" (best_sim={best_sim:.2f})'
                )
            else:
                best_name = 'unknown'
        else:
            self.get_logger().info(f'Speaker: {best_name} (sim={best_sim:.2f})')

        self._pub.publish(String(data=best_name))

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main():
    rclpy.init()
    node = DiarizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
