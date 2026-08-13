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
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool, String, Int16MultiArray

from home_robot.stt_postprocess import clean
from home_robot.voice_gate import SpeakingGate, TOPIC as SPEAKING_TOPIC


SAMPLE_RATE = 16000
CHUNK_SIZE  = 1280   # must match wake_word_node CHUNK_SIZE (80ms @ 16kHz)


class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')

        # 2026-07-24: large-v3 → medium for latency. Benchmarked on a 4.2 s Greek
        # clip (int8, 8 threads): large-v3 beam5 11.7 s vs medium beam5 4.6 s
        # (~2.5x faster) with identical transcription. large-v3 stays available
        # via `model_size:=large-v3` if a hard Greek clip needs the accuracy.
        self.declare_parameter('model_size',        'medium')
        # 8 threads is the sweet spot on this 16-core CPU (~25% faster than the
        # ctranslate2 default of ~4; 12+ regresses from oversubscription). YOLO
        # moved off the CPU to the iGPU, so these cores are free during STT.
        self.declare_parameter('cpu_threads',       8)
        self.declare_parameter('language',          'el')
        self.declare_parameter('energy_thresh',     0.065)
        self.declare_parameter('start_timeout',     5.0)
        self.declare_parameter('silence_limit',     1.5)
        self.declare_parameter('max_record_seconds', 12.0)
        self.declare_parameter('preroll_seconds',    0.5)
        self.declare_parameter('wakeword_flush_ms', 300)
        self.declare_parameter('calibrate_on_start', True)
        # Diagnosis aid: when set to a directory, every buffer sent to Whisper
        # is also written there as a wav, so a bad transcription can be
        # replayed offline instead of guessed at. Empty = disabled (default);
        # this records the room, so it stays off unless explicitly requested.
        self.declare_parameter('debug_audio_dir', '')
        # Don't treat the robot's own TTS as user speech (barge-in / self-echo).
        self.declare_parameter('suppress_on_tts', True)
        self.declare_parameter('tts_release_tail', 0.3)
        # ── hardware VAD (XVF3800, via doa_node's /voice_activity) ──────────
        # The energy threshold alone cannot tell a voice from a fan spike: both
        # are "loud". The DSP's speech flag can, so it is used as a SECOND
        # opinion for starting a recording — never as a replacement.
        #
        # It is deliberately one-directional:
        #   * to START recording, energy AND the VAD must agree (fewer false
        #     starts from fans, doors, the robot's own motors);
        #   * to KEEP recording, the VAD can only extend, never cut. A VAD that
        #     says "no speech" mid-sentence must not truncate the user.
        #
        # ‼️ And it fails OPEN. If /voice_activity has never arrived, or has
        # gone stale (doa_node crashed, ReSpeaker unplugged, use_doa:=false),
        # the gate is dropped entirely and behaviour returns to energy-only.
        # A stale `False` latched into this gate would make the robot silently
        # deaf, which is the single worst failure this pipeline has — see the
        # busy-lock bug that did exactly that.
        self.declare_parameter('use_hw_vad', True)
        # ‼️ Liveness is "does a publisher exist", NOT "did a message arrive
        # recently". /voice_activity is LATCHED STATE published only on
        # transitions (doa_node says so in as many words), so a quiet room
        # produces nothing for minutes on end. With the old 3 s arrival
        # timeout, measured 2026-08-04, the gate declared the VAD stale three
        # times in one evening and warned "Is doa_node still running?" about a
        # node that was running perfectly — and, far worse, silently fell back
        # to energy-only within 3 s of every transition. The DSP gate that was
        # added to stop a fan or a door from starting a recording was therefore
        # switched off almost all of the time.
        self.declare_parameter('vad_liveness_period', 2.0)
        # A True that never drops is the other way this breaks: the DSP flag
        # sticks and every noise passes the gate. Longer than any real
        # utterance, short enough to recover the same session.
        self.declare_parameter('vad_stuck_s', 120.0)
        # The flag is edge-triggered at 10 Hz, and the DSP drops it between
        # words. Hold it open briefly so a normal pause mid-utterance does not
        # re-close the start gate.
        self.declare_parameter('vad_hold_s', 0.8)

        model_size            = self.get_parameter('model_size').value
        self.lang             = self.get_parameter('language').value
        self.energy_thresh    = self.get_parameter('energy_thresh').value
        self.calibrate_on_start = self.get_parameter('calibrate_on_start').value
        self.debug_audio_dir  = self.get_parameter('debug_audio_dir').value
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

        self.use_hw_vad  = self.get_parameter('use_hw_vad').value
        self.vad_liveness_period = self.get_parameter('vad_liveness_period').value
        self.vad_stuck_s = self.get_parameter('vad_stuck_s').value
        self.vad_hold_s  = self.get_parameter('vad_hold_s').value
        self._vad_on     = False
        self._vad_at     = 0.0      # last message (any value)
        self._vad_true_at = 0.0     # last time it said "speech"
        self._vad_seen   = False    # has the topic EVER produced a message?
        self._vad_logged = False    # one-shot "gate active" log
        self._vad_alive  = True     # cached count_publishers, see _vad_alive_now
        self._vad_checked_at = 0.0

        self._preroll    = collections.deque(maxlen=preroll_chunks)
        self._state      = 'idle'   # idle | flushing | waiting_speech | recording
        self._record_buf = []
        self._wait_count = 0
        self._sil_count  = 0
        self._flush_rem  = 0
        self._lock       = threading.Lock()
        self._busy       = threading.Lock()
        # ‼️ This whole state machine is driven by INCOMING AUDIO and nothing
        # else — every timeout it has (start_timeout, silence_limit,
        # max_record_seconds) is counted in chunks, so it only advances while
        # chunks arrive. If mic/audio goes quiet mid-turn (the array drops off
        # the USB hub, wake_word_node's stream dies, the publisher is
        # restarted), the node stays in 'flushing'/'waiting_speech'/'recording'
        # holding _busy, and every later wake word is refused with "Already
        # transcribing" — permanently, with no error. The wall-clock watchdog
        # below is the only thing that can unstick it.
        self._last_audio = time.monotonic()
        self.declare_parameter('audio_timeout', 5.0)   # s of silence on mic/audio
        self._audio_timeout = self.get_parameter('audio_timeout').value

        # ‼️ The watchdog above covers exactly one shape of stuck: mic/audio
        # stopped WHILE a turn was in progress. It cannot cover the shape seen
        # live on 2026-08-01 — 7 wake words in 25 s, all refused with "Already
        # transcribing", zero transcriptions — because `_state` is set to
        # 'idle' BEFORE the transcribe thread starts. So a Whisper call that
        # hangs (or merely crawls: this happened at load 25 with perception on)
        # leaves _busy held while the state machine believes the turn is over
        # and audio keeps flowing. Both of that watchdog's guards then return
        # early, forever, and the robot is deaf with nothing in the log but the
        # refusals.
        #
        # So _busy gets its own wall-clock deadline, independent of state and
        # of audio. Transcription measures 2.0-4.2 s; a minute is not a slow
        # transcription, it is a hung one.
        self.declare_parameter('busy_timeout', 60.0)
        self._busy_timeout = self.get_parameter('busy_timeout').value
        self._busy_since = 0.0
        # Turn id, bumped on every acquire and every release. A thread may only
        # release the turn it owns: threading.Lock has no owner check, so after
        # the watchdog force-releases a hung turn, the hung thread finishing
        # later would otherwise release a LATER turn's lock and hand the node a
        # second stuck state that looks exactly like the first.
        self._busy_gen = 0
        self._busy_guard = threading.Lock()

        self.suppress_on_tts = self.get_parameter('suppress_on_tts').value
        self._gate = SpeakingGate(
            release_tail=self.get_parameter('tts_release_tail').value)

        cpu_threads = int(self.get_parameter('cpu_threads').value)

        self._whisper = None
        threading.Thread(target=self._load_whisper, args=(model_size, cpu_threads),
                         daemon=True).start()

        self.add_on_set_parameters_callback(self._on_param_change)
        self.create_timer(1.0, self._audio_watchdog)
        self.create_timer(2.0, self._busy_watchdog)

        self.text_pub = self.create_publisher(String, 'speech_text', 10)
        self.create_subscription(String,         'wake_word', self._on_wake_word, 10)
        self.create_subscription(Int16MultiArray, 'mic/audio', self._on_audio,    200)
        self.create_subscription(Bool,           SPEAKING_TOPIC, self._on_tts_speaking, 10)
        # transient_local to match doa_node's publisher: it is latched state,
        # published only on transitions. With plain volatile QoS the
        # subscription would not even connect (incompatible durability is a
        # silent no-match in DDS), and the gate would sit disabled for ever
        # while looking wired up.
        self.create_subscription(
            Bool, 'voice_activity', self._on_voice_activity,
            QoSProfile(depth=1,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=QoSReliabilityPolicy.RELIABLE))

        self.get_logger().info('STT node started — waiting for wake_word')

    def _on_param_change(self, params):
        for p in params:
            if p.name == 'energy_thresh':
                self.energy_thresh = p.value
                self.get_logger().info(f'energy_thresh updated to {p.value:.4f}')
            elif p.name == 'use_hw_vad':
                self.use_hw_vad = p.value
                self.get_logger().info(f'use_hw_vad updated to {p.value}')
        return SetParametersResult(successful=True)

    def _load_whisper(self, model_size, cpu_threads):
        from faster_whisper import WhisperModel
        self._whisper = WhisperModel(model_size, device='cpu', compute_type='int8',
                                     cpu_threads=cpu_threads)
        self.get_logger().info(
            f'Whisper model "{model_size}" loaded (cpu_threads={cpu_threads})')
        if self.calibrate_on_start:
            self._calibrate_energy_thresh()

    def _calibrate_energy_thresh(self):
        # Measure ambient RMS from /mic/audio over ~2s and put the threshold
        # between ambient and speech.
        import time
        chunks = []
        deadline = time.monotonic() + 2.5

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
            # The floor used to be 0.05, which on this mic sits ABOVE normal
            # speech: measured on the XVF3800 ch4 beam 2026-07-23, ambient RMS
            # is ~0.011 and speech peaks at ~0.051. max() therefore pinned the
            # threshold at 0.0500 on every boot — calibration was a no-op and
            # the node logged "No speech detected after wake word" forever
            # while the wake word itself fired perfectly. 0.018 (~1.6x ambient)
            # is the value that actually worked live.
            new_thresh = max(0.012, ambient_rms * 1.6)
            self.energy_thresh = new_thresh
            self.get_logger().info(
                f'Ambient noise RMS={ambient_rms:.4f} → energy_thresh={new_thresh:.4f}')
        else:
            self.get_logger().warn('Calibration got no audio from /mic/audio, keeping default')

    def _begin_turn(self):
        """Claim _busy for a new turn. Returns its id, or None if busy."""
        if not self._busy.acquire(blocking=False):
            return None
        with self._busy_guard:
            self._busy_gen += 1
            self._busy_since = time.monotonic()
            return self._busy_gen

    def _end_turn(self, gen) -> bool:
        """Release _busy, but only if `gen` still owns it.

        Guarded by its own lock rather than self._lock, so callers already
        holding the state lock cannot deadlock on it.
        """
        with self._busy_guard:
            if gen != self._busy_gen or not self._busy.locked():
                return False          # a stale owner, or already released
            self._busy_gen += 1
            try:
                self._busy.release()
            except RuntimeError:
                pass
            return True

    def _busy_watchdog(self):
        """Unstick a turn whose transcription never came back.

        Deliberately checks neither _state nor _last_audio: this is the case
        where both of those look perfectly healthy. See the note in __init__.
        """
        with self._busy_guard:
            if not self._busy.locked() or not self._busy_since:
                return
            held = time.monotonic() - self._busy_since
            gen = self._busy_gen
        if held < self._busy_timeout:
            return
        if self._end_turn(gen):
            self.get_logger().error(
                f'Transcription has not returned after {held:.0f}s — releasing '
                'the STT turn. Every wake word since was being refused with '
                '"Already transcribing".')

    def _audio_watchdog(self):
        """Unstick a turn that stalled because mic/audio stopped arriving.

        Only acts on a turn that is actually in progress: an idle node with no
        microphone is merely quiet, which is not a fault this can fix.
        """
        if time.monotonic() - self._last_audio < self._audio_timeout:
            return
        with self._lock:
            if self._state == 'idle':
                return
            self.get_logger().error(
                f'No mic/audio for {self._audio_timeout:.0f}s mid-turn — '
                'resetting STT (was stuck in "%s")' % self._state)
            self._state = 'idle'
            self._record_buf = []
            self._wait_count = 0
            self._sil_count = 0
        self._end_turn(self._busy_gen)

    def _on_tts_speaking(self, msg: Bool):
        self._gate.set_speaking(msg.data)

    def _on_wake_word(self, msg: String):
        if self._whisper is None:
            self.get_logger().warn('Whisper not loaded yet, ignoring wake word')
            return
        if self._begin_turn() is None:
            # INFO, not WARN: a second wake word inside the listening window is
            # ordinary — the person repeated themselves, or the room is noisy.
            # As a WARN it read like a fault and self_diagnosis treated it as
            # one, announcing "το ρομπότ είναι ΚΟΥΦΟ" all evening while every
            # transcription was going through. The genuinely stuck case has its
            # own line, from _busy_watchdog, and that is what is diagnosed.
            self.get_logger().info('Already transcribing, ignoring wake word')
            return
        with self._lock:
            self._preroll.clear()
            self._state      = 'flushing'
            self._flush_rem  = self._flush_chunks
            self._record_buf = []
            self._wait_count = 0
            self._sil_count  = 0
        self.get_logger().info('Wake word received — listening for speech')

    def _on_voice_activity(self, msg: Bool):
        now = time.monotonic()
        self._vad_at = now
        self._vad_on = bool(msg.data)
        if msg.data:
            self._vad_true_at = now
        if not self._vad_seen:
            self._vad_seen = True
            self.get_logger().info(
                'Hardware VAD connected — speech onset now needs the DSP to '
                'agree, not just loudness')

    def _vad_alive_now(self) -> bool:
        """Is anything still publishing /voice_activity?

        The graph query is the honest test — silence on a latched state topic
        means "nothing changed", not "the publisher died". Cached, because this
        is called for every audio chunk (~12 Hz) and a graph lookup is not
        free.
        """
        now = time.monotonic()
        if now - self._vad_checked_at >= self.vad_liveness_period:
            self._vad_checked_at = now
            try:
                self._vad_alive = self.count_publishers('voice_activity') > 0
            except Exception:
                self._vad_alive = True       # cannot tell: do not go deaf
        return self._vad_alive

    def _vad_trustworthy(self) -> bool:
        """Whether the DSP flag may gate anything at all right now."""
        if not self.use_hw_vad or not self._vad_seen:
            return False                     # never arrived: not our gate
        if not self._vad_alive_now():
            # doa_node is gone. Say so once, then get out of the way — a
            # latched False left in this gate would silently swallow every
            # command from now on.
            if not self._vad_logged:
                self._vad_logged = True
                self.get_logger().warn(
                    'Nothing is publishing /voice_activity any more — falling '
                    'back to the energy threshold. Did doa_node die?')
            return False
        if self._vad_on and time.monotonic() - self._vad_true_at > self.vad_stuck_s:
            if not self._vad_logged:
                self._vad_logged = True
                self.get_logger().warn(
                    f'The hardware VAD has said "speech" for over '
                    f'{self.vad_stuck_s:.0f}s — treating it as stuck and '
                    'falling back to the energy threshold.')
            return False
        self._vad_logged = False
        return True

    def _vad_gate_open(self) -> bool:
        """May a recording START right now, as far as the hardware VAD knows?

        Returns True whenever the VAD cannot be trusted, so a missing, disabled
        or crashed doa_node leaves the old energy-only behaviour intact instead
        of making the robot deaf.
        """
        if not self._vad_trustworthy():
            return True
        # Held briefly after the flag drops: the DSP releases it between words.
        return (self._vad_on
                or (time.monotonic() - self._vad_true_at) < self.vad_hold_s)

    def _on_audio(self, msg: Int16MultiArray):
        chunk = np.array(msg.data, dtype=np.int16).astype(np.float32) / 32768.0
        self._last_audio = time.monotonic()

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
            # While TTS is speaking, treat the mic as silent so the robot's own
            # voice can't be mistaken for the user's command (barge-in gate).
            if self.suppress_on_tts and self._gate.suppressed():
                energy = 0.0

            if self._state == 'waiting_speech':
                self._wait_count += 1
                # Loud AND actually a voice. The second half is what a fan
                # spike, a door or the base's own motors fail.
                if energy > self.energy_thresh and self._vad_gate_open():
                    self._state      = 'recording'
                    self._record_buf = list(self._preroll) + [chunk]
                    self._sil_count  = 0
                elif self._wait_count >= self._start_timeout_chunks:
                    self.get_logger().info('No speech detected after wake word')
                    self._state = 'idle'
                    self._end_turn(self._busy_gen)
                return

            if self._state == 'recording':
                self._record_buf.append(chunk)
                if energy < self.energy_thresh:
                    self._sil_count += 1
                else:
                    self._sil_count = 0
                # 2026-08-13: the VAD used to also reset _sil_count on its own
                # ("may only EXTEND a recording, never end one"), so the DSP's
                # speech flag alone could keep a turn open. Live on this
                # hardware it flips True/False constantly even at rest — see
                # project memory — so that reset fired on nearly every chunk
                # and every recording rode out to max_record_seconds instead of
                # ending on silence_limit. Silence is now decided by energy
                # alone; the VAD still gates the START of a recording (energy
                # AND vad must agree there), which is the check that was
                # actually validated live.

                if (self._sil_count  >= self._silence_limit_chunks or
                        len(self._record_buf) >= self._max_chunks):
                    audio = np.concatenate(self._record_buf)
                    self._state = 'idle'
                    threading.Thread(target=self._transcribe,
                                     args=(audio, self._busy_gen),
                                     daemon=True).start()

    def _dump_audio(self, audio: np.ndarray) -> str | None:
        """Write the exact buffer handed to Whisper, for offline diagnosis.

        Set the `debug_audio_dir` parameter to enable. Off by default — this is
        a microphone recording of the room, so it is never written unless
        somebody explicitly asks for it.
        """
        if not self.debug_audio_dir:
            return None
        try:
            import wave
            os.makedirs(self.debug_audio_dir, exist_ok=True)
            path = os.path.join(self.debug_audio_dir, f'utt_{time.time():.0f}.wav')
            with wave.open(path, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
            return path
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().warn(f'audio dump failed: {exc}')
            return None

    def _transcribe(self, audio: np.ndarray, gen: int):
        try:
            self.get_logger().info(f'Transcribing {len(audio)/SAMPLE_RATE:.1f}s rms={float(np.sqrt(np.mean(audio**2))):.4f}')
            dumped = self._dump_audio(audio)
            # beam_size=5 + a domain initial_prompt biases the decoder toward
            # the command vocabulary (room names / "πήγαινε στο…"), which
            # markedly improves Greek accuracy over greedy beam_size=1.
            # condition_on_previous_text=False: each utterance is an
            # independent command, so carrying decoder context across them buys
            # nothing and is a known driver of runaway repetition.
            # The prompt no longer opens with "Εντολές προς το ρομπότ Μαξ:" —
            # that framing sentence was the single most leaked string (7× in
            # one session), and it is meta-text the user can never say, so it
            # earned nothing while being maximally confusing when echoed.
            segments, _ = self._whisper.transcribe(
                audio, language=self.lang, beam_size=5,
                no_speech_threshold=0.3, vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt='σταμάτα, στοπ, πήγαινε στην κουζίνα, στο σαλόνι, '
                               'στον διάδρομο, στην τουαλέτα, στο δωμάτιο του Μαξ, '
                               'στο δωμάτιο του μπαμπά, πήγαινε στη βάση, '
                               'ακολούθησέ με, τι ώρα είναι, πόση μπαταρία έχεις.')
            raw = ' '.join(s.text for s in segments).strip()
            text = clean(raw)
            if text:
                if text != raw:
                    self.get_logger().warn(f'Prompt leakage trimmed: {raw!r} -> {text!r}')
                self.get_logger().info(f'Heard: {text}' + (f'  [{dumped}]' if dumped else ''))
                self.text_pub.publish(String(data=text))
            elif raw:
                self.get_logger().warn(f'Discarded prompt echo: {raw!r}')
            else:
                self.get_logger().info('Transcription empty')
        finally:
            self._end_turn(gen)


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
