#!/usr/bin/env python3
"""Streaming text-to-speech — edge-tts, triggered by speech_response.

Subscribes to `speech_response` (std_msgs/String, from llm_bridge_node) and
speaks each message with an edge-tts neural voice (default Greek female,
"Max" persona). Messages are queued and played back sequentially on a background
thread so the ROS callback never blocks.

edge-tts returns MP3 audio; ffmpeg (already used for wake-word training data
in this project) decodes it to PCM via a subprocess pipe before playback
with sounddevice.

edge-tts is a free, unauthenticated endpoint that fails ~20% of the time, so
there are two layers of protection: _synthesize makes `attempts` tries (2 by
default), and if that still comes back empty a local piper voice speaks instead
(home_robot/tts_fallback.py). Sample rate is therefore per-utterance, not a
constant — the local voice runs at 16 kHz against edge-tts's 24 kHz.
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

from home_robot import tts_fallback
from home_robot.voice_gate import TOPIC as SPEAKING_TOPIC, STOP_TOPIC

SAMPLE_RATE = 24000  # edge-tts default output rate


class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')

        # el-GR-AthinaNeural (female) by user preference 2026-07-25; the male
        # el-GR-NestorasNeural is the only other Greek edge-tts voice.
        self.declare_parameter('voice', 'el-GR-AthinaNeural')
        # `robot max` overrides this to piper: one consistent, offline Greek
        # female voice with no network wait. Keep edge available as an explicit
        # option for future comparisons.
        self.declare_parameter('backend', 'edge')
        self.declare_parameter('rate', '+0%')
        self.declare_parameter('volume', '+0%')
        self.declare_parameter('device_index', 7)  # pulse — works with any PulseAudio output
        # ‼️ Where the voice comes OUT decides whether echo cancellation can
        # work at all, and on THIS robot it cannot.
        #
        # The XVF3800 cancels the robot's own voice by subtracting what it is
        # playing — a reference it only has for audio sent through ITS output.
        # Here the sound goes to the motherboard's headphone jack (confirmed
        # with the user 2026-08-03: nothing is plugged into the ReSpeaker's own
        # 3.5 mm jack), so the DSP never sees the reference, cannot cancel
        # anything, and the robot hears itself say the wake word.
        #
        # THAT is why barge-in kept firing on the robot's own replies — not a
        # badly tuned threshold. No amount of threshold work fixes it, because
        # a reply and a real interruption reach the detector as the same audio.
        # The software answer is in voice_gate.utterance_is_risky(): refuse
        # barge-in for replies that contain the wake word.
        #
        # If a speaker is ever wired to the ReSpeaker, set this to 'XVF3800'
        # and the AEC loop closes for real. Empty keeps device_index.
        self.declare_parameter('device_name', '')
        # Hold `tts/speaking` True this long after playback ends so the listeners
        # ride out the room-reverb tail before re-arming (see voice_gate.py).
        self.declare_parameter('speaking_tail', 0.3)
        # Local voice used only when edge-tts fails outright (see
        # home_robot/tts_fallback.py). Empty string disables it.
        #
        # ‼️ DISABLED BY DEFAULT since 2026-08-01, at the user's request. The
        # fallback worked, and that was the problem: piper's voice
        # (el_GR-joy-medium) is a different Greek woman from edge-tts's
        # Athina, so an answer that fell back sounded like a second person
        # joining the conversation. The user reported it as "sometimes two
        # women with different voices answer me" and chose one consistent
        # voice over never being mute.
        #
        # The cost is known and accepted: when edge-tts fails every attempt,
        # the robot now says NOTHING. Set fallback_model back to
        # tts_fallback.DEFAULT_MODEL to restore the old behaviour.
        self.declare_parameter('fallback_model', '')
        # Give up on an edge-tts attempt after this long and let the retry (then
        # the local voice) take over. A healthy synth measures 0.3-1.1 s, so 3 s
        # is ~3x headroom and still an order of magnitude below the 36-46 s that
        # untimed hung attempts cost on 2026-07-30. See _synthesize().
        self.declare_parameter('synth_timeout', 3.0)
        # Raised 2 -> 4 on 2026-08-01, when the local fallback was switched off.
        # edge-tts fails server-side and independently — the SAME text that
        # failed synthesises fine on the next call — so at a measured ~30% per
        # attempt, 2 attempts leave ~9% of answers mute and 4 leave under 1%.
        # The extra wait only happens in the runs that were already failing,
        # and silence is now the alternative to waiting.
        self.declare_parameter('synth_attempts', 4)

        self.voice = self.get_parameter('voice').value
        self.backend = self.get_parameter('backend').value.strip().lower()
        self.rate = self.get_parameter('rate').value
        self.volume = self.get_parameter('volume').value
        self.device_index = self.get_parameter('device_index').value
        name_hint = (self.get_parameter('device_name').value or '').strip()
        if name_hint:
            self.device_index = self._find_output(name_hint)
        self.speaking_tail = self.get_parameter('speaking_tail').value
        self.synth_timeout = self.get_parameter('synth_timeout').value
        self.synth_attempts = max(1, int(self.get_parameter('synth_attempts').value))

        fallback_model = self.get_parameter('fallback_model').value
        self._fallback = (tts_fallback.PiperFallback(fallback_model,
                                                     self.get_logger())
                          if fallback_model else None)
        # Say so at startup: discovering there is no fallback in the middle of
        # an outage is exactly when it is least useful to find out.
        if self._fallback and self._fallback.available():
            self.get_logger().info(f'Local TTS fallback ready: {fallback_model}')
        else:
            self.get_logger().warn(
                'Single voice mode: no local fallback, so the robot stays '
                'SILENT if edge-tts fails every attempt. Watch for '
                '"TTS failed" in this log.')

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

        active_voice = ('Piper el_GR-joy-medium'
                        if self.backend == 'piper' else self.voice)
        self.get_logger().info(
            f'TTS node started — backend={self.backend} voice={active_voice}')

    def _find_output(self, hint: str) -> int:
        """Index of the first output device whose name contains `hint`.

        Falls back to the configured device_index — and says so — rather than
        raising: a mute robot is a worse failure than one whose echo
        cancellation is not closed.
        """
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_output_channels'] > 0 and hint.lower() in dev['name'].lower():
                    self.get_logger().info(
                        f"Speaking through '{dev['name']}' (index {i})")
                    return i
        except Exception as e:
            self.get_logger().warn(f'could not enumerate audio outputs: {e}')
        self.get_logger().warn(
            f"no output device matching '{hint}' — falling back to index "
            f'{self.device_index}. Echo cancellation will NOT work if that is '
            'not the ReSpeaker.')
        return self.device_index

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
        # Rate is per-utterance because the local fallback speaks at its own
        # (16 kHz for the Greek piper voice, vs edge-tts's 24 kHz).
        rate = SAMPLE_RATE
        if self.backend == 'piper':
            local = self._fallback.synthesize(text) if self._fallback else None
            if local is None:
                raise RuntimeError('Piper backend requested but the voice is unavailable')
            audio, rate = local
        else:
            try:
                mp3 = asyncio.run(self._synthesize(text))
                pcm = self._decode_mp3(mp3)
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception as e:                          # noqa: BLE001
            # Network TTS is out. Falling back matters more than it sounds:
            # everything upstream worked and only the audio is missing, so
            # staying silent here reads to the user as "the robot is broken".
                local = self._fallback.synthesize(text) if self._fallback else None
                if local is None:
                    raise
                audio, rate = local
                self.get_logger().warn(
                    f'edge-tts unavailable ({e}) — speaking with the local voice')

        if self._interrupted:   # barge-in landed during synthesis — don't start playing
            return
        device = None if self.device_index < 0 else self.device_index
        self.get_logger().info(f'Speaking: {text}')
        sd.play(audio, samplerate=rate, device=device)

        # Bounded wait instead of sd.wait(). sd.wait() blocks forever, and a
        # sd.stop() arriving from the barge-in callback while sd.play() is still
        # starting up could leave it hung: the playback thread then never
        # returned, so the robot went permanently silent AND never published
        # tts/speaking=False — which also keeps the STT gate closed, so it
        # stopped hearing commands too. Both symptoms, no error logged, node
        # still alive. Seen for real on 2026-07-23 after four rapid barge-ins.
        # Polling to a deadline derived from the audio length cannot hang.
        deadline = time.monotonic() + len(audio) / rate + 2.0
        while time.monotonic() < deadline and not self._interrupted:
            time.sleep(0.05)
        try:
            sd.stop()
        except Exception:
            pass

    async def _synthesize(self, text, attempts=None):
        attempts = self.synth_attempts if attempts is None else attempts
        # edge-tts talks to a free, unauthenticated Microsoft endpoint that
        # throttles: it intermittently closes the stream having sent no audio
        # ("NoAudioReceived"), or returns an empty one. Measured 2026-07-26 —
        # 5 failures in one session, and the *same* text that failed
        # synthesised fine on the next call, so it is server-side flakiness,
        # not the text or the parameters. Without a retry a single hiccup left
        # the robot mute for that whole answer, which reads exactly like "it
        # isn't responding" since the LLM reply only ever reached the log.
        #
        # The retry alone was not enough. edge-tts sets no network timeout, so a
        # throttled attempt does not fail fast — it hangs until the socket times
        # out on its own. Measured on 2026-07-30 logs: 8 of 27 answers waited
        # 36-46 s between the reply reaching the log and audio starting, every
        # one of them followed by "recovered on attempt 2". A healthy call takes
        # 0.3-1.1 s, so any attempt still running after SYNTH_TIMEOUT is a hung
        # one, never a slow-but-fine one: cutting it loose costs nothing and
        # hands over to the local piper voice (0.04 s) while the user is still
        # waiting rather than 40 s later.
        last = None
        for i in range(attempts):
            try:
                data = await asyncio.wait_for(
                    self._synth_once(text), timeout=self.synth_timeout)
                if data:
                    if i:
                        self.get_logger().warn(f'TTS synth recovered on attempt {i + 1}')
                    return data
                last = 'empty audio stream'
            except asyncio.TimeoutError:
                last = f'no audio within {self.synth_timeout:.1f}s'
                self.get_logger().warn(f'TTS attempt {i + 1} timed out ({last})')
            except Exception as e:                       # noqa: BLE001
                last = e
            if i < attempts - 1:
                await asyncio.sleep(0.2)
        raise RuntimeError(f'no audio after {attempts} attempts: {last}')

    async def _synth_once(self, text):
        communicate = edge_tts.Communicate(
            text, self.voice, rate=self.rate, volume=self.volume)
        chunks = []
        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                chunks.append(chunk['data'])
        return b''.join(chunks)

    def _decode_mp3(self, mp3_bytes):
        # ‼️ Bounded, for exactly the reason _synthesize is. A pipe-fed ffmpeg
        # that wedges would block the single playback thread forever: the robot
        # goes mute AND never republishes tts/speaking=False, which also holds
        # the STT gate shut, so it stops hearing commands too — the same
        # double symptom the sd.wait() hang produced on 2026-07-23. A decode of
        # a few seconds of speech takes tens of milliseconds; 10 s is not a
        # budget, it is a "this will never finish" line. Timing out raises, and
        # the caller falls back to the local piper voice.
        proc = subprocess.run(
            ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1', 'pipe:1'],
            input=mp3_bytes, capture_output=True, check=True, timeout=10.0)
        return proc.stdout


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
