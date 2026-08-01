"""Regression tests for the third 2026-08-01 audit pass — the voice pipeline
going deaf or mute without saying so.

  * stt_node's state machine counts every one of its timeouts in CHUNKS, so it
    only advances while audio arrives. If mic/audio stopped mid-turn the node
    sat in flushing/waiting_speech/recording holding _busy forever, and every
    later wake word was refused with "Already transcribing" — permanently, no
    error. It had no timer at all.

  * wake_word_node fed an UNBOUNDED queue from the realtime audio callback. Any
    time the model fell behind realtime (flm + Whisper + YOLO loaded, swap
    thrashing) the backlog grew without limit and detections were scored
    against ever-older audio, so the wake word "stopped working" while actually
    firing many seconds late.

  * wake_word_node's mic stream was a bare `with sd.InputStream(...)`: if the
    device went away the thread simply ended and nothing was published on
    mic/audio again — wake word AND STT deaf, node still alive, no log line.

  * tts_node decoded MP3 through ffmpeg with no timeout, the same unbounded
    wait that already cost 36-46 s hangs on the synth side.

Robot-free: ROS and the audio stack are stubbed.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_audio_pipeline_regressions.py -q
"""
import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_safety_regressions import (        # noqa: E402
    _BaseNode, _load, _mod, _ns)


# ── stt_node ────────────────────────────────────────────────────────────────

@pytest.fixture
def stt(monkeypatch):
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'rcl_interfaces': _mod('rcl_interfaces'),
        'rcl_interfaces.msg': _mod(
            'rcl_interfaces.msg',
            SetParametersResult=lambda successful=True: _ns(successful=successful)),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _mod('std_msgs.msg',
                             String=lambda data='': _ns(data=data),
                             Bool=lambda data=False: _ns(data=data),
                             Int16MultiArray=lambda: _ns(data=[])),
    }
    mod = _load(f'{PKG}/home_robot/nodes/stt_node.py', mods)
    # The constructor kicks off a Whisper load thread; keep it off.
    with monkeypatch.context() as m:
        m.setattr(mod.threading, 'Thread',
                  lambda *a, **k: _ns(start=lambda: None))
        node = mod.STTNode()
    return mod, node


def _stall(node, seconds):
    """Pretend the last mic chunk arrived `seconds` ago."""
    import time as _t
    node._last_audio = _t.monotonic() - seconds


def test_watchdog_unsticks_a_turn_when_the_mic_goes_silent(stt):
    """‼️ THE BUG: with mic/audio gone the chunk counters never advanced, so
    _busy was held for good and every later wake word was refused."""
    mod, node = stt
    node._busy.acquire()
    node._state = 'waiting_speech'
    _stall(node, 60.0)

    node._audio_watchdog()

    assert node._state == 'idle'
    assert not node._busy.locked(), 'STT stayed busy forever'


def test_watchdog_frees_a_stalled_recording(stt):
    mod, node = stt
    node._busy.acquire()
    node._state = 'recording'
    node._record_buf = [np.zeros(10, dtype=np.float32)]
    _stall(node, 60.0)

    node._audio_watchdog()

    assert node._state == 'idle'
    assert node._record_buf == []
    assert not node._busy.locked()


def test_watchdog_leaves_an_idle_node_alone(stt):
    """No microphone while idle is quiet, not broken — and must not release a
    lock the node does not hold."""
    mod, node = stt
    node._state = 'idle'
    _stall(node, 60.0)

    node._audio_watchdog()

    assert node._state == 'idle'
    assert not node._busy.locked()


def test_watchdog_does_not_interrupt_a_live_turn(stt):
    """Audio still flowing = healthy; the watchdog must keep its hands off."""
    mod, node = stt
    node._busy.acquire()
    node._state = 'recording'
    _stall(node, 0.0)

    node._audio_watchdog()

    assert node._state == 'recording'
    assert node._busy.locked()


def test_incoming_audio_feeds_the_watchdog_clock(stt):
    mod, node = stt
    _stall(node, 60.0)
    node._state = 'idle'
    node._on_audio(_ns(data=[0] * 32))
    import time as _t
    assert _t.monotonic() - node._last_audio < 1.0


# ── wake_word_node ──────────────────────────────────────────────────────────

@pytest.fixture
def wake(monkeypatch):
    """wake_word_node does a lot at import time (onnxruntime, openwakeword,
    sounddevice, scipy); stub the lot."""
    ort = _mod('onnxruntime', InferenceSession=lambda *a, **k: None)
    oww_model = _mod('openwakeword.model', ort=_ns(InferenceSession=None),
                     Model=lambda **k: _ns(models={'hey_robot': None},
                                           predict=lambda c: {}))
    mods = {
        'onnxruntime': ort,
        'openwakeword': _mod('openwakeword',
                             get_pretrained_model_paths=lambda: []),
        'openwakeword.model': oww_model,
        'openwakeword.utils': _mod('openwakeword.utils', ort=_ns(InferenceSession=None)),
        'openwakeword.vad': _mod('openwakeword.vad', ort=_ns(InferenceSession=None)),
        'sounddevice': _mod('sounddevice', query_devices=lambda: [],
                            InputStream=lambda **k: None, play=lambda *a, **k: None,
                            wait=lambda: None, stop=lambda: None),
        'scipy': _mod('scipy'),
        'scipy.signal': _mod('scipy.signal',
                             butter=lambda *a, **k: (np.array([1.0]), np.array([1.0])),
                             lfilter=lambda *a, **k: (np.zeros(1), np.zeros(1)),
                             lfilter_zi=lambda *a, **k: np.zeros(1)),
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _mod('std_msgs.msg',
                             String=lambda data='': _ns(data=data),
                             Bool=lambda data=False: _ns(data=data),
                             Int16MultiArray=lambda: _ns(data=[])),
        'ament_index_python': _mod('ament_index_python'),
        'ament_index_python.packages': _mod(
            'ament_index_python.packages',
            get_package_share_directory=lambda _: '/tmp'),
    }
    mod = _load(f'{PKG}/home_robot/nodes/wake_word_node.py', mods)
    with monkeypatch.context() as m:
        m.setattr(mod.threading, 'Thread',
                  lambda *a, **k: _ns(start=lambda: None))
        node = mod.WakeWordNode()
    return mod, node


def test_audio_queue_is_bounded(wake):
    """‼️ THE BUG: an unbounded queue grew forever whenever the detector fell
    behind, so wake detection ran on ever-older audio."""
    mod, node = wake
    for i in range(mod.AUDIO_QUEUE_MAX * 10):
        node._enqueue(np.full(4, i, dtype=np.int16))

    assert node._audio_q.qsize() <= mod.AUDIO_QUEUE_MAX, (
        f'queue grew to {node._audio_q.qsize()}')


def test_overflow_drops_the_oldest_not_the_newest(wake):
    """Stale audio is worthless for wake detection; the freshest chunk must
    survive, or the backlog would just be trimmed from the wrong end."""
    mod, node = wake
    total = mod.AUDIO_QUEUE_MAX * 3
    for i in range(total):
        node._enqueue(np.full(4, i, dtype=np.int16))

    chunks = []
    while not node._audio_q.empty():
        chunks.append(int(node._audio_q.get_nowait()[0]))

    assert chunks[-1] == total - 1, 'the newest chunk was dropped'
    assert chunks == sorted(chunks), 'ordering broke'
    assert node._dropped_chunks > 0, 'drops were not counted'


def test_no_drops_while_the_detector_keeps_up(wake):
    mod, node = wake
    for i in range(mod.AUDIO_QUEUE_MAX):
        node._enqueue(np.full(4, i, dtype=np.int16))
    assert node._dropped_chunks == 0


def test_lost_microphone_is_retried_not_fatal(wake, monkeypatch):
    """‼️ THE BUG: a bare `with sd.InputStream(...)` meant a device that went
    away ended this thread for good — mic/audio silent, no log, node alive."""
    mod, node = wake
    opens = []

    class _Boom(Exception):
        pass

    def _flaky(**kw):
        opens.append(1)
        if len(opens) >= 3:
            raise KeyboardInterrupt        # break the retry loop for the test
        raise _Boom('device disappeared')

    monkeypatch.setattr(mod.sd, 'InputStream', _flaky)
    monkeypatch.setattr(mod.time, 'sleep', lambda *_: None)

    with pytest.raises(KeyboardInterrupt):
        node._listen_loop(-1)

    assert len(opens) >= 2, 'the stream was never reopened after failing'


# ── tts_node ────────────────────────────────────────────────────────────────

def test_mp3_decode_is_bounded():
    """‼️ THE BUG: ffmpeg was run with no timeout. A wedged decode blocks the
    only playback thread, which leaves the robot mute AND never republishes
    tts/speaking=False — which also holds the STT gate shut."""
    src = open(f'{PKG}/home_robot/nodes/tts_node.py').read()
    decode = src.split('def _decode_mp3')[1].split('def ')[0]
    assert 'timeout=' in decode, '_decode_mp3 still runs ffmpeg unbounded'


# ── arm_driver ──────────────────────────────────────────────────────────────

@pytest.fixture
def arm(monkeypatch):
    class _SerialException(Exception):
        pass

    opened = []

    class _Port:
        """A port that dies on demand, then can be reopened."""
        def __init__(self, *a, **k):
            opened.append(1)
            self.dead = False

        @property
        def in_waiting(self):
            if self.dead:
                raise _SerialException('device disconnected')
            return 0

        def readline(self):
            return b''

        def write(self, _b):
            if self.dead:
                raise _SerialException('device disconnected')

        def close(self):
            pass

    serial_mod = _mod('serial', Serial=_Port, SerialException=_SerialException)
    mods = {
        'serial': serial_mod,
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'sensor_msgs': _mod('sensor_msgs'),
        'sensor_msgs.msg': _mod('sensor_msgs.msg',
                                JointState=lambda: _ns(header=_ns(stamp=None),
                                                       name=[], position=[])),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _mod('std_msgs.msg',
                             String=lambda data='': _ns(data=data),
                             Float32=lambda data=0.0: _ns(data=data)),
    }
    mod = _load(f'{PKG}/home_robot/nodes/arm_driver.py', mods)
    return mod, mod.ArmDriver(), opened


def test_arm_survives_the_serial_going_away(arm):
    """‼️ THE BUG: _read_serial caught only JSONDecodeError, so a SerialException
    from in_waiting escaped a TIMER callback and took rclpy's spin down with it
    — the whole arm driver died instead of waiting for the arm to come back."""
    mod, node, opened = arm
    node.ser.dead = True

    node._read_serial()          # must not raise

    assert node.ser is None, 'the dead port was kept'


def test_arm_reopens_after_a_drop(arm, monkeypatch):
    mod, node, opened = arm
    before = len(opened)
    node.ser.dead = True
    node._read_serial()
    node._reopen_at = 0.0        # pretend the backoff elapsed

    node._read_serial()          # this call should reopen

    assert len(opened) > before, 'the port was never reopened'
    assert node.ser is not None


def test_arm_pose_is_forgotten_across_a_reconnect(arm):
    """T:102 sets ALL joints at once, so acting on a pre-unplug pose would snap
    un-commanded joints to a stale value."""
    mod, node, opened = arm
    node._current_joints = {'base': 0.5}
    node.ser.dead = True
    node._read_serial()
    node._reopen_at = 0.0
    node._read_serial()

    assert node._current_joints is None


def test_arm_write_failure_does_not_raise(arm):
    mod, node, opened = arm
    node.ser.dead = True
    node._send_json({'T': 105})      # must not raise
    assert node.ser is None
