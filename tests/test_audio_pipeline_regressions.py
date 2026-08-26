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
        # stt_node subscribes to /voice_activity with transient_local QoS to
        # match doa_node's latched publisher.
        'rclpy.qos': _mod('rclpy.qos',
                          QoSProfile=lambda **kw: _ns(**kw),
                          QoSDurabilityPolicy=_ns(TRANSIENT_LOCAL=1, VOLATILE=2),
                          QoSReliabilityPolicy=_ns(RELIABLE=1, BEST_EFFORT=2)),
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


# ── the stuck shape the first watchdog cannot see (2026-08-01, live) ────────
# Measured on the robot: 7 wake words in 25 s, every one refused with "Already
# transcribing", zero transcriptions, and stt_node holding no microphone handle
# at all. _state is set to 'idle' BEFORE the transcribe thread starts, so a
# Whisper call that hangs leaves _busy held while the state machine believes
# the turn is over and audio keeps flowing — both of _audio_watchdog's guards
# return early, forever.

def test_the_audio_watchdog_cannot_see_a_hung_transcription(stt):
    """Pinning WHY a second watchdog was needed: this is not a regression, it
    is the documented blind spot. If this ever starts failing, the two
    watchdogs have merged and one of them is redundant."""
    mod, node = stt
    gen = node._begin_turn()
    assert gen is not None
    node._state = 'idle'          # exactly what _on_audio does before the thread
    node._last_audio = __import__('time').monotonic()   # audio still flowing

    node._audio_watchdog()

    assert node._busy.locked(), 'the blind spot closed by accident'


def test_the_busy_watchdog_releases_a_hung_transcription(stt):
    mod, node = stt
    node._begin_turn()
    node._state = 'idle'
    node._busy_since = __import__('time').monotonic() - 300.0

    node._busy_watchdog()

    assert not node._busy.locked(), 'STT stayed deaf'


def test_the_busy_watchdog_ignores_audio_and_state(stt):
    """It must not inherit the guards that made the first one blind."""
    import inspect
    src = inspect.getsource(type(stt[1])._busy_watchdog)
    # Strip the docstring: it *describes* those two guards, deliberately.
    body = src.split('"""')[2] if src.count('"""') >= 2 else src
    assert '_last_audio' not in body, 'inherited the audio guard'
    assert 'self._state' not in body, 'inherited the state guard'


def test_a_transcription_still_running_is_left_alone(stt):
    mod, node = stt
    node._begin_turn()
    node._busy_since = __import__('time').monotonic() - 3.0   # normal: 2.0-4.2 s

    node._busy_watchdog()

    assert node._busy.locked(), 'a healthy transcription was cut off'


def test_an_idle_node_is_never_released(stt):
    mod, node = stt
    node._busy_since = __import__('time').monotonic() - 300.0

    node._busy_watchdog()          # _busy was never acquired

    assert not node._busy.locked()


def test_a_freed_hung_thread_cannot_release_the_next_turn(stt):
    """‼️ threading.Lock has no owner check. Without the turn id, the hung
    Whisper call finishing later would release the NEXT turn's lock and hand
    the node a second stuck state that looks exactly like the first."""
    mod, node = stt
    hung = node._begin_turn()
    node._busy_since = __import__('time').monotonic() - 300.0
    node._busy_watchdog()                      # watchdog frees the hung turn
    assert not node._busy.locked()

    fresh = node._begin_turn()                 # a new wake word arrives
    assert fresh is not None and fresh != hung

    node._end_turn(hung)                       # the hung thread finally returns

    assert node._busy.locked(), 'the stale owner released a live turn'


def test_the_owner_can_end_its_own_turn(stt):
    mod, node = stt
    gen = node._begin_turn()
    assert node._end_turn(gen) is True
    assert not node._busy.locked()
    assert node._end_turn(gen) is False, 'double release must be a no-op'


def test_a_second_wake_word_is_refused_while_busy(stt):
    mod, node = stt
    assert node._begin_turn() is not None
    assert node._begin_turn() is None


def test_both_watchdogs_are_actually_scheduled(stt):
    mod, node = stt
    names = {getattr(c, '__name__', '') for _, c in node.timers}
    assert '_audio_watchdog' in names
    assert '_busy_watchdog' in names, 'the hung-transcription watchdog never runs'


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


# ── barge-in that hears the loudspeaker ─────────────────────────────────────
# ‼️ THE BUG, measured on hardware 2026-08-04: 81 wake detections in a 20-minute
# session, 46 of them while the robot was speaking, scoring 0.78-1.00, every one
# followed by "No speech detected after wake word". The wake channel (ch4) is a
# raw capsule with no echo canceller, so the robot's own loudspeaker arrives as
# ordinary loud speech and no threshold can tell it from a person. The robot
# spent the evening cutting itself off and listening to an empty room.

def test_barge_in_disarms_itself_when_nobody_is_talking(wake):
    mod, node = wake
    node.allow_barge_in = True
    for _ in range(mod.BARGE_IN_DISARM_AFTER):
        node._barge_ins_without_speech += 1
        if node._barge_ins_without_speech >= mod.BARGE_IN_DISARM_AFTER:
            node._disarm_barge_in()
    assert node._barge_in_disarmed
    assert not node.allow_barge_in, 'it can still interrupt itself'


def test_a_real_interruption_resets_the_counter(wake):
    """Someone who interrupts and then SPEAKS is using the feature correctly;
    that must not count towards disarming it."""
    mod, node = wake
    node.allow_barge_in = True
    node._barge_ins_without_speech = mod.BARGE_IN_DISARM_AFTER - 1
    node._on_speech_text(_ns(data='σταμάτα'))
    assert node._barge_ins_without_speech == 0
    assert node.allow_barge_in


def test_empty_speech_does_not_count_as_someone_talking(wake):
    mod, node = wake
    node._barge_ins_without_speech = 2
    node._on_speech_text(_ns(data='   '))
    assert node._barge_ins_without_speech == 2


def test_disarming_is_announced_once(wake):
    mod, node = wake
    errors = []
    node.get_logger = lambda: _ns(info=lambda *a: None, warn=lambda *a: None,
                                  debug=lambda *a: None,
                                  error=lambda m: errors.append(m))
    node.allow_barge_in = True
    node._barge_ins_without_speech = mod.BARGE_IN_DISARM_AFTER
    node._disarm_barge_in()
    node._disarm_barge_in()
    assert len(errors) == 1, 'the explanation repeats every detection'
    assert 'barge-in' in errors[0]
    assert 'allow_barge_in:=false' in errors[0], 'no way to act on it'


def test_the_threshold_leaves_room_for_a_real_person(wake):
    """Two interruptions with nothing said can be a person changing their
    mind. Three cannot."""
    mod, _ = wake
    assert mod.BARGE_IN_DISARM_AFTER >= 3


# ── tts_node ────────────────────────────────────────────────────────────────

def test_mp3_decode_is_bounded():
    """‼️ THE BUG: ffmpeg was run with no timeout. A wedged decode blocks the
    only playback thread, which leaves the robot mute AND never republishes
    tts/speaking=False — which also holds the STT gate shut."""
    src = open(f'{PKG}/home_robot/nodes/tts_node.py').read()
    decode = src.split('def _decode_mp3')[1].split('def ')[0]
    assert 'timeout=' in decode, '_decode_mp3 still runs ffmpeg unbounded'


# ── arm_driver ──────────────────────────────────────────────────────────────

def _arm_modules(serial_mod):
    return {
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
    mod = _load(f'{PKG}/home_robot/nodes/arm_driver.py',
                _arm_modules(serial_mod))
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


def test_arm_can_start_unplugged_and_reconnect_later():
    class _SerialException(Exception):
        pass

    attempts = []

    class _Port:
        def __init__(self, *a, **k):
            attempts.append(1)
            if len(attempts) == 1:
                raise _SerialException('not plugged in')

        @property
        def in_waiting(self):
            return 0

        def close(self):
            pass

    serial_mod = _mod('serial', Serial=_Port,
                      SerialException=_SerialException)
    mod = _load(f'{PKG}/home_robot/nodes/arm_driver.py',
                _arm_modules(serial_mod))

    node = mod.ArmDriver()            # must not raise when /dev/arm is absent
    assert node.ser is None
    node._reopen_at = 0.0
    node._read_serial()
    assert node.ser is not None
    assert len(attempts) == 2


def test_arm_rejects_nonfinite_joint_and_gripper_commands(arm):
    mod, node, opened = arm
    node._current_joints = {
        'base': 0.0, 'shoulder': 0.0, 'elbow': 1.0,
        'wrist': 0.0, 'roll': 0.0, 'hand': 2.0,
    }
    sent = []
    node._send_json = sent.append

    node._joint_cmd_cb(_ns(name=['base'], position=[float('nan')]))
    node._gripper_cb(_ns(data=float('inf')))
    node._raw_cmd_cb(_ns(data='{"T":104,"x":NaN,"y":0,"z":100}'))

    assert sent == []


def test_arm_driver_is_the_final_cartesian_unit_guard(arm):
    mod, node, opened = arm
    sent = []
    node._send_json = sent.append

    # Metres accidentally sent to firmware as millimetres: inside the base.
    node._raw_cmd_cb(_ns(data='{"T":104,"x":0.3,"y":0,"z":0.1}'))
    # Impossible for this arm, usually a bad frame or unit conversion.
    node._raw_cmd_cb(_ns(data='{"T":104,"x":1000,"y":0,"z":0}'))
    assert sent == []

    node._raw_cmd_cb(_ns(data='{"T":104,"x":300,"y":0,"z":100}'))
    assert len(sent) == 1


def test_arm_rejects_impossible_feedback_pose(arm):
    mod, node, opened = arm
    node._home_pending = False
    feedback = {
        'v': 1318, 'b': 0.0, 's': 0.0, 'e': -1.0,
        't': 0.0, 'r': 0.0, 'g': 2.0,
    }

    node._parse_feedback(feedback)

    assert node._current_joints is None
    assert node.pubs['arm/joint_states'].sent == []


def test_arm_limits_cannot_escape_mechanical_range(arm):
    mod, node, opened = arm
    node._params['limit_base'] = [10.0, 20.0]

    assert node.limits('base') == (mod.MECH_LIMITS['base'][1],) * 2
    assert node._clamp('base', 15.0) == mod.MECH_LIMITS['base'][1]
