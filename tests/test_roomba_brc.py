"""Tests for the Roomba BRC (wake line) pulse interlock in roomba_driver.py.

Why this file exists: per the iRobot Open Interface spec, pulsing BRC low
**three times within 5 seconds switches the robot to 19200 baud**. The old code
called pycreate2's `wake()` (two pulses per call) from a 2-second timer whenever
the robot was unresponsive, which is comfortably over that limit -- harmless
only because BRC is not physically wired on this robot yet. The moment someone
runs the wire, that pattern would silently reconfigure the robot and it would
look *more* broken than merely asleep, with nothing in any log to say why.

So the rate limit is a safety interlock, not tuning, and it gets a test. These
run robot-free: `_brc_pulse` is exercised on a stand-in object holding only the
state it touches, so no serial port, ROS node or hardware is involved.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_roomba_brc.py -q
"""
import os
import re
import types


PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = f'{PKG}/home_robot/nodes/roomba_driver.py'

# The OI limit we must stay clear of.
OI_BAUD_CHANGE_PULSES = 3
OI_BAUD_CHANGE_WINDOW_S = 5.0


def _load_brc_pulse():
    """Pull `_brc_pulse` off the class without importing rclpy/pycreate2.

    roomba_driver.py imports the whole ROS stack at module level, which is not
    available (or wanted) in a unit test. The method only touches attributes we
    can supply ourselves, so compile the file and lift the function out.
    """
    src = open(DRIVER).read()
    match = re.search(r'\n    def _brc_pulse\(self\):.*?(?=\n    def )', src, re.S)
    assert match, '_brc_pulse not found — did the method get renamed?'
    body = 'import time\n' + '\n'.join(
        line[4:] for line in match.group(0).splitlines()
    )
    ns = {}
    exec(compile(body, DRIVER, 'exec'), ns)
    return ns['_brc_pulse']


class _FakeThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
        self._alive = False

    def start(self):
        # Deliberately does NOT run the target: these tests are about the
        # decision to pulse, not the pulse itself (which touches a serial port).
        self._alive = True

    def is_alive(self):
        return self._alive


def _driver(clock, min_interval=10.0):
    """Minimal stand-in exposing exactly the attributes _brc_pulse uses."""
    d = types.SimpleNamespace(
        _last_brc_pulse=-1e9,          # far in the past: first pulse allowed
        _brc_thread=None,
        brc_min_interval_s=min_interval,
        brc_pulse_s=0.6,
        # Referenced as the thread target; never invoked, since _FakeThread.start
        # does not run it (that path would open a serial port).
        _do_brc_pulse=lambda: None,
        pulses=[],
    )
    d._clock = clock
    return d


def _call(pulse_fn, driver, now, threads_finish=True):
    """Invoke _brc_pulse at time `now`, recording whether it fired."""
    fn_globals = pulse_fn.__globals__
    fn_globals['time'] = types.SimpleNamespace(monotonic=lambda: now)
    fn_globals['threading'] = types.SimpleNamespace(Thread=_FakeThread)

    before = driver._last_brc_pulse
    pulse_fn(driver)
    fired = driver._last_brc_pulse != before
    if fired:
        driver.pulses.append(now)
    if threads_finish and driver._brc_thread is not None:
        driver._brc_thread._alive = False    # pretend the pulse completed
    return fired


def test_first_pulse_fires():
    pulse = _load_brc_pulse()
    d = _driver(None)
    assert _call(pulse, d, now=1000.0) is True


def test_rapid_calls_cannot_reach_the_baud_change_threshold():
    """The real regression: _keep_awake used to call this every 2 seconds while
    the robot was unresponsive. Simulate that and count pulses in any 5s window."""
    pulse = _load_brc_pulse()
    d = _driver(None, min_interval=10.0)

    t = 1000.0
    for _ in range(60):                 # two minutes of 2-second retries
        _call(pulse, d, now=t)
        t += 2.0

    for i, start in enumerate(d.pulses):
        in_window = [p for p in d.pulses if start <= p < start + OI_BAUD_CHANGE_WINDOW_S]
        assert len(in_window) < OI_BAUD_CHANGE_PULSES, (
            f'{len(in_window)} pulses within {OI_BAUD_CHANGE_WINDOW_S}s starting at '
            f'{start} — the OI would switch to 19200 baud')


def test_pulse_is_allowed_again_after_the_interval():
    pulse = _load_brc_pulse()
    d = _driver(None, min_interval=10.0)
    assert _call(pulse, d, now=1000.0) is True
    assert _call(pulse, d, now=1005.0) is False    # too soon
    assert _call(pulse, d, now=1010.5) is True     # past the interval


def test_no_overlapping_pulses_while_one_is_in_flight():
    """A pulse holds the line low for ~0.6s; starting a second one meanwhile
    would produce a ragged waveform the OI could read as extra pulses."""
    pulse = _load_brc_pulse()
    d = _driver(None, min_interval=10.0)
    assert _call(pulse, d, now=1000.0, threads_finish=False) is True
    # Interval satisfied, but the previous pulse thread is still running.
    assert _call(pulse, d, now=1020.0, threads_finish=False) is False
    d._brc_thread._alive = False
    assert _call(pulse, d, now=1030.0) is True


def test_driver_no_longer_calls_pycreate2_wake():
    """pycreate2's wake() emits two pulses AND blocks for 3 seconds; it must not
    come back into the driver by accident."""
    src = open(DRIVER).read()
    assert 'bot.wake()' not in src, (
        'pycreate2 wake() is back: it double-pulses BRC and blocks 3s inside a '
        '2s timer callback')
