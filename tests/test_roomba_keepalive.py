"""Tests for the wake path in scripts/roomba_keepalive.py.

Why this file exists: the daemon spent months writing a "wake pulse" that could
not possibly wake anything. Opening a tty asserts DTR, so the BRC line is
already LOW before the code touches it -- and the old sequence was
``ser.dtr = True`` (low → low) then False. The robot therefore saw BRC held down
forever and never a falling edge, which is what BRC actually triggers on. The
failure is invisible from the outside: a robot that ignores a broken pulse looks
exactly like a robot whose wake wire is cut, so the bug survived a hardware
investigation that concluded "BRC is not wired".

That makes the *shape of the pulse* worth pinning down in a test: release HIGH,
settle, pull LOW past the OI's 500ms minimum, release. These run robot-free --
os.open/os.read/ioctl are stubbed, so no serial port or hardware is involved.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_roomba_keepalive.py -q
"""
import importlib.util
import os
import termios

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = f'{PKG}/scripts/roomba_keepalive.py'

# The OI's own numbers, restated here so a regression in the constants is caught
# rather than silently accepted by reading them from the module under test.
OI_MIN_BRC_LOW_S = 0.5
OI_BAUD_CHANGE_PULSES = 3
OI_BAUD_CHANGE_WINDOW_S = 5.0


def _load():
    spec = importlib.util.spec_from_file_location('roomba_keepalive', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ka(monkeypatch):
    """The module with the serial layer replaced by a recorder.

    ``events`` collects ('low'|'high', t) for BRC transitions and ('write', b)
    for OI bytes, on a fake clock so the test never actually sleeps.
    """
    mod = _load()
    events = []
    clock = {'t': 0.0}

    def fake_ioctl(fd, request, arg):
        if request == termios.TIOCMBIS:
            events.append(('low', clock['t']))
        elif request == termios.TIOCMBIC:
            events.append(('high', clock['t']))
        return arg

    monkeypatch.setattr(mod.fcntl, 'ioctl', fake_ioctl)
    monkeypatch.setattr(mod.os, 'open', lambda *a, **k: 42)
    monkeypatch.setattr(mod.os, 'close', lambda fd: None)
    monkeypatch.setattr(mod.os, 'write', lambda fd, b: events.append(('write', bytes(b))))
    monkeypatch.setattr(mod.termios, 'tcgetattr', lambda fd: [0, 0, 0, 0, 0, 0, [0] * 32])
    monkeypatch.setattr(mod.termios, 'tcsetattr', lambda fd, when, attrs: None)
    monkeypatch.setattr(mod.termios, 'tcflush', lambda fd, q: None)
    monkeypatch.setattr(mod.time, 'sleep', lambda s: clock.update(t=clock['t'] + s))

    mod._events = events
    mod._clock = clock
    return mod


def _transitions(events):
    return [e for e in events if e[0] in ('low', 'high')]


def test_pulse_releases_the_line_before_pulling_it_low(ka):
    """The regression. os.open() leaves DTR asserted (line LOW), so the very
    first thing a pulse must do is let it go HIGH -- otherwise there is no edge
    to fall."""
    ka._brc_pulse(42)
    first = _transitions(ka._events)[0]
    assert first[0] == 'high', (
        'the pulse drove the line LOW without releasing it first: no falling '
        'edge reaches the robot, exactly the bug this file exists for')


def test_pulse_is_high_then_low_then_high(ka):
    ka._brc_pulse(42)
    assert [e[0] for e in _transitions(ka._events)] == ['high', 'low', 'high']


def test_low_phase_is_long_enough_for_the_oi(ka):
    """The OI ignores a BRC low shorter than ~500ms."""
    ka._brc_pulse(42)
    _, low_t = _transitions(ka._events)[1]
    _, release_t = _transitions(ka._events)[2]
    assert release_t - low_t >= OI_MIN_BRC_LOW_S


def test_line_settles_high_before_the_edge(ka):
    """A falling edge is only unambiguous if the line sat high first."""
    ka._brc_pulse(42)
    _, high_t = _transitions(ka._events)[0]
    _, low_t = _transitions(ka._events)[1]
    assert low_t - high_t > 0, 'the release and the pull-low are simultaneous'


def test_awake_robot_is_not_pulsed(ka, monkeypatch):
    """An awake robot is already in Full mode and needs no edge. Not pulsing it
    also keeps us clear of the OI's three-pulses-in-5s baud switch."""
    monkeypatch.setattr(ka, '_handshake', lambda fd: True)
    ka.assert_full_mode(pulse=False)
    assert _transitions(ka._events) == []


def test_cycle_cannot_reach_the_baud_change_threshold(ka, monkeypatch):
    """Per the OI spec, three BRC pulses within 5 seconds switch the robot to
    19200 baud, where it is alive but mute to us. The steady-state cycle pulses
    at most once per INTERVAL_S, and the boot burst spaces its retries out."""
    assert ka.INTERVAL_S >= OI_BAUD_CHANGE_WINDOW_S * (OI_BAUD_CHANGE_PULSES - 1)
    assert ka.BOOT_WAKE_GAP_S >= OI_BAUD_CHANGE_WINDOW_S


def test_cycle_stays_inside_the_879_sleep_window(ka):
    """The 879 sleeps after ~90s in Passive mode, not the spec's 5 minutes
    (measured 2026-07-27). A slower cycle would let it doze off between pokes."""
    assert ka.INTERVAL_S < 90.0


def test_failed_115200_handshake_retries_at_19200(ka, monkeypatch):
    """A past triple-pulse can leave the robot on 19200, where it is awake but
    silent to us -- indistinguishable from asleep unless we look."""
    bauds = []

    def fake_configure(fd, baud=termios.B115200):
        bauds.append(baud)

    monkeypatch.setattr(ka, '_configure', fake_configure)
    monkeypatch.setattr(ka, '_handshake', lambda fd: False)
    assert ka.assert_full_mode(pulse=True) is False
    assert termios.B19200 in bauds


def test_liveness_is_proved_by_a_reply_not_by_writing(ka, monkeypatch):
    """Writing start/full proves nothing: a sleeping Roomba swallows every byte.
    Only a sensor reply counts as awake."""
    monkeypatch.setattr(ka.os, 'read', lambda fd, n: b'')
    monkeypatch.setattr(ka, '_configure', lambda fd, baud=termios.B115200: None)
    assert ka._handshake(42) is False
    monkeypatch.setattr(ka.os, 'read', lambda fd, n: b'\x00\x64')
    assert ka._handshake(42) is True


def test_keepalive_no_longer_uses_pyserial(ka):
    """pyserial asserts DTR on open with no way to prevent it -- the exact
    behaviour that killed the falling edge. It must not come back."""
    src = open(SCRIPT).read()
    assert 'import serial' not in src, (
        'pyserial is back: it asserts DTR on open, which destroys the BRC '
        'falling edge')
