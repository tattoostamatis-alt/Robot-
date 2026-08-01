"""Tests for what happens when the serial link fails mid-drive.

‼️ The bug these exist for. `drive_direct` was called bare, in a 20 Hz timer
callback, with no error handling anywhere in _motor_control. That is not a
dropped frame:

  * the OI holds the last velocity it was given — it has no watchdog;
  * in `oi_mode='full'` (the default) the Roomba does not act on its own bumper
    or cliff either, so all of that protection lives in our code;
  * an exception out of the timer callback means no zero is ever sent.

So a link lost at 200 mm/s meant 200 mm/s until somebody caught the robot.

The second half is the bumper and cliff going stale: they are only refreshed
inside get_encoders(), so a stalled link freezes them at their last value while
bump_block_s quietly expires — the forward block lifting on a timer with
nothing watching the front of the robot.

Robot-free: the methods are lifted out of the driver onto a stub with a fake
serial bot, so no port and no rclpy context is needed.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_serial_link_safety.py -q
"""
import os
import re
import time

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = f'{PKG}/home_robot/nodes/roomba_driver.py'
_SRC = open(DRIVER, encoding='utf-8').read()


def _method(name):
    m = re.search(rf'^    def {name}\(.*?(?=\n    def )', _SRC, re.M | re.S)
    assert m, f'{name} changed shape'
    return m.group(0)


_ns = {'time': time}
exec(compile(  # noqa: S102 - test-only extraction
    'class Driver:\n' + _method('_drive') + _method('_on_write_failure')
    + _method('_sensors_fresh'),
    'roomba_link', 'exec'), _ns)
Driver = _ns['Driver']


class FakeBot:
    """A serial link that can be broken on demand."""

    def __init__(self, fail_after=None):
        self.writes = []
        self.fail_after = fail_after      # None = always healthy
        self.broken = False

    def drive_direct(self, right, left):
        if self.broken:
            raise OSError('[Errno 5] Input/output error')
        self.writes.append((right, left))
        if self.fail_after is not None and len(self.writes) >= self.fail_after:
            self.broken = True


class FakeLogger:
    def __init__(self):
        self.lines = []

    def _add(self, s):
        self.lines.append(s)

    error = warn = info = _add

    def text(self):
        return ' | '.join(self.lines)


@pytest.fixture
def drv():
    d = Driver()
    d.bot = FakeBot()
    d._log = FakeLogger()
    d.get_logger = lambda: d._log
    d._write_failures = 0
    d._last_write_ok = 0.0
    d._target_left = d._target_right = 100.0
    d._cur_left = d._cur_right = 100.0
    d._cur_angular = 0.5
    d.sensor_stale_s = 1.0
    d._last_ok_time = time.monotonic()
    d._connect_oi = lambda: None
    return d


# ── the healthy path is unchanged ────────────────────────────────────────────

def test_a_good_write_goes_straight_through(drv):
    assert drv._drive(120, 118) is True
    assert drv.bot.writes == [(120, 118)]
    assert drv._write_failures == 0


def test_a_good_write_marks_the_link_alive(drv):
    drv._drive(0, 0)
    assert drv._last_write_ok > 0


# ── the failure this file exists for ─────────────────────────────────────────

def test_a_failed_write_is_reported_not_raised(drv):
    """It runs in a 20 Hz timer callback. Raising took the timer with it, so
    nothing ever sent a zero afterwards."""
    drv.bot.broken = True
    assert drv._drive(200, 200) is False          # must not raise


def test_a_failed_write_tries_to_stop_the_robot(drv):
    """The OI holds the last speed, so stopping is the only thing that matters
    once a write fails. A single dropped byte is the common case: the drive
    write fails, and the zero that follows it gets through."""
    class Flaky(FakeBot):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def drive_direct(self, right, left):
            self.calls += 1
            if self.calls == 1:              # only the drive command is lost
                raise OSError('[Errno 5] Input/output error')
            self.writes.append((right, left))

    drv.bot = Flaky()

    assert drv._drive(200, 200) is False
    assert (0, 0) in drv.bot.writes, 'never tried to stop'


def test_the_speed_is_zeroed_so_a_recovered_link_does_not_lurch(drv):
    """Coming back at the old speed is how a robot that lost its link for two
    seconds resumes at full tilt the moment it returns."""
    drv.bot.broken = True
    drv._drive(200, 200)

    assert drv._target_left == 0.0 and drv._target_right == 0.0
    assert drv._cur_left == 0.0 and drv._cur_right == 0.0
    assert drv._cur_angular == 0.0


def test_the_first_failure_says_what_it_means(drv):
    """"Wheel command failed" alone reads as a dropped frame. It is not one."""
    drv.bot.broken = True
    drv._drive(200, 200)
    assert 'KEEPS ITS LAST SPEED' in drv._log.text()


def test_a_dead_link_is_reconnected(drv):
    reconnects = []
    drv._connect_oi = lambda: reconnects.append(1)
    drv.bot.broken = True

    drv._drive(200, 200)

    assert reconnects, 'never tried to reopen the port'


def test_reconnect_failure_is_not_fatal(drv):
    def boom():
        raise OSError('port is gone')
    drv._connect_oi = boom
    drv.bot.broken = True

    drv._drive(200, 200)          # must not raise


def test_recovery_is_announced_and_the_counter_resets(drv):
    drv.bot.broken = True
    drv._drive(200, 200)
    assert drv._write_failures > 0

    drv.bot.broken = False
    drv._drive(0, 0)

    assert drv._write_failures == 0
    assert 'Serial link back' in drv._log.text()


def test_failures_are_not_logged_on_every_tick(drv):
    """At 20 Hz an unplugged robot would write 1200 identical lines a minute
    and bury the reason."""
    drv.bot.broken = True
    for _ in range(60):
        drv._drive(200, 200)
    loud = [ln for ln in drv._log.lines if 'reconnecting' in ln]
    assert len(loud) <= 3, f'{len(loud)} repeats of the same error'


# ── stale bumper/cliff readings ──────────────────────────────────────────────

def test_fresh_readings_are_trusted(drv):
    assert drv._sensors_fresh() is True


def test_stale_readings_are_not(drv):
    drv._last_ok_time = time.monotonic() - 5.0
    assert drv._sensors_fresh() is False


def test_nothing_read_yet_counts_as_stale(drv):
    """Before the first successful read the robot has no idea what is in front
    of it, and must not be allowed to drive into it."""
    drv._last_ok_time = 0.0
    assert drv._sensors_fresh() is False


def test_the_staleness_check_is_wired_into_the_forward_block():
    """The check is worth nothing if _motor_control does not consult it."""
    body = _SRC.split('def _motor_control')[1].split('\n    def ')[0]
    assert '_sensors_fresh()' in body, 'stale sensors no longer block forward'
    assert '_strip_forward' in body


def test_stale_sensors_block_forward_but_not_turning():
    """Same shape as a real bump: forward is cancelled, the turn survives, and
    reverse is never routed through the block at all."""
    body = _SRC.split('def _motor_control')[1].split('\n    def ')[0]
    stale = body.split('_sensors_fresh()')[1][:400]
    assert '_is_forward(' in stale
    assert '_strip_forward(' in stale
    assert 'target_left = target_right = 0' not in stale, 'this is not a full stop'


def test_every_wheel_write_goes_through_the_guarded_path():
    """A single bare drive_direct left is a path with no failure handling."""
    bare = [ln.strip() for ln in _SRC.split('\n')
            if 'self.bot.drive_direct(' in ln]
    # The only permitted ones are inside _drive/_on_write_failure themselves.
    guarded = _method('_drive') + _method('_on_write_failure')
    for ln in bare:
        assert ln in guarded, f'unguarded wheel write: {ln}'
