"""The Roomba must never be left in Passive, because Passive is where it sleeps.

Per the iRobot Open Interface spec the 5-minute inactivity sleep timer runs
only in Passive mode; Safe and Full never auto-sleep. Once this robot is
asleep nothing in software revives it — BRC is not wired on this chassis
(proven 2026-07-25, 282 pulses, no response), so only the CLEAN button does.

That makes a silent demotion to Passive — any safety abort does it, e.g. the
wheel drop when someone picks the robot up — an unrecoverable event dressed up
as nothing at all. On 2026-07-31 it cost a whole session: the robot slept at
20:40:24, a patrol started at 20:49:01, and six rooms were reported as
navigation timeouts while the pose never moved a millimetre.

These run robot-free: `_keep_awake` is exercised on a stand-in holding only
the state it touches, so no serial port, ROS node or hardware is involved.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_roomba_no_sleep.py -q
"""
import os
import re
import types

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = f'{PKG}/home_robot/nodes/roomba_driver.py'

OI_MODE_PASSIVE = 1
OI_MODE_SAFE = 2


def _keep_awake_fn():
    """Extract _keep_awake so the test needs neither rclpy nor a serial port."""
    src = open(DRIVER).read()
    body = re.search(r'^    def _keep_awake\(self\):.*?(?=^    def )', src,
                     re.S | re.M).group(0)
    ns = {'time': __import__('time'), 'OI_MODE_PASSIVE': OI_MODE_PASSIVE}
    exec(compile('class D:\n' + body, 'keepawake', 'exec'), ns)  # noqa: S102
    return ns['D']._keep_awake


class _Bot:
    def __init__(self):
        self.calls = []

    def _log(self, name):
        self.calls.append(name)


def _stub(oi_mode, *, ok_age=0.0, brc_age=0.0, docking=False):
    import time as _t
    now = _t.monotonic()
    d = types.SimpleNamespace(
        _docking=docking,
        _last_ok_time=now - ok_age,
        _last_brc_pulse=now - brc_age,
        _last_oi_mode=oi_mode,
        oi_mode='safe',
        calls=[],
    )
    d._connect_oi = lambda: d.calls.append('connect_oi')
    d._enter_drive_mode = lambda: d.calls.append('enter_drive_mode')
    d._brc_pulse = lambda: d.calls.append('brc_pulse')
    d.get_logger = lambda: types.SimpleNamespace(warn=lambda *_: None,
                                                 info=lambda *_: None)
    return d


def test_passive_is_escaped_immediately():
    d = _stub(OI_MODE_PASSIVE)
    _keep_awake_fn()(d)
    assert 'enter_drive_mode' in d.calls, \
        'left in Passive — the sleep timer is now running and only CLEAN will help'


def test_the_mode_is_re_read_rather_than_re_asserted_forever():
    d = _stub(OI_MODE_PASSIVE)
    _keep_awake_fn()(d)
    assert d._last_oi_mode is None, 'stale Passive would re-assert every cycle'


def test_safe_mode_is_left_alone():
    d = _stub(OI_MODE_SAFE)
    _keep_awake_fn()(d)
    assert 'enter_drive_mode' not in d.calls


def test_an_unresponsive_robot_still_wins_over_the_mode_check():
    # No good read for >2s means asleep or out of OI entirely; a mode value
    # from before that is stale, so reconnecting has to come first.
    d = _stub(OI_MODE_PASSIVE, ok_age=5.0)
    _keep_awake_fn()(d)
    assert d.calls == ['connect_oi']


def test_docking_is_never_interrupted():
    d = _stub(OI_MODE_PASSIVE, docking=True)
    _keep_awake_fn()(d)
    assert d.calls == []


def test_brc_pulse_is_not_the_thing_keeping_it_awake():
    # It is a no-op on this chassis, so it must not be reached before the mode
    # check — otherwise a Passive robot waits up to a minute to be rescued.
    d = _stub(OI_MODE_PASSIVE, brc_age=120.0)
    _keep_awake_fn()(d)
    assert 'brc_pulse' not in d.calls
    assert 'enter_drive_mode' in d.calls


def test_idle_does_not_enter_passive():
    src = open(DRIVER).read()
    idle = re.search(r'^    def _check_idle\(self\).*?(?=^    def )', src,
                     re.S | re.M).group(0)
    assert '.passive(' not in idle and 'start()' not in idle, \
        '_check_idle must stop the wheels without demoting the OI'
