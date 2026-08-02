"""Tests for how roomba_driver reports a base that stops answering.

2026-08-02: the dashboard's log tab was unusable. 256 messages, every one of
them a roomba_driver retry, pushing every other warning off the page — and not
one of them said what was wrong or what to do. Two pollers, three retries each,
at 20 Hz: three WARNs and one ERROR per poll, ~80 log lines a second, all
identical.

The shape of the failure was the diagnosis nobody printed: EVERY read returning
zero bytes is the signature of a sleeping Roomba, not a flaky cable. And on this
robot the driver cannot fix it — the BRC wake line is dead (all 10
DTR/RTS/polarity/baud combinations swept 2026-07-31), so only a power cycle
brings the base back. That is what the log says now, once, then every 30 s.

Measured after the change: 4 lines in 45 s on the same sleeping base.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_roomba_link_logging.py -q
"""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1]
       / 'home_robot' / 'nodes' / 'roomba_driver.py').read_text()


def _method(name):
    m = re.search(r'\n    def %s\(self.*?(?=\n    (?:def |[A-Z_]+ ?=))' % name, SRC, re.S)
    assert m, f'{name}() is gone or changed shape'
    return m.group(0)


# ── the retry loops no longer log per attempt ───────────────────────────────

def test_the_retry_loops_do_not_log_each_attempt():
    """Three WARNs per poll at 20 Hz is the flood this replaces."""
    for fn in ('_safe_get_sensors', '_safe_get_encoders'):
        body = _method(fn)
        assert 'attempt {' not in body and 'attempt %' not in body, \
            f'{fn} logs every retry again'
        assert not re.search(r'get_logger\(\)\.warn', body), \
            f'{fn} warns inside the retry loop again'


def test_both_pollers_report_through_one_place():
    for fn in ('_safe_get_sensors', '_safe_get_encoders'):
        assert '_note_link_failure(' in _method(fn), \
            f'{fn} no longer funnels its failure through the throttled reporter'


def test_both_pollers_clear_the_outage():
    for fn in ('_safe_get_sensors', '_safe_get_encoders'):
        assert '_note_link_ok()' in _method(fn), \
            f'{fn} never marks the link healthy, so recovery is silent'


# ── the report is throttled ─────────────────────────────────────────────────

def test_repeat_reports_are_throttled():
    body = _method('_note_link_failure')
    assert 'LINK_REPORT_PERIOD' in body, 'the repeat is no longer rate-limited'
    m = re.search(r'LINK_REPORT_PERIOD = ([\d.]+)', SRC)
    assert m, 'LINK_REPORT_PERIOD is gone'
    assert float(m.group(1)) >= 10.0, \
        'a short period brings the flood back at 20 Hz polling'


def test_the_first_failure_is_reported_immediately():
    """Waiting 30 s to say anything would look like a hang."""
    body = _method('_note_link_failure')
    assert 'self._link_last_report = 0.0' in body, \
        'the first outage must not be swallowed by the throttle'


def test_the_outage_is_timed():
    body = _method('_note_link_failure')
    assert '_link_down_since' in body and 'secs' in body, \
        'the message should say how long the base has been silent'


# ── it names the cause and the fix ──────────────────────────────────────────

def test_zero_bytes_is_diagnosed_as_sleep():
    body = _method('_note_link_failure')
    assert 'not 9 bytes long, it is: 0' in body, \
        'the encoder zero-length signature is no longer recognised'
    assert 'not 80 bytes long, it is: 0' in body, \
        'the sensor zero-length signature is no longer recognised'


def test_the_message_tells_the_user_to_power_cycle():
    """The only thing that works — the BRC line is dead on this chassis."""
    body = _method('_note_link_failure')
    assert 'power cycle' in body, \
        'the log must name the one action that actually revives the base'


def test_a_different_failure_is_not_called_sleep():
    """A real serial fault must not be reported as a sleeping base."""
    body = _method('_note_link_failure')
    assert 'else:' in body, 'every failure now reports as sleep'


def test_recovery_says_so_once():
    body = _method('_note_link_ok')
    assert 'if self._link_down_since' in body, \
        'the recovery line would print on every successful read'
    assert '_link_down_since = 0.0' in body, 'the outage state is never reset'
