"""The slip map must ignore AMCL's initial convergence (slip_monitor_node.py).

Found on the first live run: eleven seconds after boot the map recorded
11.7 cm / 3.7° at the spot the robot was standing. That is the filter finding
itself, not the floor stealing anything — and left in, every startup would mark
wherever the robot was switched on, which is usually the dock. After fifty
boots the dock would be the worst place in the house.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_slip_monitor_settle.py -q
"""
import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'slip_monitor_node.py').read_text()


def _method(name):
    m = re.search(rf'\n    def {name}\(.*?(?=\n    def |\n\ndef )', _SRC, re.S)
    assert m, f'{name} not found'
    return m.group(0)


def test_there_is_a_settle_window():
    assert "declare_parameter('slip_map_settle_seconds'" in _SRC
    assert '_settle_until' in _SRC


def test_nothing_is_recorded_during_it():
    body = _method('_sample_correction')
    guard = body.index('self._settle_until')
    record = body.index('self._slipmap.add(')
    assert guard < record, \
        'corrections during startup would mark wherever the robot booted'
    # And the guard has to actually leave the method, not just note the time.
    after_guard = body[guard:record]
    assert re.search(r'\n\s+return\b', after_guard), 'the guard never returns'


def test_the_reference_is_still_tracked_while_settling():
    """‼️ The transform must be READ during the window even though nothing is
    recorded. Skipping it entirely would leave _corr_prev stale, so the first
    sample after the window would measure the whole convergence in one go and
    attribute it to wherever the robot was standing by then — the same bug,
    thirty seconds later."""
    body = _method('_sample_correction')
    guard = body.index('self._settle_until')
    # Between the guard and the guard's own return, the reference is updated.
    tail = body[guard:]
    ret = re.search(r'\n\s+return\b', tail).start()
    assert '_corr_prev' in tail[:ret], \
        'the reference is not kept current while settling'


def test_the_default_is_long_enough_to_cover_convergence():
    m = re.search(r"declare_parameter\('slip_map_settle_seconds', ([\d.]+)\)", _SRC)
    assert m and float(m.group(1)) >= 20.0, \
        'the observed convergence landed 11 s in; a shorter window would keep it'
