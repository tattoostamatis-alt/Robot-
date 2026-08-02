"""Tests for the dashboard's IMU panel (web_dashboard_node).

Added 2026-08-02 with the panel itself. The BNO085 on this robot is surrounded
by things that are easy to display dishonestly, and each of these guards one:

  - It is a GAME rotation vector: gyro+accel fusion with the magnetometer left
    out on purpose (bno085_imu.ino explains why — indoors the Roomba's DC motors
    corrupted the absolute heading until AMCL could not stay converged). So the
    heading is RELATIVE, with an arbitrary zero on every boot. Confirmed live:
    yaw read -32.7° and then 179.6° across an ESP32 reboot with the robot never
    having moved. A rose labelled N/S/E/W would be a confident lie.

  - Acceleration is not streamed at all. SH2_LINEAR_ACCELERATION is never
    enabled, so ax/ay/az arrive as literal 0.0. Printing "0.00 / 0.00 / 0.00"
    would read as a robot sitting perfectly still rather than as no data.

  - A dead BNO085 is SILENT — it stops publishing rather than reporting an
    error, so a purely callback-driven panel freezes on the last good frame and
    looks healthy.

  - The gyro-dead alarm must be qualified by motion. The first version counted
    exact zeros unconditionally and fired "ΓΥΡΟΣΚΟΠΙΟ ΝΕΚΡΟ" at a robot parked
    on the carpet: the BNO085 quantises small rates straight to 0 at rest.
    Caught on the live stream, not in review.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_imu.py -q
"""
import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()


def _fn(name):
    m = re.search(r'\n    def %s\(self.*?(?=\n    def )' % re.escape(name), _SRC, re.S)
    assert m, f'{name}() is gone'
    return m.group(0)


# ── the panel must not invent a compass ─────────────────────────────────────

def test_the_heading_is_never_labelled_with_cardinal_points():
    """No magnetometer is in the fix, so N/E/S/W would be fabricated."""
    rose = re.search(r'function drawRose\(.*?\n\}', _SRC, re.S)
    assert rose, 'drawRose() is gone'
    # The tick labels are drawn here; degrees are honest, letters are not.
    labels = re.findall(r"fillText\('([^']*)'", rose.group(0))
    for lbl in labels:
        assert not re.fullmatch(r'[NSEWΒΝΑΔ]', lbl.strip()), \
            f'cardinal label {lbl!r} on a magnetometer-free heading'


def test_the_pane_says_the_compass_is_relative():
    pane = re.search(r'<section class="pane" id="p-imu">(.*?)</section>', _SRC, re.S)
    assert pane, 'the IMU pane is gone'
    body = pane.group(1)
    assert 'ΣΧΕΤΙΚΗ' in body, 'the pane no longer says the heading is relative'
    assert 'GAME_ROTATION_VECTOR' in body, \
        'the pane should name the report it actually shows'


# ── acceleration is absent, not zero ────────────────────────────────────────

def test_absent_acceleration_is_not_rendered_as_zeroes():
    m = re.search(r"\$\('i-acc'\)\.textContent\s*=(.*?);\n", _SRC, re.S)
    assert m, 'the acceleration row is gone'
    expr = m.group(1)
    assert 'm.ax===0 && m.ay===0 && m.az===0' in expr.replace('  ', ' '), \
        'all-zero acceleration is not special-cased, so "no data" reads as "at rest"'
    assert 'ανενεργό report' in expr, 'the absent-report wording is gone'


# ── a silent sensor has to be reported as silent ────────────────────────────

def test_the_dead_imu_alarm_is_driven_by_a_timer():
    """Callback-driven only, a dead IMU freezes the panel on its last good frame."""
    assert re.search(r'create_timer\([\d.]+,\s*self\._imu_health\)', _SRC), \
        'no health timer — a silent BNO085 would leave the panel looking fine'
    body = _fn('_imu_health')
    assert "'alive': False" in body, 'the health timer never reports the sensor dead'


def test_the_frontend_blanks_the_readings_when_the_sensor_dies():
    """Leaving the last good numbers on screen next to a DEAD badge invites
    reading them as current."""
    m = re.search(r'function onImu\(m\)\{(.*?)\n\}', _SRC, re.S)
    assert m, 'onImu() is gone'
    dead_branch = m.group(1).split('return;')[0]
    assert "'i-yaw'" in dead_branch and "textContent='—'" in dead_branch.replace(' ', ''), \
        'the readings are not cleared when the IMU goes silent'


# ── the false alarm found on the live stream ────────────────────────────────

def test_the_gyro_alarm_only_counts_while_the_robot_turns():
    body = _fn('_cb_imu')
    zero_block = re.search(r'if g\.x == 0\.0.*?else:', body, re.S)
    assert zero_block, 'the zero-gyro detector is gone'
    assert 'self._turning' in zero_block.group(0), (
        'zero gyro samples are counted at rest too — this fired '
        '"gyro dead" at a stationary robot, because the BNO085 quantises '
        'small rates to exactly 0')


def test_turning_is_decided_well_above_the_rotation_floor():
    """Below the 879's ~0.31 rad/s floor the wheels do not turn, so a threshold
    under it would call a standing robot 'turning' on odometry noise."""
    body = _fn('_cb_odom')
    m = re.search(r'abs\(wz\)\s*>\s*([\d.]+)', body)
    assert m, '_cb_odom no longer derives the turning flag'
    assert 0.05 <= float(m.group(1)) <= 0.31, \
        f'threshold {m.group(1)} is not a sane fraction of the rotation floor'


# ── bandwidth: this streams to a phone ──────────────────────────────────────

def test_the_imu_stream_is_throttled():
    body = _fn('_cb_imu')
    assert re.search(r'now - self\._imu_last < [\d.]+', body), \
        'the IMU broadcast is unthrottled — it runs ~19 Hz to every open phone'


def test_the_rate_counter_sees_every_sample_not_just_broadcast_ones():
    """The Hz readout is a health number; counting only post-throttle samples
    would peg it at the throttle rate and hide a degraded sensor."""
    body = _fn('_cb_imu')
    inc = body.index('self._imu_n += 1')
    throttle = body.index('self._imu_last = now')
    assert inc < throttle, 'the sample counter sits behind the throttle'
