"""Tests for the slip map core (see home_robot/slip_map.py).

The point of this feature is telling "one bad rug" apart from "bad calibration
everywhere", so the tests are mostly about accumulation landing in the right
cell, the decay being a view rather than a mutation, and the file refusing to
load across a remap.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_slip_map.py -q
"""
import json
import math

from home_robot.slip_map import (
    SlipMap, correction_delta, is_significant,
)

DAY = 86400.0


# ── what counts as a correction ──────────────────────────────────────────

def test_no_previous_sample_is_no_correction():
    assert correction_delta(None, (1.0, 2.0, 0.5)) == (0.0, 0.0)


def test_only_the_change_counts():
    """‼️ The absolute map->odom also carries wherever the odom origin was at
    boot — it read 366 cm on a perfectly localizing robot. Only the delta is
    the correction."""
    shift, turn = correction_delta((3.66, 0.0, 2.1), (3.67, 0.0, 2.1))
    assert round(shift, 3) == 0.01


def test_turn_takes_the_short_way_round():
    # Across ±180 a naive subtraction reads 359° and every pass through that
    # heading would light up the map.
    _, turn = correction_delta((0, 0, math.pi - 0.01), (0, 0, -math.pi + 0.01))
    assert turn < 0.05


def test_jitter_is_ignored_and_a_real_shift_is_not():
    assert not is_significant(0.005, 0.0)
    assert is_significant(0.05, 0.0)


def test_a_pure_rotation_counts_too():
    assert is_significant(0.0, math.radians(5))


# ── accumulation ─────────────────────────────────────────────────────────

def test_nearby_events_land_in_one_cell():
    """Corrections land wherever AMCL ran its update, not on the rug itself,
    so centimetre precision would scatter one rug across a hundred points."""
    m = SlipMap(cell=0.5)
    a = m.add(1.10, 2.10, 0.2, 0.0)
    b = m.add(1.35, 2.40, 0.2, 0.0)
    assert a == b
    assert len(m.cells) == 1
    assert m.cells[a]['n'] == 2


def test_distant_events_do_not():
    m = SlipMap(cell=0.5)
    m.add(1.0, 1.0, 0.2, 0.0)
    m.add(5.0, 5.0, 0.2, 0.0)
    assert len(m.cells) == 2


def test_the_worst_cell_sorts_first():
    m = SlipMap(cell=0.5)
    m.add(0.0, 0.0, 0.05, 0.0)
    for _ in range(5):
        m.add(3.0, 3.0, 0.20, 0.0)
    worst = m.decayed()[0]
    assert round(worst['x'], 2) == 3.25 and round(worst['y'], 2) == 3.25
    assert worst['n'] == 5


def test_cell_centres_are_reported_not_corners():
    m = SlipMap(cell=0.5)
    m.add(0.1, 0.1, 0.1, 0.0)
    c = m.decayed()[0]
    assert (round(c['x'], 3), round(c['y'], 3)) == (0.25, 0.25)


def test_negative_coordinates_do_not_fold_onto_positive_ones():
    # int() truncates toward zero; floor is what keeps -0.2 out of the +0 cell.
    m = SlipMap(cell=0.5)
    assert m.key(-0.2, -0.2) != m.key(0.2, 0.2)


# ── decay ────────────────────────────────────────────────────────────────

def test_old_events_fade():
    m = SlipMap(cell=0.5, half_life_days=14.0)
    now = 1_000_000.0
    m.add(0.0, 0.0, 1.0, 0.0, now=now - 14 * DAY)
    faded = m.decayed(now=now)[0]
    assert abs(faded['m'] - 0.5) < 1e-6


def test_decay_does_not_change_the_stored_total():
    """‼️ The decay is a VIEW. Folding it back in would make the totals depend
    on how often anyone happened to look at the map."""
    m = SlipMap(cell=0.5, half_life_days=14.0)
    now = 1_000_000.0
    m.add(0.0, 0.0, 1.0, 0.0, now=now - 28 * DAY)
    for _ in range(5):
        m.decayed(now=now)
    assert m.cells[(0, 0)]['m'] == 1.0


def test_prune_drops_what_has_faded_to_nothing():
    m = SlipMap(cell=0.5, half_life_days=1.0)
    now = 1_000_000.0
    m.add(0.0, 0.0, 0.5, 0.0, now=now - 30 * DAY)     # long gone
    m.add(9.0, 9.0, 0.5, 0.0, now=now)               # fresh
    assert m.prune(now=now) == 1
    assert len(m.cells) == 1


# ── the file ─────────────────────────────────────────────────────────────

def test_it_survives_a_round_trip(tmp_path):
    path = str(tmp_path / 'slip.json')
    a = SlipMap(path=path, map_name='malou2', cell=0.5)
    a.add(1.0, 2.0, 0.3, 0.1)
    assert a.save()

    b = SlipMap(path=path, map_name='malou2')
    assert b.load()
    assert b.cells == a.cells
    assert b.cell == 0.5


def test_a_slip_map_from_another_map_is_refused(tmp_path):
    """‼️ Coordinates only mean anything in the frame they were recorded in.
    Loading last house's cells would paint its rug onto this one's floor."""
    path = str(tmp_path / 'slip.json')
    a = SlipMap(path=path, map_name='kela3')
    a.add(1.0, 2.0, 0.3, 0.0)
    a.save()

    b = SlipMap(path=path, map_name='malou2')
    assert b.load() is False
    assert b.cells == {}


def test_a_corrupt_file_is_not_fatal(tmp_path):
    path = tmp_path / 'slip.json'
    path.write_text('{ this is not json')
    m = SlipMap(path=str(path), map_name='malou2')
    assert m.load() is False
    assert m.cells == {}


def test_missing_file_is_not_an_error(tmp_path):
    m = SlipMap(path=str(tmp_path / 'nope.json'), map_name='malou2')
    assert m.load() is False


def test_the_save_is_atomic(tmp_path):
    """A half-written file read at the next boot would throw away weeks of
    accumulation for a power cut during a 200-byte write."""
    path = tmp_path / 'slip.json'
    m = SlipMap(path=str(path), map_name='malou2')
    m.add(1.0, 1.0, 0.2, 0.0)
    m.save()
    # Whatever is on disk must be complete, parseable JSON.
    data = json.loads(path.read_text())
    assert data['map'] == 'malou2' and len(data['cells']) == 1
    # And no temp files left lying around.
    assert [p.name for p in tmp_path.iterdir()] == ['slip.json']
