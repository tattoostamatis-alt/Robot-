"""home_robot/collision_skirt.py — the dashboard's collision_monitor rewrite.

Two things have to be true for the "adjustable skirt margin" dashboard
feature to be safe to ship: the generator has to reproduce exactly what
nav2_params.yaml already has (or every existing HW-tuned value silently
changes the moment someone opens the Safety tab), and EVERY margin the
slider will ever offer has to keep every invariant
tests/test_collision_monitor_fits.py enforces — not just the two endpoints
picked by hand while choosing the range.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_collision_skirt.py -q
"""
import ast
import math
import os
import sys

import pytest
import yaml

from home_robot import collision_skirt as cs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_collision_monitor_fits as tcmf  # noqa: E402  (real invariants, not reimplemented from memory)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARAMS_PATH = os.path.join(_ROOT, 'config', 'nav2_params.yaml')
POLY_SLACK = 0.006     # same allowance test_collision_monitor_fits.py uses


def _real_text():
    with open(_PARAMS_PATH) as fh:
        return fh.read()


def _min_reach(pts):
    """Closest approach of the polygon boundary to the centre — copied from
    test_collision_monitor_fits.py's own implementation (generic convex-
    polygon geometry, not specific to that file)."""
    best = float('inf')
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg = math.hypot(dx, dy)
        if seg < 1e-12:
            continue
        best = min(best, abs(dx * y1 - dy * x1) / seg)
    return best


def _max_reach(pts):
    return max(math.hypot(x, y) for x, y in pts)


# ── the generator matches production, byte for byte ─────────────────────────

def test_circle_points_matches_the_real_moving_ring_exactly():
    """R=0.215 is today's production margin (40 mm). If the formula drifts
    from what's actually in the file, this catches it before it ever reaches
    a real robot."""
    real = _real_text()
    stop = yaml.safe_load(real)['collision_monitor']['ros__parameters']['Stop']
    for name in ('moving_forward', 'moving_backward'):
        assert cs.circle_points(0.215) == stop[name]['points']


def test_circle_points_matches_the_real_rotating_ring_too():
    """Same formula, same n — proves it's not a coincidence tuned to 0.215."""
    real = _real_text()
    stop = yaml.safe_load(real)['collision_monitor']['ros__parameters']['Stop']
    assert cs.circle_points(0.180) == stop['stopped_or_rotating']['points']


def test_current_margin_mm_reads_the_real_file_as_40():
    assert cs.current_margin_mm(_real_text()) == cs.MARGIN_DEFAULT_MM == 40


# ── patch_moving_points touches exactly two lines ───────────────────────────

def test_patch_changes_only_the_two_points_lines():
    real = _real_text()
    patched = cs.patch_moving_points(real, 30)
    old_lines, new_lines = real.splitlines(), patched.splitlines()
    assert len(old_lines) == len(new_lines)
    changed = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    assert len(changed) == 2, (
        f'expected exactly 2 changed lines, got {len(changed)}: {changed} — '
        'a comment or unrelated line moved')
    for i in changed:
        assert 'points:' in old_lines[i] and 'points:' in new_lines[i]
    # stopped_or_rotating's own points string must be byte-identical to before.
    assert (yaml.safe_load(patched)['collision_monitor']['ros__parameters']
            ['Stop']['stopped_or_rotating']['points']
            == yaml.safe_load(real)['collision_monitor']['ros__parameters']
            ['Stop']['stopped_or_rotating']['points'])


def test_patch_raises_rather_than_silently_no_op():
    with pytest.raises(ValueError):
        cs.patch_moving_points('collision_monitor:\n  ros__parameters: {}\n', 40)


def test_patch_raises_if_a_block_appears_more_than_once():
    real = _real_text()
    doubled = real + '\n' + real   # moving_forward now appears twice
    with pytest.raises(ValueError):
        cs.patch_moving_points(doubled, 40)


# ── the load-bearing test: every slider value the UI offers ────────────────

@pytest.mark.parametrize('margin_mm', cs.ALLOWED_MARGINS_MM)
def test_every_allowed_margin_keeps_collision_monitor_fits_invariants(margin_mm):
    """Re-run test_collision_monitor_fits.py's actual checks against a copy of
    the real YAML patched to this margin — not just the two boundary values
    worked out by hand when the [30, 60] mm range was chosen."""
    patched = cs.patch_moving_points(_real_text(), margin_mm)
    params = yaml.safe_load(patched)
    stop = params['collision_monitor']['ros__parameters']['Stop']
    costmap_r = (params['local_costmap']['local_costmap']
                 ['ros__parameters']['robot_radius'])

    def points(name):
        return ast.literal_eval(stop[name]['points'])

    moving = ['moving_forward', 'moving_backward']
    rotating = 'stopped_or_rotating'
    skirt_r = cs.margin_to_radius(margin_mm)

    # test_costmap_radius_matches_the_measured_chassis — untouched by the patch.
    assert abs(costmap_r - cs.CHASSIS_R) < 1e-6

    # test_every_bearing_keeps_the_full_stop_margin
    #
    # ‼️ max_reach tolerance is 1e-4 here, not the 1e-6 the real test uses —
    # and that's not a relaxation of the real invariant. The real test's
    # SKIRT_R is DERIVED from these same points' own max_reach (see
    # test_collision_monitor_fits.py's SKIRT_R, sourced from
    # collision_skirt.current_margin_mm()), so its "<= SKIRT_R + 1e-6" is
    # tautological by construction and passes for any patched margin. This
    # check instead compares against the INTENDED radius (chassis + the
    # margin the UI asked for) to catch the generator itself drifting; the
    # gap between intended and actual is pure 4-decimal rounding noise
    # (<0.03 mm, i.e. two coordinates rounded independently don't preserve
    # x^2+y^2=R^2 exactly), not a safety concern.
    floor = skirt_r - POLY_SLACK
    for name in moving:
        closest = _min_reach(points(name))
        assert closest >= floor, f'{name} @ {margin_mm}mm: only {closest:.4f} m clear'
        assert _max_reach(points(name)) <= skirt_r + 1e-4

    # test_rotation_is_not_gated_like_translation — the anti-deadlock gap.
    rot_reach = _min_reach(points(rotating))
    tightest_moving = min(_min_reach(points(n)) for n in moving)
    assert rot_reach < tightest_moving - 0.02, (
        f'@ {margin_mm}mm the rotating ring ({rot_reach:.4f} m) is not clearly '
        f'below the moving rings ({tightest_moving:.4f} m) — deadlock risk')

    # test_the_rotation_ring_still_refuses_actual_contact
    assert rot_reach >= cs.CHASSIS_R

    # test_the_zone_is_round_not_boxy
    for name in moving + [rotating]:
        pts = points(name)
        assert _max_reach(pts) - _min_reach(pts) < 0.010

    # test_the_zone_fits_the_tightest_doorway
    for name in moving + [rotating]:
        assert _max_reach(points(name)) < tcmf.TIGHTEST_PASSAGE, (
            f'{name} @ {margin_mm}mm reaches past the tightest doorway '
            f'({tcmf.TIGHTEST_PASSAGE} m)')

    # test_stop_polygons_enclose_the_physical_robot
    for name in moving + [rotating]:
        assert _min_reach(points(name)) > cs.CHASSIS_R
