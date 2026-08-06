"""How far `pick` drives to close the last metre — llm_bridge._approach_gap.

An object on the floor in front of the robot sits ~1 m from the arm base, and
the arm reaches 0.45 m, so "πιάσε την παντόφλα" refused essentially every time
until pick learned to take the step itself. This is the arithmetic that decides
whether to step, and how far.

_approach_gap is exercised unbound so the decision stays testable without
rclpy, wheels or a camera.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MAX = 1.5        # pick_approach_max — beyond this it is a `fetch`
MARGIN = 0.7     # stop at this fraction of the reach envelope
STEP = 0.25      # pick_approach_step — most one nudge may cover


def _bridge(max_m=MAX, margin=MARGIN, step=STEP):
    return types.SimpleNamespace(pick_approach_max=max_m,
                                 pick_approach_margin=margin,
                                 pick_approach_step=step)


def _gap(result, **kw):
    from home_robot.nodes.llm_bridge_node import LLMBridgeNode
    return LLMBridgeNode._approach_gap(_bridge(**kw), result)


def _refusal(distance, reach=0.45):
    return {'status': 'out_of_reach', 'distance': distance, 'max_reach': reach}


# ── the reported case ─────────────────────────────────────────────────────────

def test_a_slipper_a_metre_away_is_approached_one_step_at_a_time():
    """The live measurement: slipper on the floor at 0.99 m, arm reaches 0.45.
    The whole gap is 0.675 m, but it is covered in capped nudges — the caller
    re-measures between them."""
    assert _gap(_refusal(0.99)) == STEP


def test_no_single_nudge_can_overshoot():
    """‼️ The camera looks forward, so a floor object leaves the frame as the
    robot closes in. Overshooting does not mean "a bit too far" — it means the
    target is gone. Live 2026-08-06: one 0.45 m nudge drove past the slipper
    and the next look saw only floor."""
    for distance in (0.5, 0.6, 0.8, 1.0, 1.4):
        assert _gap(_refusal(distance)) <= STEP, distance


def test_the_last_step_stops_short_of_the_reach_limit():
    """Once the remaining gap is under one step, the nudge is sized to leave
    the object inside the arm's envelope rather than under the robot."""
    remaining = 0.6 - _gap(_refusal(0.6))
    assert 0 < remaining < 0.45


# ── when NOT to step ──────────────────────────────────────────────────────────

def test_too_far_is_not_a_nudge():
    """Past pick_approach_max this is a navigation problem — obstacles, doors,
    re-looking. Driving blindly at something metres away is what `fetch` exists
    to avoid doing."""
    assert _gap(_refusal(1.6)) is None
    assert _gap(_refusal(4.0)) is None


def test_the_boundary_is_inclusive():
    assert _gap(_refusal(MAX)) is not None
    assert _gap(_refusal(MAX + 0.01)) is None


def test_a_result_without_a_distance_cannot_be_approached():
    """Never invent a drive distance: an out_of_reach with no measurement is a
    bug upstream, and guessing sends the robot forward by an arbitrary amount."""
    assert _gap({'status': 'out_of_reach'}) is None
    assert _gap({'status': 'out_of_reach', 'distance': None}) is None
    assert _gap({'status': 'out_of_reach', 'distance': 'κοντά'}) is None


def test_a_missing_reach_falls_back_to_the_documented_default():
    """max_reach is pick_place_node's parameter; if an older node omits it the
    step must still be sane rather than the whole distance."""
    gap = _gap({'status': 'out_of_reach', 'distance': 0.5})
    assert abs(gap - (0.5 - 0.45 * MARGIN)) < 1e-6


def test_something_already_within_reach_needs_no_drive():
    """Defensive: _approach_gap should never be reached for these, but if it
    is, the answer is "do not move", not a negative distance."""
    assert _gap(_refusal(0.2)) == 0.0
