"""Tests for "✕ Ακύρωση στόχου" and the stuck commentary.

2026-08-04, both from the owner in the same session:

  * "πατάω ακύρωση στόχου και δεν ακυρώνεται" — the button published ONE empty
    Twist on /cmd_vel_safe and nothing else. bt_navigator still owned the goal
    and wrote a fresh command over it milliseconds later, so the robot paused
    for a frame and carried on. Nothing was listening to /mission/cancel at
    all — not recovery_manager, and not the `cancel` gesture's own publisher.

  * "λέει όλη την ώρα κόλλησα προσπαθώ να ξεκολλήσω" — eight announcements in
    fifteen minutes from one pinch point, in «Κόλλησα»/«Ξεκόλλησα» pairs. The
    commentary tells a person in the room nothing they cannot see.

‼️ And the two are connected: recovery_manager re-issues the goal it reads off
/plan after every escape, which is right for a multi-doorway journey and wrong
the moment a human cancels. A cancel that DID land came back within 30 s.

Static checks over both sources — the wiring is the part that was missing.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_cancel_goal.py -q
"""
import os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DASH = open(f'{PKG}/home_robot/nodes/web_dashboard_node.py').read()
_RECOV = open(f'{PKG}/home_robot/nodes/recovery_manager_node.py').read()


def _fn(src, name):
    """Body of a method, up to the next one."""
    return src.split(f'def {name}')[1].split('\n    def ')[0]


# ── the button has to reach the goal, not the wheels ─────────────────────────

def test_the_button_sends_cancel_not_stop():
    """‼️ THE BUG. 'stop' is one zero Twist; the navigator overwrites it."""
    line = [l for l in _DASH.splitlines() if "b-xnav" in l and 'addEventListener' in l]
    assert line, 'the cancel button lost its handler'
    assert "type:'cancel_nav'" in line[0]
    assert "type:'stop'" not in line[0]


def test_the_server_cancels_the_navigation_goal():
    body = _fn(_DASH, '_cancel_navigation')
    assert 'CancelGoal.Request()' in body, 'the NavigateToPose goal is never cancelled'
    assert '_nav_cancel_cli' in body


def test_it_cancels_every_owner_that_can_drive():
    """Five things can put the robot in motion and each has its own off switch.
    Cancelling only the navigator leaves a patrol or an explore still going."""
    body = _fn(_DASH, '_cancel_navigation')
    for pub in ('_mission_pub', '_patrol_pub', '_explore_pub', '_follow_pub'):
        assert pub in body, f'{pub} is not cancelled'


def test_the_zero_twist_goes_last():
    """Sent before the owners are told, it is simply overwritten on their next
    tick — which is exactly what the old button did."""
    body = _fn(_DASH, '_cancel_navigation')
    assert body.index('_vel_pub.publish') > body.index('_patrol_pub.publish')


def test_a_missing_cancel_service_does_not_skip_the_rest():
    """If bt_navigator is down there is still a patrol to stop."""
    body = _fn(_DASH, '_cancel_navigation')
    assert 'service_is_ready' in body
    assert body.index('_mission_pub.publish') > body.index('service_is_ready'), \
        'the other cancels are inside the service branch'


# ── recovery must lose to a human cancel ─────────────────────────────────────

def test_recovery_listens_for_the_cancel():
    """‼️ Nothing subscribed to /mission/cancel before this — the gesture
    binding published into the void."""
    assert "'/mission/cancel'" in _RECOV
    assert '_on_cancel' in _RECOV


def test_a_cancelled_goal_is_forgotten():
    body = _fn(_RECOV, '_on_cancel')
    assert '_last_goal = None' in body, 'the goal survives the cancel and comes back'
    assert '_reissue_count = 0' in body


def test_a_cancel_stops_the_behavior_already_running():
    """BackUp alone is ~10 s of reversing after the person said stop."""
    body = _fn(_RECOV, '_on_cancel')
    assert 'cancel_goal_async' in body
    assert '_active_behavior' in body


def test_recovery_checks_the_flag_between_steps():
    """The four-step sequence is ~20 s of shuffling; checking only at the top
    would run the whole thing out."""
    body = _fn(_RECOV, '_run_recovery')
    assert body.count('self._cancelled') >= 2


def test_a_cancelled_recovery_does_not_reissue():
    body = _fn(_RECOV, '_recovered')
    assert 'if not self._cancelled' in body


def test_a_fresh_stuck_clears_the_flag():
    """Otherwise one cancel disables recovery for the rest of the session —
    the same trap the FAILED status had.

    2026-08-11: both stuck-detection paths (_check_stuck, cmd_vel active but
    no displacement; _on_nav_feedback, planner never even produces a cmd_vel)
    now share one trigger, _declare_stuck — check the flag gets cleared THERE
    and that both entry points actually call it, rather than duplicating the
    reset in each.
    """
    assert '_cancelled = False' in _fn(_RECOV, '_declare_stuck')
    assert '_declare_stuck(' in _fn(_RECOV, '_check_stuck')
    assert '_declare_stuck(' in _fn(_RECOV, '_on_nav_feedback')


# ── the running commentary is off by default ─────────────────────────────────

def test_the_stuck_commentary_is_silent_by_default():
    assert "declare_parameter('announce_progress', False)" in _RECOV


def test_both_halves_of_the_pair_are_marked_progress():
    """«Κόλλησα» and «Ξεκόλλησα» always came as a pair; silencing one would
    leave the robot cheering about an escape it never announced."""
    for line in ('Κόλλησα, προσπαθώ να ξεκολλήσω.', 'Ξεκόλλησα!'):
        call = [l for l in _RECOV.splitlines() if line in l and '_speak(' in l]
        assert call, f'{line} is no longer spoken through _speak'
        assert 'progress=True' in call[0], f'{line} is not marked as commentary'


def test_the_give_up_line_is_still_spoken():
    """"Δεν μπορώ να ξεκολλήσω" is said once, and it is the one the person
    actually needs: the robot is not getting out of this by itself."""
    call = [l for l in _RECOV.splitlines()
            if 'Δεν μπορώ να ξεκολλήσω' in l and '_speak(' in l]
    assert call
    assert 'progress=True' not in call[0]


def test_silenced_lines_still_reach_the_log():
    """Silent is not invisible — the dashboard's Log tab is where this goes."""
    body = _fn(_RECOV, '_speak')
    assert 'silent' in body.lower()
    assert 'get_logger' in body
