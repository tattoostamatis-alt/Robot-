"""Tests for the hard-coded spoken stop gate (see home_robot/stop_command.py).

Two layers, both robot-free:

  * Unit tests of is_stop_command over the exact phrasings measured against
    the NPU LLM on 2026-07-26 — including the ones it answered with a false
    "Σταμάτησα." while still driving.
  * Static wiring checks that llm_bridge_node consults the gate BEFORE the
    busy lock and before the LLM call, which is the whole point: during a
    blocking goto the lock is held for up to nav_timeout, so a stop routed
    the normal way would be dropped exactly when it is needed.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_stop_command.py -q
"""
import os

import pytest

from home_robot.stop_command import is_stop_command, strip_accents

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = f'{PKG}/home_robot/nodes'


# ── phrasings the LLM got RIGHT (must keep working) ───────────────────

@pytest.mark.parametrize('utterance', [
    'Σταμάτα.',
    'Σταμάτα αμέσως!',
    'Σταμάτησε.',
    'Έι ρομπότ, σταμάτα τώρα.',
])
def test_phrasings_the_llm_already_handled(utterance):
    assert is_stop_command(utterance)


# ── phrasings the LLM got WRONG (the reason this module exists) ───────

@pytest.mark.parametrize('utterance', [
    'Σταμάτα!',                 # exclamation mark alone broke the tool call
    'Μαξ, σταμάτα!',
    'Μαξ σταμάτα',
    'Στοπ!',
    'Σταμάτα σε παρακαλώ.',
    'Σταμάτα να κινείσαι.',
    'Λοιπόν, σταμάτα τώρα.',
    'Έι ρομπότ, σταμάτα.',
    'Ακίνητος!',
])
def test_phrasings_the_llm_answered_with_a_false_confirmation(utterance):
    assert is_stop_command(utterance), \
        f'{utterance!r} must be caught here — the LLM says "Σταμάτησα." and keeps driving'


# ── robustness ────────────────────────────────────────────────────────

@pytest.mark.parametrize('utterance', [
    'ΣΤΑΜΑΤΑ',          # STT sometimes upper-cases
    'σταματα',          # ...and sometimes drops accents
    'Σταματήστε!',      # plural/polite
    'Αλτ!',
    'Σταμάτα την αποστολή.',
])
def test_case_accent_and_inflection_robustness(utterance):
    assert is_stop_command(utterance)


def test_strip_accents_normalizes_greek():
    assert strip_accents('ΣΤΑΜΆΤΑ') == 'σταματα'


# ── must NOT fire: ordinary conversation and other commands ───────────

@pytest.mark.parametrize('utterance', [
    'Πήγαινε στην κουζίνα.',
    'Τακτοποίησε το σαλόνι.',
    'Ακολούθησέ με.',
    'Πόση μπαταρία έχεις;',
    'Πήγαινε να φορτίσεις.',
    'Γεια σου Μαξ.',
    'Έι ρομπότ',
    'Τι καιρό κάνει;',
])
def test_other_commands_still_reach_the_llm(utterance):
    assert not is_stop_command(utterance)


@pytest.mark.parametrize('utterance', [
    'Μη σταματάς.',
    'Μην σταματήσεις!',
    'Γιατί σταμάτησες;',
    'Θα σταματήσεις μετά.',
    'Αν σταματήσεις πες μου.',
    'Όταν σταματήσεις, γύρνα.',
])
def test_negated_and_hypothetical_mentions_are_not_commands(utterance):
    assert not is_stop_command(utterance)


# ── wiring contract with llm_bridge_node ──────────────────────────────

def test_bridge_checks_stop_before_busy_lock_and_llm():
    src = open(f'{NODES}/llm_bridge_node.py').read()
    assert 'from home_robot.stop_command import' in src, \
        'llm_bridge_node must use the shared stop gate, not its own copy'
    gate = src.index('is_stop_command(text)')
    lock = src.index('self._busy.acquire')
    assert gate < lock, \
        'stop must be handled BEFORE the busy lock — otherwise a blocking ' \
        'goto swallows it for up to nav_timeout seconds'


def test_emergency_stop_halts_wheels_nav_and_behaviours():
    src = open(f'{NODES}/llm_bridge_node.py').read()
    body = src[src.index('def _emergency_stop'):]
    body = body[:body.index('\n    def ', 1)]
    assert 'cmd_vel_pub.publish(Twist())' in body, 'must zero the wheels'
    assert 'cancel_goal_async' in body, \
        'must cancel the Nav2 goal, else the controller overwrites the zero Twist'
    for pub in ('patrol_pub', 'explore_pub', 'follow_pub', 'mission_pub'):
        assert pub in body, f'must also cancel {pub} — those nodes ignore cmd_vel'


def test_nav_goal_handle_is_reachable_for_cancellation():
    src = open(f'{NODES}/llm_bridge_node.py').read()
    assert 'self._nav_goal_handle = gh' in src, \
        '_navigate must publish the goal handle on self so stop can cancel it'
    assert 'self._nav_goal_handle = None' in src, \
        'the handle must be cleared when navigation finishes'
