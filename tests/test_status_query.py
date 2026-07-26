"""Tests for the status-question gate (see home_robot/status_query.py).

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_status_query.py -q
"""
import os

import pytest

from home_robot.status_query import (format_status, is_status_query,
                                     wants_battery)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = f'{PKG}/home_robot/nodes'


@pytest.mark.parametrize('utterance', [
    'Πόση μπαταρία έχεις;',
    'πόση μπαταρία έχεις',          # as the STT actually delivers it
    'Πόση μπαταρία σου έμεινε;',
    'Τι μπαταρία έχεις;',
    'Πες μου την κατάστασή σου.',
    'Τι θερμοκρασία έχει η CPU;',
    'Πόση μνήμη χρησιμοποιείς;',
])
def test_status_questions_are_gated(utterance):
    assert is_status_query(utterance)


@pytest.mark.parametrize('utterance', [
    'Πήγαινε να φορτίσεις.',        # dock command — must reach the LLM
    'Πήγαινε στη βάση.',
    'Πήγαινε στην κουζίνα.',
    'Τακτοποίησε το σαλόνι.',
    'Ακολούθησέ με.',
    'Σταμάτα!',
    'Τι ώρα είναι;',
    'Γεια σου Μαξ.',
])
def test_commands_and_chat_are_not_gated(utterance):
    assert not is_status_query(utterance)


def test_dock_command_mentioning_charging_is_not_a_status_query():
    # 'φορτισ' deliberately absent from the status vocabulary: this sentence
    # is an instruction, and gating it would break docking by voice.
    assert not is_status_query('Πήγαινε να φορτίσεις τώρα.')


@pytest.mark.parametrize('utterance,expected', [
    ('Πόση μπαταρία έχεις;', True),
    ('Πες μου την κατάστασή σου.', False),
    ('Τι θερμοκρασία έχει η CPU;', False),
])
def test_battery_specific_detection(utterance, expected):
    assert wants_battery(utterance) is expected


# ── formatting: never invent a number ─────────────────────────────────

def test_battery_reported_when_known():
    assert format_status({'battery_percent': 87.0}, battery_only=True) == 'Έχω 87% μπαταρία.'


def test_battery_charging_is_mentioned():
    out = format_status({'battery_percent': 42.0, 'battery_charging': True},
                        battery_only=True)
    assert 'φορτίζω' in out and '42' in out


def test_missing_battery_says_so_instead_of_guessing():
    out = format_status({'cpu_percent': 10.0}, battery_only=True)
    assert 'CLEAN' in out
    assert not any(c.isdigit() for c in out.replace('CLEAN', ''))


def test_general_status_lists_what_it_has():
    out = format_status({'battery_percent': 50.0, 'cpu_percent': 12.0,
                         'cpu_temp_c': 55.0, 'ram_percent': 71.0},
                        battery_only=False)
    for token in ('50', '12', '55', '71'):
        assert token in out


def test_general_status_flags_absent_battery():
    out = format_status({'cpu_percent': 12.0}, battery_only=False)
    assert 'μπαταρίας δεν έχω' in out


def test_empty_telemetry_is_honest():
    assert 'Δεν μπορώ' in format_status({}, battery_only=False)


# ── wiring contract ───────────────────────────────────────────────────

def test_bridge_gates_status_before_the_llm():
    src = open(f'{NODES}/llm_bridge_node.py').read()
    assert 'from home_robot.status_query import' in src
    gate = src.index('is_status_query(text)')
    llm = src.index('def _handle_text_inner')
    assert gate < llm, 'status must be answered before the LLM path'


def test_status_answer_uses_the_real_tool():
    src = open(f'{NODES}/llm_bridge_node.py').read()
    body = src[src.index('def _answer_status'):]
    body = body[:body.index('\n    def ', 1)]
    assert "_dispatch_tool('system_status'" in body, \
        'must read real telemetry, not compose a number'
    assert 'self._busy.release()' in body, 'must not leak the busy lock'
