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


# ── formatting: the battery reading is never quoted ───────────────────
# The pack was removed 2026-07-26; the robot runs off a power bank. The OI
# still reports a figure and it is nonsense (41% at 14.44 V, a full-pack
# voltage), so it must never reach the user.

def test_battery_question_says_there_is_no_battery():
    out = format_status({'battery_percent': 87.0}, battery_only=True)
    assert 'powerbank' in out
    assert '87' not in out


def test_battery_figure_never_leaks_even_when_present():
    out = format_status({'battery_percent': 41.3, 'battery_charging': True,
                         'cpu_percent': 12.0}, battery_only=False)
    assert '41' not in out
    assert 'μπαταρ' not in out.lower()


def test_general_status_lists_machine_telemetry():
    out = format_status({'battery_percent': 50.0, 'cpu_percent': 12.0,
                         'cpu_temp_c': 55.0, 'ram_percent': 71.0},
                        battery_only=False)
    for token in ('12', '55', '71'):
        assert token in out
    assert '50' not in out          # the battery figure stays out


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
