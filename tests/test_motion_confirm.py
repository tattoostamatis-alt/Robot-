"""Tests for the ask-before-you-drive gate (home_robot/motion_confirm.py).

2026-08-04, from the owner: "το ρομπότ πάει συνέχεια βόλτες χωρίς να του πω".
The episodic timeline had the whole story in two lines:

    18:59:32  heard (Speaker_5): «Λοιπόν, πάμε στο σαλόνι, πάμε στο διάδρομο.»
    18:59:43  said:              «Πηγαίνω στο σαλόνι και μετά στον διάδρομο.»

Nobody addressed the robot. stt_node records up to 12 s after a wake word, so a
family conversation arrives as one utterance and the model finds an instruction
inside it — quite reasonably, since one is there.

Two layers, both robot-free:

  * Unit tests of the yes/no gate and the spoken question.
  * Static wiring checks that llm_bridge asks BEFORE dispatching, speaks the
    question itself, and suppresses whatever the model says afterwards. ‼️ That
    last one is not belt-and-braces: on the first live run, with the tool
    result carrying an explicit "say exactly «Να πάω στο σαλόνι;»", Gemini
    replied «Πηγαίνω στο σαλόνι και μετά στον διάδρομο» — announcing the trip
    it had just been refused.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_motion_confirm.py -q
"""
import os

import pytest

from home_robot.motion_confirm import (MOTION_TOOLS, PENDING_TTL,
                                       confirm_question, is_affirmative,
                                       is_negative)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRIDGE = open(f'{PKG}/home_robot/nodes/llm_bridge_node.py').read()


# ── which tools have to ask ──────────────────────────────────────────────────

@pytest.mark.parametrize('tool', ['goto', 'goto_pointed', 'dock', 'patrol',
                                  'explore', 'tidy', 'find', 'move'])
def test_tools_that_move_the_robot_must_ask(tool):
    assert tool in MOTION_TOOLS


@pytest.mark.parametrize('tool', ['stop', 'system_status', 'recall',
                                  'distance_to', 'open_app', 'report_clutter'])
def test_tools_that_do_not_move_the_robot_must_not_ask(tool):
    """‼️ `stop` above all. A halt that waits for a second sentence is not a
    halt, and it is the one command the whole stop gate exists to make
    unconditional."""
    assert tool not in MOTION_TOOLS


# ── yes ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('answer', [
    'Ναι.', 'ναι', 'Ναι, πήγαινε.', 'Πήγαινε.', 'Εντάξει.', 'Οκ.', 'ΟΚ',
    'Βεβαίως.', 'Φυσικά.', 'Άντε.', 'Ξεκίνα.', 'Κάνε το.', 'Προχώρα.',
])
def test_affirmative(answer):
    assert is_affirmative(answer)
    assert not is_negative(answer)


# ── no ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('answer', [
    'Όχι.', 'όχι', 'Άσε.', 'Άσ\' το.', 'Μην πας.', 'Ακύρωσε.', 'Σταμάτα.',
    'Αργότερα.', 'Τίποτα.', 'Όχι, άσε.',
])
def test_negative(answer):
    assert is_negative(answer)
    assert not is_affirmative(answer)


# ── neither: the conversation moved on ───────────────────────────────────────

@pytest.mark.parametrize('utterance', [
    'Τι ώρα είναι;',
    'Νερό, νερό θες?',                      # overheard, same afternoon
    'Αυτή είναι άλλη άνθρωπος, τώρα.',      # overheard, same afternoon
    'Πάμε στο σαλόνι.',                     # ‼️ the sentence that started this
])
def test_unrelated_speech_is_neither_yes_nor_no(utterance):
    """The caller drops the pending action on this. ‼️ «πάμε» must NOT read as
    a yes: it is the exact word of the overheard sentence, so accepting it
    would let the robot confirm its own misunderstanding."""
    assert not is_affirmative(utterance)
    assert not is_negative(utterance)


def test_a_no_beats_a_yes_in_the_same_sentence():
    """"Όχι, εντάξει, άσε" contains 'εντάξει'. It is still a no."""
    assert is_negative('Όχι, εντάξει, άσε το.')
    assert not is_affirmative('Όχι, εντάξει, άσε το.')


# ── what it says out loud ────────────────────────────────────────────────────

def test_the_question_uses_greek_room_names():
    """‼️ The first live run asked «Να πάω  στο saloni;» — the robot reading a
    map key aloud, with a double space where the article should be."""
    q = confirm_question('goto', {'location': 'saloni'})
    assert q == 'Να πάω στο σαλόνι;'
    assert 'saloni' not in q
    assert '  ' not in q


def test_the_question_is_inflected():
    """'στο διάδρομος' is what a bare noun lookup gives. room_locative_el
    carries the article for exactly this reason."""
    assert confirm_question('goto', {'location': 'diadromos'}) == 'Να πάω στον διάδρομο;'
    assert confirm_question('goto', {'location': 'kouzina'}) == 'Να πάω στην κουζίνα;'


@pytest.mark.parametrize('tool,args', [
    ('dock', {}), ('patrol', {}), ('explore', {}), ('goto_pointed', {}),
    ('move', {'direction': 'forward'}), ('find', {'object': 'κλειδιά'}),
    ('tidy', {'room': 'all'}), ('goto', {}),
])
def test_every_motion_tool_has_a_yes_no_question(tool, args):
    q = confirm_question(tool, args)
    assert q.endswith(';'), f'{tool} does not ask a question'
    assert '{' not in q, f'{tool} left a template placeholder in: {q}'


def test_the_window_is_short_enough_to_be_safe():
    """A "ναι" belonging to a different conversation must not launch a trip.
    Long enough to answer, short enough not to linger."""
    assert 10.0 <= PENDING_TTL <= 60.0


# ── wiring: llm_bridge must ask before it dispatches ─────────────────────────

def test_the_gate_runs_at_the_top_of_dispatch():
    body = _BRIDGE.split('def _dispatch_tool')[1]
    gate = body.index('MOTION_TOOLS')
    first_tool = body.index("if name == 'tidy'")
    assert gate < first_tool, 'a tool can run before the confirmation gate'


def test_the_robot_speaks_the_question_itself():
    """‼️ Not left to the model — see the module docstring."""
    body = _BRIDGE.split('def _dispatch_tool')[1].split("if name == 'tidy'")[0]
    assert 'response_pub.publish' in body, 'the question is never spoken'


def test_the_model_reply_is_suppressed_after_a_question():
    """Otherwise the person hears the question AND «Πηγαίνω στο σαλόνι» — two
    sentences, the second one false."""
    assert '_confirm_asked' in _BRIDGE
    assert 'def _publish_reply' in _BRIDGE
    # The keyword gates (status, location) answer from telemetry and never go
    # near the model, so they publish directly and should. It is the three
    # BACKEND handlers that must not.
    for backend in ('_handle_text_lemonade', '_handle_text_ollama',
                    '_handle_text_gemini'):
        body = _BRIDGE.split(f'def {backend}')[1].split('\n    def ')[0]
        assert 'self.response_pub.publish(String(data=reply))' not in body, \
            f'{backend} publishes the model reply directly, bypassing the gate'
        assert '_publish_reply' in body, f'{backend} never speaks'


def test_the_confirmation_answer_is_checked_before_the_busy_lock():
    """Same reason stop is: during a blocking goto the lock is held for up to
    nav_timeout, and a yes/no dropped there is a question left unanswered."""
    handler = _BRIDGE.split('def _on_speech_text')[1]
    answer = handler.index('_answer_pending_motion')
    lock = handler.index('self._busy.acquire')
    assert answer < lock


def test_a_stop_clears_a_pending_question():
    handler = _BRIDGE.split('def _on_speech_text')[1].split('def ')[0]
    stop = handler.index('is_stop_command')
    clear = handler.index('self._pending_motion = None')
    assert stop < clear, 'a stop leaves the pending motion armed'


def test_confirmed_calls_are_not_asked_about_again():
    """The confirmed dispatch comes back through _dispatch_tool; without the
    flag it would ask the same question forever."""
    assert '_motion_confirmed' in _BRIDGE
    body = _BRIDGE.split('def _dispatch_tool')[1].split("if name == 'tidy'")[0]
    assert 'not self._motion_confirmed' in body


def test_it_can_be_switched_off():
    assert "declare_parameter('confirm_motion'" in _BRIDGE
