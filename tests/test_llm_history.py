"""Tests for the rolling-history policy in llm_bridge_node.

Action turns must NOT stay in the history. Measured on qwen3.5:4b (2026-07-30):
with the tool_call + tool result + "Πήγα στην κουζίνα." kept, repeating the same
command answers from that history instead of calling the tool again — explore,
follow, dock and goto each fired 1/3, while the reply claimed the action had
happened and the robot stood still. Dropping action turns took the same bench
to 15/15 (scripts/tool_repeat.py).

Robot-free: _remember_turn is exercised unbound against a stand-in, and the
rest is a static wiring check.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_llm_history.py -q
"""
import ast
import os

from home_robot.nodes.llm_bridge_node import ACTION_STARTED_NOTE, LLMBridgeNode

NODE_SRC = os.path.join(os.path.dirname(__file__), os.pardir,
                        'home_robot', 'nodes', 'llm_bridge_node.py')


class _Stand_in:
    """Just the two attributes _remember_turn touches."""

    def __init__(self, history_turns=4):
        self._history = []
        self.history_turns = history_turns

    remember = LLMBridgeNode._remember_turn


def test_chat_turn_is_remembered():
    n = _Stand_in()
    turn = [{'role': 'user', 'content': 'γεια σου Μαξ'},
            {'role': 'assistant', 'content': 'Γεια!'}]
    n.remember(turn, acted=False)
    assert n._history == [turn]


def test_action_turn_is_not_remembered():
    n = _Stand_in()
    n.remember([{'role': 'user', 'content': 'πήγαινε στην κουζίνα'},
                {'role': 'tool', 'content': '{"status": "ok"}'},
                {'role': 'assistant', 'content': 'Πήγα στην κουζίνα.'}],
               acted=True)
    assert n._history == []


def test_repeated_command_sees_no_trace_of_the_first():
    """The regression itself: nothing in history can say "already done"."""
    n = _Stand_in()
    for _ in range(3):
        n.remember([{'role': 'user', 'content': 'ξεκίνα εξερεύνηση'},
                    {'role': 'assistant', 'content': 'Ξεκινώ την εξερεύνηση.'}],
                   acted=True)
    assert n._history == []


def test_history_is_trimmed_to_history_turns():
    n = _Stand_in(history_turns=2)
    for i in range(5):
        n.remember([{'role': 'user', 'content': f'κουβέντα {i}'}], acted=False)
    assert len(n._history) == 2
    assert n._history[0][0]['content'] == 'κουβέντα 3'


def test_started_note_is_present_tense_only():
    """The note exists to stop past-tense claims about an action just begun."""
    assert 'ενεστώτα' in ACTION_STARTED_NOTE
    assert 'ΜΟΛΙΣ ξεκίνησε' in ACTION_STARTED_NOTE


def test_backends_route_history_through_remember_turn():
    """No backend may append to _history directly — that is how the bug got in."""
    tree = ast.parse(open(NODE_SRC, encoding='utf-8').read())
    appends = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'append'
               and isinstance(n.func.value, ast.Attribute)
               and n.func.value.attr == '_history']
    assert len(appends) == 1, 'self._history.append outside _remember_turn'

    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == '_remember_turn')
    inside = list(ast.walk(fn))
    assert all(a in inside for a in appends), \
        'the only _history.append must live in _remember_turn'

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == '_remember_turn']
    assert len(calls) == 3, 'lemonade, ollama and gemini must each commit turns'


def test_lemonade_keeps_system_message_first():
    """FLM's template rejects a system message that follows another role, and
    reports it as {"error": ...} with HTTP 200 — so the follow-up note must be
    folded into messages[0], never appended."""
    src = open(NODE_SRC, encoding='utf-8').read()
    head, _, tail = src.partition('def _handle_text_lemonade')
    body = tail.partition('# ── ollama backend')[0]
    assert 'ACTION_STARTED_NOTE' in body
    assert "messages.append({'role': 'system'" not in body
    assert 'messages[0] = ' in body
