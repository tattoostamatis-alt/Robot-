"""Tests for the TOOLS prefill router.

The asymmetry that shapes these tests: sending too many tools costs latency,
sending too few loses a command outright. So the false-negative cases (every
real command reaches its tool) are the ones that matter, and they are checked
against phrasings actually seen in the 2026-07-30 voice logs, STT mangling
included.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from home_robot.status_query import ROOM_NAMES_EL  # noqa: E402
from home_robot.tool_router import select_tools  # noqa: E402

# Import the real TOOLS so the tests break if a tool is renamed or added
# without being routed. The room enum is derived from status_query in the
# node, so seed it here — exec'ing the snippet runs no imports.
_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'llm_bridge_node.py').read_text()
_ns = {'ROOM_NAMES_EL': ROOM_NAMES_EL}
exec(compile(  # noqa: S102 - test-only extraction of a module-level constant
    __import__('re').search(r'^_APPS = .*?^\]', _SRC, 8 | 16).group(0)
    + '\n'
    + __import__('re').search(r'^_ROOMS = .*?^\]|^_ROOMS = .*$', _SRC, 8 | 16).group(0)
    + '\n'
    + __import__('re').search(r'^TOOLS = \[.*?^\]', _SRC, 8 | 16).group(0),
    'llm_bridge_tools', 'exec'), _ns)
TOOLS = _ns['TOOLS']
ALL_NAMES = {t['function']['name'] for t in TOOLS}


def names(text):
    return {t['function']['name'] for t in select_tools(text, TOOLS)}


# ── commands must always reach their tool ────────────────────────────────────

@pytest.mark.parametrize('text,expected', [
    ('Πήγαινε στο σαλόνι', 'goto'),
    ('πήγαινε στη κουζίνα', 'goto'),
    ('Πήγαινε στη βάση', 'goto'),
    ('Γύρνα στη βάση φόρτισης', 'dock'),
    ('Καθάρισε την κουζίνα', 'tidy'),
    ('Τακτοποίησε το δωμάτιο του Μαξ', 'tidy'),
    ('Ξεκίνα περιπολία', 'patrol'),
    ('Εξερεύνησε το σπίτι', 'explore'),
    ('Προχώρα λίγο μπροστά', 'move'),
    ('Στρίψε δεξιά', 'move'),
    ('Ακολούθησέ με', 'follow'),
    ('Άνοιξε το Firefox!', 'open_app'),
    ('άνοιξε τον κειμενογράφο', 'open_app'),
    ('τι είναι ανοιχτό στην οθόνη;', 'list_windows'),
    ('Πόση μπαταρία έχεις;', 'system_status'),
    ('Πόση μνήμη RAM έχεις ελεύθερη;', 'system_status'),
    ('Τι θερμοκρασία έχεις;', 'system_status'),
    ('τι σκόρπια αντικείμενα υπάρχουν;', 'report_clutter'),
    ('Φέρε μου το μπουκάλι', 'fetch'),
    ('Πιάσε την κούπα', 'pick'),
    ('Θυμήσου ότι τα κλειδιά είναι στο συρτάρι', 'remember'),
    # Added 2026-08-01 with the tools themselves.
    ('Κλείσε όλα', 'close_app'),
    ('κλείσε το firefox', 'close_app'),
    ('Πόσο μακριά είναι ο άνθρωπος;', 'distance_to'),
    ('Πόση απόσταση έχει;', 'distance_to'),
    ('πόσα μέτρα απέχει η καρέκλα;', 'distance_to'),
])
def test_command_reaches_its_tool(text, expected):
    assert expected in names(text), f'{text!r} would lose {expected}'


# ── pure conversation must cost nothing ──────────────────────────────────────

@pytest.mark.parametrize('text', [
    'τι ώρα είναι;',
    'τι μέρα έχουμε σήμερα;',
    'Γεια σου Max, τι κάνεις;',
    'Έι, ρομπότ!',
    'Ευχαριστώ.',
    'Πάω για ύπνο.',
    'Καληνύχτα!',
    'Πώς σε λένε;',
    'Τι είσαι;',
])
def test_chitchat_sends_no_tools(text):
    assert names(text) == set(), f'{text!r} pays for tools it cannot use'


# ── the catch-all keeps unclassified commands working ────────────────────────

@pytest.mark.parametrize('text', [
    'Κάνε κάτι χρήσιμο',
    'Μπορείς να με βοηθήσεις με κάτι;',
    'Θέλω να ξεκινήσεις',
])
def test_unclassified_action_falls_back_to_full_set(text):
    assert names(text) == ALL_NAMES, f'{text!r} should degrade to the full set'


# ── STT garbage must not crash or silently swallow a real command ────────────

def test_stt_garbage_is_handled():
    # Straight from the log — the STT mangled a real "πήγαινε" command.
    assert 'goto' in names('Πήγαισε τον υπουλοκιστήλιο.')
    # Pure noise routes to nothing, which is correct: the model cannot act on it.
    assert names('Ροδικρον, μάλταν, νου, μάλτα, δαμμου.') == set()


def test_routing_is_a_real_saving():
    # The whole point: a nav command must not drag the arm/desktop tools along.
    nav = names('Πήγαινε στο σαλόνι')
    assert not nav & {'pick', 'fetch', 'open_app', 'list_windows', 'remember'}
    assert len(nav) < len(ALL_NAMES)


def test_every_tool_is_reachable():
    """No tool may be orphaned by the router — that would make it dead code."""
    covered = set()
    for text, _ in [
        ('Πήγαινε στο σαλόνι', 0), ('Γύρνα στη βάση', 0), ('Καθάρισε', 0),
        ('περιπολία', 0), ('εξερεύνησε', 0), ('σταμάτα την εξερεύνηση', 0),
        ('προχώρα μπροστά', 0), ('ακολούθησέ με', 0),
        ('σταμάτα να με ακολουθείς', 0), ('άνοιξε το firefox', 0),
        ('τι είναι ανοιχτό στην οθόνη', 0), ('πόση μπαταρία', 0),
        ('σκόρπια αντικείμενα', 0), ('πιάσε την κούπα', 0),
        ('φέρε μου το βιβλίο', 0), ('θυμήσου ότι', 0),
        ('κλείσε όλα', 0), ('πόσο μακριά είναι ο άνθρωπος', 0),
    ]:
        covered |= names(text)
    # `stop` is gated by stop_command.py before the model, never routed here.
    assert ALL_NAMES - covered == {'stop'}, ALL_NAMES - covered
