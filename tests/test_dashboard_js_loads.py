"""The dashboard's JavaScript must actually RUN.

2026-08-04, reported from a phone: "δεν μου δείχνει τα κουμπιά κάμερα, IMU και
τα υπόλοιπα". The page served all 21 tabs in its HTML and every existing test
passed — 1137 of them — because they all read the source as text. Not one
executed it.

The chip builders added that day looped at top level calling `t()`, which reads
`LANG`. `LANG` is declared with `let` further down the file, so the call sat in
the temporal dead zone and threw:

    Cannot access 'LANG' before initialization

That aborts the whole script. renderTabs() is defined below the throw, so the
tab bar was never built: measured in WebKit at 390x664 and 320x568, ZERO tabs
rendered. Desktop was equally dead; the phone is just where it was noticed.

These checks are static and cheap, but they encode the two properties that
would have caught it:

  1. the script parses at all (node --check), and
  2. nothing calls t() at top level before LANG exists.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_js_loads.py -q
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()

# The page's inline scripts, in the order the browser parses them.
_SCRIPTS = re.findall(r'<script>(.*?)</script>', _SRC, re.S)
_RAW = '\n'.join(_SCRIPTS)


def _blank_comments_and_strings(js):
    """Replace comments and string literals with spaces, preserving offsets.

    Required before any brace counting: this file is heavily commented, the
    comments discuss t() and contain braces, and the Greek UI strings are full
    of punctuation. Offsets are preserved so a failure can quote real source.
    """
    out = list(js)
    i, n = 0, len(js)
    while i < n:
        c = js[i]
        nxt = js[i + 1] if i + 1 < n else ''
        if c == '/' and nxt == '/':
            while i < n and js[i] != '\n':
                out[i] = ' '
                i += 1
        elif c == '/' and nxt == '*':
            while i < n and not (js[i] == '*' and js[i + 1:i + 2] == '/'):
                if js[i] != '\n':
                    out[i] = ' '
                i += 1
            out[i:i + 2] = '  '
            i += 2
        elif c in '\'"`':
            quote = c
            out[i] = ' '
            i += 1
            while i < n and js[i] != quote:
                if js[i] == '\\':
                    out[i] = ' '
                    i += 1
                if i < n and js[i] != '\n':
                    out[i] = ' '
                i += 1
            if i < n:
                out[i] = ' '
            i += 1
        else:
            i += 1
    return ''.join(out)


_CODE = _blank_comments_and_strings(_RAW)


def _top_level_t_calls(js):
    """Offsets of `t(` calls that are NOT inside a function body.

    Approximated by brace depth: at depth 0 a statement runs while the script is
    being parsed, which is exactly when LANG may not exist yet. Inside a
    function the call is deferred until something invokes it, by which time the
    whole script has been evaluated.
    """
    depth, out = 0, []
    for i, c in enumerate(js):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif c == 't' and depth == 0 and js[i + 1:i + 2] == '(':
            before = js[i - 1] if i else ' '
            if not (before.isalnum() or before in '_$.'):
                out.append(i)
    return out


# ── the script has to load at all ────────────────────────────────────────────

def test_the_page_has_inline_scripts():
    assert _SCRIPTS, 'no <script> blocks found — the extraction regex is stale'


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_the_javascript_parses():
    """A syntax error anywhere kills every tab, every button and the socket."""
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as fh:
        fh.write('\n;\n'.join(_SCRIPTS))
        path = fh.name
    proc = subprocess.run(['node', '--check', path], capture_output=True,
                          text=True)
    assert proc.returncode == 0, f'dashboard JS does not parse:\n{proc.stderr}'


def test_no_top_level_translation_call_before_lang_exists():
    """‼️ The exact bug: t() at top level, above `let LANG`, is a TDZ throw that
    silently kills the entire dashboard."""
    m = re.search(r'^\s*let\s+LANG\b', _CODE, re.M)
    assert m, 'the `let LANG` declaration is gone — this guard needs updating'
    early = [off for off in _top_level_t_calls(_CODE) if off < m.start()]
    context = [_RAW[max(0, o - 60):o + 40].replace('\n', ' ') for o in early[:3]]
    assert not early, (
        't() is called at top level before `let LANG` — this throws '
        '"Cannot access \'LANG\' before initialization" and no tab ever '
        'renders:\n  ' + '\n  '.join(context))


def test_chips_are_built_from_a_function_not_at_top_level():
    """The regression that caused it: the chip loops must stay inside
    renderChips(), which applyLang() calls once LANG is real."""
    assert 'function renderChips(' in _SRC, \
        'renderChips() is gone — chips are probably being built inline again'
    assert re.search(r'renderTabs\(\);\s*\n\s*renderChips\(\);', _SRC), \
        'renderChips() is no longer called from applyLang() beside renderTabs()'


def test_every_pane_referenced_by_a_tab_exists():
    """A tab whose pane is missing blanks the page when tapped.

    Checks ALL_TABS, not TABS: TABS is now the curated bar
    (`ALL_TABS.filter(...)`, not a literal array — see CORE_TAB_IDS), and the
    rest are reached from Ρυθμίσεις → "Περισσότερα εργαλεία" instead of the
    bar. ALL_TABS is still every tab the page has, bar or not, so it is the
    right list to demand a pane for."""
    m = re.search(r'const ALL_TABS = \[(.*?)\];', _SRC, re.S)
    assert m, 'the ALL_TABS table is gone'
    for tab_id in re.findall(r"\['(\w+)'", m.group(1)):
        assert f'id="p-{tab_id}"' in _SRC, f'tab {tab_id!r} has no pane'


# ── the stripper itself, since everything above trusts it ────────────────────

def test_stripper_removes_line_comments():
    assert 't(' not in _blank_comments_and_strings('// talk about t(x)\nvar a=1')


def test_stripper_removes_block_comments():
    assert 't(' not in _blank_comments_and_strings('/* t(x) and { */ var a=1')


def test_stripper_removes_strings():
    assert 't(' not in _blank_comments_and_strings("var s='t(x)';")
    assert 't(' not in _blank_comments_and_strings('var s=`t(x)`;')


def test_stripper_preserves_offsets_and_real_code():
    src = "// note\nt('x');"
    out = _blank_comments_and_strings(src)
    assert len(out) == len(src)
    assert out.index('t(') == src.index("t('x')")


def test_stripper_keeps_braces_outside_comments():
    assert _blank_comments_and_strings('function f(){ /* } */ }').count('}') == 1


# ── live microphone stream ───────────────────────────────────────────────────

def test_binary_frames_never_reach_json_parse():
    """Audio arrives as binary websocket frames on the SAME socket as the JSON
    state. Without the ArrayBuffer branch every chunk hits JSON.parse and throws
    ten times a second."""
    assert 'ws.binaryType' in _RAW, 'binaryType is not set — frames arrive as Blob'
    m = re.search(r'ws\.onmessage\s*=.*?JSON\.parse', _RAW, re.S)
    assert m, 'the onmessage handler changed shape'
    assert 'ArrayBuffer' in m.group(0), \
        'binary frames are not split off before JSON.parse'


def test_audio_context_does_not_force_a_sample_rate():
    """‼️ `new AudioContext({sampleRate:16000})` is ACCEPTED and then sits
    'suspended' — the stream arrives and nothing plays. The default-rate
    context resumes; createBuffer still declares 16 kHz so the browser
    resamples, which is all that was needed."""
    body = re.search(r'function startListening\(\)\{(.*?)\n\}', _RAW, re.S)
    assert body, 'startListening() is gone'
    assert 'sampleRate: MIC_RATE' not in body.group(1), \
        'the AudioContext forces a sample rate again — iOS will stay silent'
    assert re.search(r'new AC\(\s*\)', body.group(1)), \
        'the context is no longer created with default options'


def test_ios_unlock_buffer_is_started_in_the_gesture():
    """iOS needs a real sound started inside the click before the context will
    run; without it the state reads 'running' but nothing is audible."""
    body = re.search(r'function startListening\(\)\{(.*?)\n\}', _RAW, re.S)
    assert 'createBuffer(1, 1,' in body.group(1), 'the unlock buffer is gone'
    assert 'unlock.start(0)' in body.group(1)


def test_the_buffer_still_declares_the_microphone_rate():
    """The resampling only works because the buffer says 16 kHz."""
    assert 'createBuffer(1, pcm.length, MIC_RATE)' in _RAW


def test_audio_context_is_created_from_the_click_not_at_load():
    """Browsers refuse to start audio without a user gesture; a context built at
    page load comes up suspended and stays silent."""
    body = re.search(r'function startListening\(\)\{(.*?)\n\}', _RAW, re.S)
    assert body, 'startListening() is gone'
    assert 'new AC(' in body.group(1), \
        'the AudioContext is no longer created inside startListening()'


# ── every element the script wires up must exist ─────────────────────────────

def _html_ids():
    """Every id="..." in the page markup."""
    return set(re.findall(r'\bid="([\w-]+)"', _SRC))


def _wired_ids():
    """ids the script calls addEventListener/onclick on at TOP LEVEL.

    Those run as the page parses, so a missing element throws immediately and
    aborts the rest of the script — the same failure mode as the TDZ bug above.
    Inside a function the call is deferred and a stale id is merely dead code.

    Top level is detected by INDENTATION, not by brace depth: template literals
    and regex literals make brace counting unreliable, and every top-level
    statement in this file starts at column 0.
    """
    out = []
    pattern = re.compile(
        r"^\$\('([\w-]+)'\)\s*\.(addEventListener|onclick|onchange)", re.M)
    for m in pattern.finditer(_RAW):
        out.append(m.group(1))
    return out


def test_every_top_level_wired_element_exists():
    """‼️ Regression: the face-enrolment card was moved to another tab and its
    `$('b-fi-enrol').addEventListener` was left behind. That threw on load and
    took the whole dashboard down with it — zero tabs again."""
    ids = _html_ids()
    missing = sorted({i for i in _wired_ids() if i not in ids})
    assert not missing, (
        'the script wires elements that do not exist in the markup, which '
        'throws during parse and kills the page: ' + ', '.join(missing))
