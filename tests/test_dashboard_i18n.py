"""Tests for the dashboard's Greek/English/German interface.

The one that matters is test_every_ui_string_is_translated: it pulls every
Greek string back out of the dashboard's markup and JS and fails if it is not
in the table. Without it, translations rot the moment someone adds a button —
and the rot is invisible, because an untranslated string still renders (as
Greek), so English and German quietly turn into a mixture.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_i18n.py -q
"""
import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from home_robot.dashboard_i18n import LANGUAGES, TRANSLATIONS, as_js_table

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()
_TPL = _SRC.split('HTML_TEMPLATE = r"""', 1)[1].split('</html>', 1)[0]
_MARKUP, _JS = _TPL.split('<script>', 1)

GREEK = re.compile(r'[Ͱ-Ͽἀ-῿]')


def _norm(s):
    return re.sub(r'\s+', ' ', s).strip()


def markup_strings():
    """Greek text the browser renders straight out of the template."""
    out = []
    for m in re.finditer(r'>([^<>{}]+)<', _MARKUP):
        if GREEK.search(m.group(1)):
            out.append(_norm(m.group(1)))
    for m in re.finditer(r'(?:placeholder|title)="([^"]+)"', _MARKUP):
        if GREEK.search(m.group(1)):
            out.append(_norm(m.group(1)))
    # The browser hands t() the DECODED text, so &amp; must be looked up as &.
    return [s for s in dict.fromkeys(html.unescape(s) for s in out) if s]


# A t() argument may be several literals joined with +, because the source
# wraps at 80 columns. Matching one literal at a time would look for half a
# sentence in the table and "find" a missing translation that is not missing.
_LITERAL = r"""(?:'[^'\n]*'|"[^"\n]*")"""
T_CALL = re.compile(rf"\bt\(\s*({_LITERAL}(?:\s*\+\s*{_LITERAL})*)\s*\)")
_COMMENTS = re.compile(r'//[^\n]*|/\*.*?\*/', re.S)


def _join(literals):
    return ''.join(part[1:-1] for part in re.findall(_LITERAL, literals))


# Quoted spans that CONTAIN Greek, on a single line. Deliberately not a full
# JS tokeniser: quote pairing inside template literals (`${x ? '' : 'ΝΑΙ'}`)
# and regex literals makes a general scanner fragile, and everything this file
# needs to police is a Greek run between two quotes with no newline in it.
_GREEK_LITERAL = re.compile(
    r"'([^'\n]*[Ͱ-Ͽἀ-῿][^'\n]*)'"
    r'|"([^"\n]*[Ͱ-Ͽἀ-῿][^"\n]*)"')


def js_strings():
    """Every Greek string literal in the page's JS, wherever it is written —
    passed to t() directly, or sitting in a table that is t()'d at render."""
    js = _COMMENTS.sub('', _JS)
    # A t() argument may be several literals joined with + (the source wraps at
    # 80 columns). The JOINED form is the key to look up; the halves are not
    # strings the UI ever shows, so they are skipped rather than reported as
    # missing translations that do not exist.
    calls = [(m.start(), m.end(), _norm(_join(m.group(1))))
             for m in T_CALL.finditer(js)]
    found = [text for _, _, text in calls]
    found += [_norm(m.group(1) or m.group(2))
              for m in _GREEK_LITERAL.finditer(js)
              if not any(a <= m.start() < b for a, b, _ in calls)]
    return [s for s in dict.fromkeys(found) if s]


# Where an untranslated string would actually reach the user's eyes. NOT
# `title:` — that is the key in the VNC_APPS table, whose text is t()'d where
# it is rendered, not where it is declared.
_SINKS = re.compile(r'\b(textContent|innerHTML|fillText|confirm|alert)\b')


def js_greek_outside_t():
    """Greek that reaches the page without passing through t()."""
    # Strip t() calls from the WHOLE text first: they wrap across lines, and
    # per-line stripping leaves the first half looking unwrapped.
    js = T_CALL.sub('', _COMMENTS.sub('', _JS))
    return [ln.strip() for ln in js.splitlines()
            if _SINKS.search(ln) and GREEK.search(ln)]


# ── the table must keep up with the page ─────────────────────────────────────

def test_every_ui_string_is_translated():
    table = {_norm(k) for k in TRANSLATIONS}
    missing = [s for s in markup_strings() + js_strings() if s not in table]
    # Room names come from locations.yaml — the user's own words, not UI.
    assert not missing, ('untranslated UI strings:\n  '
                         + '\n  '.join(repr(s) for s in missing))


def test_no_greek_is_left_outside_t():
    """A literal that never reaches t() stays Greek in every language — the
    exact way a translated UI ends up half Greek without anyone noticing."""
    leftover = js_greek_outside_t()
    assert not leftover, ('Greek not routed through t():\n  '
                          + '\n  '.join(leftover))


def test_no_dead_entries():
    """A translation for a string the page no longer has is dead weight, and
    hides the fact that the real string went untranslated."""
    live = {_norm(s) for s in markup_strings() + js_strings()}
    dead = [k for k in TRANSLATIONS if _norm(k) not in live]
    assert not dead, 'entries no longer used by the page: ' + repr(dead)


# ── the table itself ─────────────────────────────────────────────────────────

def test_all_three_languages_are_offered():
    assert [c for c, _ in LANGUAGES] == ['el', 'en', 'de']


def test_every_entry_has_english_and_german():
    for src, val in TRANSLATIONS.items():
        assert len(val) == 2, src
        assert all(isinstance(v, str) and v.strip() for v in val), src


# Text the user has to SAY OUT LOUD stays Greek in every UI language: the wake
# word is a trained model and the voice tools match Greek keywords, so an
# English rendering would be an instruction that does not work. Such phrases are
# always quoted in the copy, so strip quoted spans before checking — that keeps
# the rule honest (prose must be translated) without special-casing each string.
_QUOTED = re.compile(r'«[^»]*»|"[^"]*"|„[^“]*“')


def _greek_outside_quotes(text):
    return GREEK.search(_QUOTED.sub('', text))


def test_no_translation_is_left_as_greek():
    """Copy-pasting the Greek into the English column is how a table silently
    does nothing."""
    for src, (en, de) in TRANSLATIONS.items():
        if not GREEK.search(src):
            continue
        exempt = 'ρομπότ' in src.lower()
        assert not _greek_outside_quotes(en) or exempt, f'en still Greek: {src!r}'
        assert not _greek_outside_quotes(de) or exempt, f'de still Greek: {src!r}'


def test_emoji_prefixes_are_preserved():
    """The buttons are identified by their icon as much as their text."""
    for src, (en, de) in TRANSLATIONS.items():
        lead = re.match(r'^([\U0001F300-\U0001FAFF -➿️]+\s*)', src)
        if lead:
            assert en.startswith(lead.group(1).strip()), f'en lost the icon: {src!r}'
            assert de.startswith(lead.group(1).strip()), f'de lost the icon: {src!r}'


def test_the_table_reaches_the_page():
    assert '__I18N__' in _JS and '__LANGS__' in _JS
    assert "'__I18N__'" in _SRC and "'__LANGS__'" in _SRC


def test_js_table_shape():
    tbl = as_js_table()
    sample = next(iter(tbl.values()))
    assert set(sample) == {'en', 'de'}
    assert len(tbl) == len(TRANSLATIONS)


# ── the browser's own t(), run for real ──────────────────────────────────────

@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_t_falls_back_to_greek_rather_than_showing_a_blank():
    """An unknown key must render as Greek. Returning undefined would put the
    string "undefined" on a button."""
    fn = re.search(r'function t\(s\)\{.*?\n\}', _JS, re.S)
    assert fn, 't() changed shape'
    script = f'''
const I18N = {json.dumps(as_js_table(), ensure_ascii=False)};
let LANG = 'de';
{fn.group(0)}
const out = [t('Χάρτης'), t('κάτι που δεν υπάρχει'), t(''), t(null)];
LANG = 'el';
out.push(t('Χάρτης'));
console.log(JSON.stringify(out));
'''
    # Through a FILE, not `node -e`: the table is well past the kernel's
    # 128 KB limit on a single argument, and passing it inline failed with
    # "Argument list too long" as soon as the table grew.
    with tempfile.NamedTemporaryFile('w', suffix='.js', encoding='utf-8',
                                     delete=False) as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True,
                           timeout=30)
    finally:
        Path(path).unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got[0] == 'Karte'
    assert got[1] == 'κάτι που δεν υπάρχει'      # unknown -> unchanged
    assert got[2] == ''
    assert got[3] is None                        # must not crash on null
    assert got[4] == 'Χάρτης'                    # el is the source language


# ── the page has to actually run ─────────────────────────────────────────────
# ‼️ 2026-08-01: i18nCollect() was called next to the button wiring, ABOVE the
# `let i18nNodes` that it assigns to. let/const are hoisted but not
# initialised, so the browser threw "Cannot access 'i18nNodes' before
# initialization" and the ENTIRE script died there. Everything after it —
# building the tabs, the camera stream, connect() — never ran, so the page drew
# its layout and then sat there dead. The string tests above all passed
# throughout: they check the table, not whether the page executes.

def _js_lines():
    return _COMMENTS.sub('', _JS).splitlines()


@pytest.mark.parametrize('name', ['i18nNodes', 'LANGS', 'I18N', 'LANG'])
def test_no_top_level_use_before_declaration(name):
    """A `let`/`const` may not be used by top-level code above its declaration."""
    lines = _js_lines()
    decl = next((i for i, ln in enumerate(lines)
                 if re.match(rf'\s*(let|const)\s+{name}\b', ln)), None)
    assert decl is not None, f'{name} is no longer declared'
    for i, ln in enumerate(lines[:decl]):
        # Top-level statements only: anything indented is inside a function,
        # which does not run until it is called.
        if ln.startswith((' ', '\t', '}')) or not ln.strip():
            continue
        assert not re.search(rf'\b{name}\b', ln), (
            f'line {i + 1} uses {name} before it is declared on line {decl + 1}: '
            f'{ln.strip()!r}')


def test_the_startup_calls_are_in_the_start_block():
    """They belong at the bottom, after every declaration — not next to the
    button wiring, which is where they were when the page died.

    Top-level calls only: applyLang() also appears inside setLang(), which is a
    function body and does not run at load.
    """
    # Locate the block on the un-stripped source: _COMMENTS would remove the
    # very marker being looked for.
    start = _JS.index('// ── start')
    tail = _JS[start:]
    js = _COMMENTS.sub('', _JS)
    for call in ('i18nCollect();', 'applyLang();', 'connect();'):
        # Unindented = top level. The lines carry trailing comments and
        # `resize(); connect();` shares a line, so match on containment.
        top_level = [ln for ln in js.splitlines()
                     if call in ln and not ln.startswith((' ', '\t'))]
        assert top_level, f'{call} is no longer called at the top level'
        assert call in tail, f'{call} runs before the start block'


def test_connect_is_the_last_thing_that_runs():
    """If anything after connect() throws it only loses that feature; if
    anything BEFORE it throws, the robot goes silent behind a page that looks
    perfectly fine. Keep the websocket last."""
    js = _COMMENTS.sub('', _JS).split('</script>')[0].rstrip()
    assert js.endswith('connect();'), 'connect() is no longer last'


def test_tabs_are_built_before_the_first_showTab():
    """showTab() marks a tab active; if applyLang() has not built them yet the
    page opens with no navigation at all."""
    js = _COMMENTS.sub('', _JS)
    assert js.index('applyLang();') < js.index("showTab('map');")
