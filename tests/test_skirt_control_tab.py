"""The collision_monitor hard-stop skirt control, rendered in a real browser.

Same reasoning as test_dashboard_safety_tab.py's header — a top-level throw is
an HTTP 200 and a dead control, invisible to string-only tests. This one is
separate from that file because the skirt control is NOT one of SAFETY_SPECS
(it needs a restart, not a live SetParameters write) and has its own confirm
dialog + fetch flow to verify, closer to the map-switch buttons than to the
safety sliders.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_skirt_control_tab.py -q
"""
import json
import re
from pathlib import Path

import pytest

from home_robot import arm_settings as arms
from home_robot import collision_skirt as cs
from home_robot import mic_settings as ms
from home_robot import safety_settings as ss
from home_robot.dashboard_i18n import LANGUAGES, as_js_table

playwright = pytest.importorskip('playwright.sync_api',
                                 reason='playwright not installed')

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()
_TEMPLATE = _SRC.split('HTML_TEMPLATE = r"""', 1)[1].split('"""', 1)[0]


def _usb_devices():
    body = re.search(r'USB_DEVICES = (\{.*?\n\})', _SRC, re.S).group(1)
    return dict(re.findall(r"'(\w+)':\s*'([^']*)'", body))


_SUBS = {
    '__ROOMS__': json.dumps(['saloni', 'kouzina']),
    '__TOKEN_QS__': json.dumps(''),
    '__ARM_LIMITS__': re.search(r'ARM_LIMITS = (\{.*?\n\})', _SRC, re.S).group(1)
                        .replace("'", '"').replace(',\n}', '\n}'),
    '__ARM_JOINTS__': re.search(r'ARM_JOINTS = (\[.*?\])', _SRC, re.S).group(1)
                        .replace("'", '"'),
    '__ARM_MECH_LIMITS__': json.dumps({j: list(v) for j, v in arms.MECH_LIMITS.items()}),
    '__HAS_NOVNC__': 'false',
    '__USB_DEVICES__': json.dumps(_usb_devices(), ensure_ascii=False),
    '__SAFETY_SPECS__': json.dumps(
        {s.key: {'kind': s.kind, 'def': s.default, 'lo': s.lo, 'hi': s.hi,
                 'step': s.step, 'warn_above': s.warn_above,
                 'warn_below': s.warn_below} for s in ss.SPECS}),
    '__SAFETY_INFO__': json.dumps(ss.INFO_ONLY),
    '__MIC_SPECS__': json.dumps(
        {s.key: {'kind': s.kind, 'def': s.default, 'lo': s.lo, 'hi': s.hi,
                 'step': s.step, 'warn_above': s.warn_above,
                 'warn_below': s.warn_below} for s in ms.SPECS}),
    '__MIC_INFO__': json.dumps(ms.INFO_ONLY),
    '__WAKE_MODEL_CHOICES__': json.dumps(ms.WAKE_MODEL_CHOICES),
    '__SKIRT_MARGINS__': json.dumps(cs.ALLOWED_MARGINS_MM),
    '__SKIRT_DEFAULT_MM__': json.dumps(cs.MARGIN_DEFAULT_MM),
    '__I18N__': json.dumps(as_js_table(), ensure_ascii=False),
    '__LANGS__': json.dumps(LANGUAGES, ensure_ascii=False),
}


def _page_html():
    html = _TEMPLATE
    for token, value in _SUBS.items():
        html = html.replace(token, value)
    left = re.findall(r'__[A-Z_]+__', html)
    assert not left, f'unsubstituted placeholders, this harness is stale: {left}'
    # The photorealistic-scan viewer is a type="module" script importing "three"
    # through an importmap of ABSOLUTE server paths. set_content() serves the
    # page from about:blank, where /vendor/… resolves to nothing and the import
    # throws before the rest of the page's script gets to run. Nothing this file
    # tests lives in that module, so it is dropped rather than stubbed.
    return re.sub(r'<script type="(?:module|importmap)">.*?</script>', '',
                  html, flags=re.S)


@pytest.fixture(scope='module')
def page():
    with playwright.sync_playwright() as pw:
        browser = pw.webkit.launch()
        pg = browser.new_page(viewport={'width': 390, 'height': 664})
        errors = []
        pg.on('pageerror', lambda e: errors.append(str(e)))
        # confirm() blocks a headless page forever unless a handler answers it;
        # default to accepting so the click flow can be exercised, and record
        # every dialog's message so tests can assert on the wording too.
        pg.__dict__['_dialogs'] = []
        pg.on('dialog', lambda d: (pg._dialogs.append(d.message), d.accept()))
        pg.set_content(_page_html())
        pg.wait_for_timeout(400)
        fatal = [e for e in errors if 'WebSocket' not in e and 'ws' not in e.lower()]
        assert not fatal, 'the dashboard threw on load:\n  ' + '\n  '.join(fatal)
        pg.evaluate('window.__sent = []; window.send = m => window.__sent.push(m);')
        yield pg
        browser.close()


def _show_safety(page):
    page.evaluate("showTab('safe')")
    page.wait_for_timeout(60)
    page.evaluate('window.__dialogs = []')  # fresh per-test, real list lives on pg


# ── the control exists and starts sane ──────────────────────────────────────

def test_the_select_offers_exactly_the_allowed_margins(page):
    _show_safety(page)
    values = page.evaluate(
        "Array.from(document.querySelectorAll('#sk-mm option')).map(o => o.value)")
    assert values == [str(mm) for mm in cs.ALLOWED_MARGINS_MM]


def test_the_select_defaults_to_the_shipped_margin(page):
    _show_safety(page)
    assert page.evaluate("$('sk-mm').value") == str(cs.MARGIN_DEFAULT_MM)
    assert page.evaluate("$('sk-badge').textContent") == f'{cs.MARGIN_DEFAULT_MM} mm'


# ── a live 'safety' broadcast updates the display ───────────────────────────

def test_a_safety_broadcast_updates_the_badge_and_select(page):
    _show_safety(page)
    page.evaluate("HANDLERS.safety({type:'safety', v:{}, skirt_margin_mm: 60})")
    assert page.evaluate("$('sk-badge').textContent") == '60 mm'
    assert page.evaluate("$('sk-mm').value") == '60'


def test_a_broadcast_without_the_field_does_not_clear_the_display(page):
    """Older/unrelated 'safety' messages (e.g. a plain knob tick before this
    field existed) must not blank a value that was just shown."""
    _show_safety(page)
    page.evaluate("HANDLERS.safety({type:'safety', v:{}, skirt_margin_mm: 45})")
    page.evaluate("HANDLERS.safety({type:'safety', v:{}})")
    assert page.evaluate("$('sk-badge').textContent") == '45 mm'


def test_the_select_does_not_fight_a_focused_user(page):
    """Same rule the safety sliders follow: an update must not yank a control
    out from under a finger that's mid-choice."""
    _show_safety(page)
    page.evaluate("$('sk-mm').focus()")
    page.evaluate("HANDLERS.safety({type:'safety', v:{}, skirt_margin_mm: 55})")
    assert page.evaluate("$('sk-mm').value") != '55'
    assert page.evaluate("$('sk-badge').textContent") == '55 mm'   # badge still updates


# ── applying sends the right request, after confirming ──────────────────────

def test_apply_asks_for_confirmation_first(page):
    _show_safety(page)
    page.click('#b-sk-apply')
    dialogs = page.evaluate('1') and page._dialogs   # access the python-side list
    assert dialogs, 'no confirm() dialog was shown'
    assert 'επανεκκίνηση' in dialogs[-1].lower() or 'restart' in dialogs[-1].lower() \
        or 'σκληρό όριο' in dialogs[-1]


def test_apply_fetches_the_chosen_margin(page):
    """‼️ Some unrelated background call (present even on a bare page load
    with no interaction — not this feature's doing) occasionally logs a
    null/undefined URL through this same window.fetch override. Filtering
    for the skirt endpoint is what this test actually cares about: exactly
    one call, to the chosen margin."""
    _show_safety(page)
    page.evaluate("""
        window.__fetched = [];
        window.fetch = (url) => { window.__fetched.push(url);
            return Promise.resolve({json: () => Promise.resolve({ok:true})}); };
    """)
    page.select_option('#sk-mm', '30')
    page.click('#b-sk-apply')
    page.wait_for_timeout(50)
    fetched = [u for u in page.evaluate('window.__fetched') if u and 'skirt' in u]
    assert fetched == ['/safety/skirt/30']


def test_apply_shows_the_error_from_a_failed_response(page):
    _show_safety(page)
    page.evaluate("""
        window.fetch = () => Promise.resolve(
            {json: () => Promise.resolve({error:'boom'})});
    """)
    page.click('#b-sk-apply')
    page.wait_for_timeout(50)
    assert 'boom' in page.evaluate("$('sk-msg').textContent")
