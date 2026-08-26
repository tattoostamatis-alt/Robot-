"""The NeRF tab's live coverage map, rendered in a real browser engine.

‼️ String-only tests do not catch a top-level throw — see
[[feedback_dashboard_js_never_tested]] and test_dashboard_safety_tab.py's
header for why this suite runs in WebKit instead.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_nerf_coverage_map.py -q
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
        pg.set_content(_page_html())
        pg.wait_for_timeout(400)
        fatal = [e for e in errors if 'WebSocket' not in e and 'ws' not in e.lower()]
        assert not fatal, 'the dashboard threw on load:\n  ' + '\n  '.join(fatal)
        pg.evaluate('window.__sent = []; window.send = m => window.__sent.push(m);')
        yield pg
        browser.close()


def _send_nerf(page, **kw):
    msg = {'type': 'nerf', 'active': True, 'frames': 0, 'max_frames': 600,
           'dir': '/tmp/house', 'last_xy': None}
    msg.update(kw)
    page.evaluate('m => HANDLERS.nerf(m)', msg)


def test_frames_and_state_render(page):
    _send_nerf(page, frames=3, last_xy=[0.1, 0.2])
    assert page.evaluate("$('nf-frames').textContent") == '3 / 600'
    assert 'ok' in page.evaluate("$('nf-state').className")


def test_coverage_points_accumulate(page):
    _send_nerf(page, frames=0, last_xy=None)     # fresh session start
    assert page.evaluate('nfPoints.length') == 0
    for i in range(1, 6):
        _send_nerf(page, frames=i, last_xy=[i * 0.1, i * 0.05])
    assert page.evaluate('nfPoints.length') == 5
    assert page.evaluate('nfPoints[nfPoints.length-1]') == [0.5, 0.25]


def test_repeated_status_without_new_frame_does_not_duplicate(page):
    _send_nerf(page, frames=0, last_xy=None)
    _send_nerf(page, frames=1, last_xy=[1.0, 1.0])
    for _ in range(3):
        _send_nerf(page, frames=1, last_xy=[1.0, 1.0])   # same status, re-published every 1s
    assert page.evaluate('nfPoints.length') == 1


def test_new_session_clears_old_points(page):
    _send_nerf(page, frames=0, last_xy=None)
    _send_nerf(page, frames=1, last_xy=[9.0, 9.0])
    assert page.evaluate('nfPoints.length') == 1
    _send_nerf(page, frames=0, last_xy=None)             # restart
    assert page.evaluate('nfPoints.length') == 0


def test_canvas_exists_in_nerf_tab(page):
    assert page.evaluate("!!$('nf-map')")
