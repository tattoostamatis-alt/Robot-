"""Pose/AprilTag on-demand toggles on the Κάμερα tab, rendered in a real
browser engine.

‼️ String-only tests do not catch a top-level throw — see
[[feedback_dashboard_js_never_tested]] and test_dashboard_safety_tab.py's
header for why this suite runs in WebKit instead.

Both toggles read their state off the last 'sys' broadcast's `nodes` list
(`updatePerceptionToggles`) rather than tracking it themselves client-side, so
a click just asks for the opposite of whatever the button currently shows —
these tests exercise exactly that: sys() driving the button/pill, and a click
sending the right `on` value for whatever state sys() last put it in.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_camera_toggles.py -q
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
    return html


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


def _sys(page, nodes):
    page.evaluate('n => HANDLERS.sys({type: "sys", nodes: n, cpu: 0, mem: 0})', nodes)


def _show_cam(page):
    page.evaluate("showTab('cam')")
    page.wait_for_timeout(60)


def test_both_buttons_exist_in_the_camera_tab(page):
    assert page.evaluate("!!$('b-pose-toggle')")
    assert page.evaluate("!!$('b-apriltag-toggle')")
    assert page.evaluate("!!$('pose-pill')")
    assert page.evaluate("!!$('apriltag-pill')")


def test_sys_with_neither_node_shows_both_off(page):
    _sys(page, ['robot_state_publisher', 'amcl'])
    assert page.evaluate("$('b-pose-toggle').classList.contains('pri')") is False
    assert page.evaluate("$('b-apriltag-toggle').classList.contains('pri')") is False
    assert page.evaluate("$('pose-pill').textContent") == 'σβηστό'
    assert page.evaluate("$('apriltag-pill').textContent") == 'σβηστό'


def test_sys_with_pose_node_only_lights_up_just_pose(page):
    _sys(page, ['robot_state_publisher', 'pose_node', 'amcl'])
    assert page.evaluate("$('b-pose-toggle').classList.contains('pri')") is True
    assert page.evaluate("$('pose-pill').textContent") == 'ενεργό'
    assert page.evaluate("$('b-apriltag-toggle').classList.contains('pri')") is False
    assert page.evaluate("$('apriltag-pill').textContent") == 'σβηστό'


def test_sys_with_apriltag_node_only_lights_up_just_apriltag(page):
    _sys(page, ['robot_state_publisher', 'apriltag_node', 'amcl'])
    assert page.evaluate("$('b-apriltag-toggle').classList.contains('pri')") is True
    assert page.evaluate("$('apriltag-pill').textContent") == 'ενεργό'
    assert page.evaluate("$('b-pose-toggle').classList.contains('pri')") is False


def test_clicking_pose_while_off_asks_to_turn_it_on(page):
    _show_cam(page)
    _sys(page, [])
    page.evaluate("window.__sent = []")
    page.click('#b-pose-toggle')
    assert page.evaluate('window.__sent') == [{'type': 'toggle_pose', 'on': True}]


def test_clicking_pose_while_on_asks_to_turn_it_off(page):
    _show_cam(page)
    _sys(page, ['pose_node'])
    page.evaluate("window.__sent = []")
    page.click('#b-pose-toggle')
    assert page.evaluate('window.__sent') == [{'type': 'toggle_pose', 'on': False}]


def test_clicking_apriltag_while_off_asks_to_turn_it_on(page):
    _show_cam(page)
    _sys(page, [])
    page.evaluate("window.__sent = []")
    page.click('#b-apriltag-toggle')
    assert page.evaluate('window.__sent') == [{'type': 'toggle_apriltag', 'on': True}]


def test_clicking_apriltag_while_on_asks_to_turn_it_off(page):
    _show_cam(page)
    _sys(page, ['apriltag_node'])
    page.evaluate("window.__sent = []")
    page.click('#b-apriltag-toggle')
    assert page.evaluate('window.__sent') == [{'type': 'toggle_apriltag', 'on': False}]
