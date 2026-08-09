"""The map tab's room name/colour editor, rendered in a real browser engine.

‼️ String-only tests do not catch a top-level throw — see
[[feedback_dashboard_js_never_tested]] and test_dashboard_safety_tab.py's
header for why this suite runs in WebKit instead.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_room_editor_tab.py -q
"""
import json
import re
from pathlib import Path

import pytest

from home_robot import collision_skirt as cs
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
    '__HAS_NOVNC__': 'false',
    '__USB_DEVICES__': json.dumps(_usb_devices(), ensure_ascii=False),
    '__SAFETY_SPECS__': json.dumps(
        {s.key: {'kind': s.kind, 'def': s.default, 'lo': s.lo, 'hi': s.hi,
                 'step': s.step, 'warn_above': s.warn_above,
                 'warn_below': s.warn_below} for s in ss.SPECS}),
    '__SAFETY_INFO__': json.dumps(ss.INFO_ONLY),
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


_ROOMS = {'saloni': [204, 68, 255], 'kouzina': [255, 204, 0]}


def _load_rooms(page, rooms=_ROOMS):
    page.evaluate('rooms => HANDLERS.map({type:"map", width:10, height:10,'
                  'resolution:0.05, origin:[0,0], image:"", rooms, tinted:true})',
                  rooms)
    page.wait_for_timeout(30)


def test_editor_builds_one_row_per_room(page):
    _load_rooms(page)
    rows = page.evaluate("document.querySelectorAll('#room-edit [data-room]').length")
    assert rows == len(_ROOMS)


def test_editor_hides_itself_with_no_rooms(page):
    _load_rooms(page, {})
    assert page.evaluate("$('room-edit-row').style.display") == 'none'
    assert page.evaluate("document.querySelectorAll('#room-edit [data-room]').length") == 0
    _load_rooms(page)   # restore state for the tests below


def test_colour_input_matches_the_room_colour(page):
    _load_rooms(page)
    hex_val = page.evaluate(
        "document.querySelector('#room-edit [data-room=\"kouzina\"] .re-color').value")
    assert hex_val.lower() == '#ffcc00'


def test_saving_sends_every_row_with_edited_name_and_colour(page):
    _load_rooms(page)
    page.evaluate("""
        const row = document.querySelector('#room-edit [data-room="saloni"]');
        row.querySelector('.re-name').value = 'σαλόνι';
        row.querySelector('.re-color').value = '#112233';
        window.__sent = [];
    """)
    page.click('#b-room-save')
    sent = page.evaluate('window.__sent')
    assert len(sent) == 1 and sent[0]['type'] == 'save_rooms'
    by_old = {r['old']: r for r in sent[0]['rooms']}
    assert by_old['saloni']['name'] == 'σαλόνι'
    assert by_old['saloni']['color'] == [0x11, 0x22, 0x33]
    assert by_old['kouzina']['name'] == 'kouzina'   # untouched row still round-trips


def test_a_blank_name_drops_that_room_from_the_save(page):
    _load_rooms(page)
    page.evaluate("""
        document.querySelector('#room-edit [data-room="saloni"] .re-name').value = '  ';
        window.__sent = [];
    """)
    page.click('#b-room-save')
    sent = page.evaluate('window.__sent')[0]['rooms']
    assert [r for r in sent if r['old'] == 'saloni'] == []
    assert len(sent) == 1


def test_room_saved_ack_updates_the_status_line(page):
    _load_rooms(page)
    page.evaluate('HANDLERS.room_saved({type:"room_saved", ok:true})')
    assert 'θηκε' in page.evaluate("$('room-edit-msg').textContent").lower()
    page.evaluate('HANDLERS.room_saved({type:"room_saved", ok:false, error:"boom"})')
    assert page.evaluate("$('room-edit-msg').textContent") == 'boom'


def test_map_rooms_broadcast_refreshes_both_legend_and_editor(page):
    """A save from ANOTHER browser tab must update this one too — rooms are
    shared state, not per-tab local edits."""
    page.evaluate("""HANDLERS.map_rooms({type:'map_rooms',
        rooms:{'diadromos':[68,68,255]}})""")
    page.wait_for_timeout(30)
    assert page.evaluate(
        "document.querySelectorAll('#room-edit [data-room]').length") == 1
    assert 'diadromos' in page.evaluate("$('room-legend').textContent")
    _load_rooms(page)   # restore


# ── click-to-pick-room on the map canvas ────────────────────────────────────

def test_pick_mode_off_by_default_click_still_navigates(page):
    _load_rooms(page)
    page.evaluate("window.__sent = []; $('b-pick-room').checked = false;")
    page.click('#map-canvas')
    sent = page.evaluate('window.__sent')
    assert sent and sent[0]['type'] == 'nav_goal'


def test_pick_mode_sends_pick_room_not_nav_goal(page):
    _load_rooms(page)
    page.evaluate("window.__sent = []; $('b-pick-room').checked = true;")
    page.click('#map-canvas')
    sent = page.evaluate('window.__sent')
    assert len(sent) == 1 and sent[0]['type'] == 'pick_room'
    assert 'x' in sent[0] and 'y' in sent[0]
    page.evaluate("$('b-pick-room').checked = false;")   # restore


def test_room_picked_highlights_and_focuses_the_row(page):
    _load_rooms(page)
    page.evaluate("HANDLERS.room_picked({type:'room_picked', name:'kouzina'})")
    row = page.evaluate(
        "document.querySelector('#room-edit [data-room=\"kouzina\"]').classList.contains('picked')")
    assert row
    focused = page.evaluate(
        "document.activeElement === document.querySelector('#room-edit [data-room=\"kouzina\"] .re-name')")
    assert focused


def test_room_picked_with_no_match_shows_a_message(page):
    _load_rooms(page)
    page.evaluate("HANDLERS.room_picked({type:'room_picked', name:null})")
    assert 'δωμάτιο' in page.evaluate("$('room-edit-msg').textContent").lower()
