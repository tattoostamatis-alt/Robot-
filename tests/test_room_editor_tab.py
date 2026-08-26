"""The map tab's room manager (#room-mgr), rendered in a real browser engine.

The panel that replaced the old legend + one-room editor + paint checkbox
(2026-08-19): a mode strip (select / add / divide / merge), the rooms as cards,
and the selected room's name+colour editor, all floating over the map because
every one of its tools is finished by touching the map.

‼️ String-only tests do not catch a top-level throw — see
[[feedback_dashboard_js_never_tested]] and test_dashboard_safety_tab.py's
header for why this suite runs in WebKit instead.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_room_editor_tab.py -q
"""
import base64
import io
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
    assert left == [], f'unsubstituted placeholders, this harness is stale: {left}'
    # The photorealistic-scan viewer is a type="module" script importing
    # "three" through an importmap of ABSOLUTE server paths. set_content()
    # serves the page from about:blank, where /vendor/… resolves to nothing
    # and the import throws before anything in this file gets to run. None of
    # the room manager lives in that module, so it is dropped outright rather
    # than stubbed.
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
        pg.evaluate('window.confirm = () => true;')
        yield pg
        browser.close()


_ROOMS = {'saloni': [204, 68, 255], 'kouzina': [255, 204, 0]}
# A 100x100 cell, 5x5 m map with the two rooms in opposite corners, so which
# one a tap is nearest is not a coin toss. _TAP is up in the top-left of the
# canvas: inside saloni, and clear of the panel, which is bottom-anchored and
# covers the lower half of the map while a tool is armed.
_CENTERS = {'saloni': [1.0, 4.0, 12.5], 'kouzina': [4.0, 1.0, 8.0]}
_TAP = {'x': 93, 'y': 112}
_TAP2 = {'x': 150, 'y': 60}


def _load_rooms(page, rooms=_ROOMS, centers=None):
    page.evaluate('a => HANDLERS.map({type:"map", width:100, height:100,'
                  'resolution:0.05, origin:[0,0], image:"", rooms:a[0],'
                  'centers:a[1], tinted:true})',
                  [rooms, _CENTERS if centers is None else centers])
    page.wait_for_timeout(30)


def _open(page, mode='select'):
    """Room manager open, on a known mode, nothing selected, outbox empty."""
    _load_rooms(page)
    page.evaluate("""mode => {
        setMapView('2d');
        setRoomMgr(true);
        selectedRoomName = null;
        rmSetMode(mode);
        window.__sent = [];
    }""", mode)
    page.wait_for_timeout(30)


# ── the panel itself ───────────────────────────────────────────────────────

def test_panel_is_closed_until_asked_for(page):
    _load_rooms(page)
    assert page.evaluate("$('room-mgr').style.display") == 'none'
    page.evaluate("setRoomMgr(true)")
    assert page.evaluate("$('room-mgr').style.display") != 'none'
    page.evaluate("setRoomMgr(false)")
    assert page.evaluate("$('room-mgr').style.display") == 'none'


def test_every_entry_point_opens_the_same_panel(page):
    """The 🏠 fab, the sheet's "Χωρισμός Δωματίων" tool and the button in the
    Δωμάτια card are three doors into one room manager."""
    for click in ('b-room-mgr', 'mes-rooms', 'b-room-mgr-open'):
        page.evaluate("setRoomMgr(false)")
        page.evaluate(f"$('{click}').click()")
        assert page.evaluate("roomMgrOn"), click
    page.evaluate("setRoomMgr(false)")


def test_one_card_per_room_with_its_area(page):
    _open(page)
    cards = page.evaluate("document.querySelectorAll('#rm-cards .rm-card').length")
    assert cards == len(_ROOMS)
    text = page.evaluate("$('rm-cards').textContent")
    assert 'saloni' in text and 'kouzina' in text
    assert '12.5 m²' in text and '8 m²' in text


def test_placeholder_names_are_shown_as_unnamed(page):
    """auto_rooms.py leaves room1/room2 behind — a placeholder, not a name."""
    _load_rooms(page, {'room1': [1, 2, 3]}, {'room1': [1.0, 4.0, 4.0]})
    page.evaluate("setRoomMgr(true)")
    assert 'room1' not in page.evaluate("$('rm-cards').textContent")
    _load_rooms(page)


def test_selecting_a_card_opens_that_rooms_editor(page):
    _open(page)
    page.click('#rm-cards .rm-card[data-room="kouzina"]')
    assert page.evaluate("selectedRoomName") == 'kouzina'
    assert page.evaluate("$('rm-color').value").lower() == '#ffcc00'
    assert page.evaluate("$('rm-name').value") == 'kouzina'


def test_a_preset_fills_the_name(page):
    _open(page)
    page.click('#rm-cards .rm-card[data-room="saloni"]')
    page.click('#rm-edit .rm-preset')
    assert page.evaluate("$('rm-name').value") == 'Σαλόνι'


def test_saving_sends_only_the_edited_room(page):
    _open(page)
    page.click('#rm-cards .rm-card[data-room="saloni"]')
    page.evaluate("""
        $('rm-name').value = 'σαλόνι';
        $('rm-color').value = '#112233';
        window.__sent = [];
    """)
    page.click('#rm-edit [data-act="save"]')
    sent = page.evaluate('window.__sent')
    assert len(sent) == 1 and sent[0]['type'] == 'save_rooms'
    assert sent[0]['rooms'] == [{'old': 'saloni', 'name': 'σαλόνι',
                                 'color': [0x11, 0x22, 0x33]}]


def test_a_blank_name_is_not_saved(page):
    _open(page)
    page.click('#rm-cards .rm-card[data-room="saloni"]')
    page.evaluate("$('rm-name').value = '  '; window.__sent = [];")
    page.click('#rm-edit [data-act="save"]')
    assert page.evaluate('window.__sent') == []
    assert page.evaluate("$('rm-msg').textContent")


def test_delete_asks_then_sends(page):
    _open(page)
    page.click('#rm-cards .rm-card[data-room="kouzina"]')
    page.evaluate("window.__sent = []; window.confirm = () => false;")
    page.click('#rm-edit [data-act="del"]')
    assert page.evaluate('window.__sent') == [], 'deleted without a confirmation'
    page.evaluate("window.confirm = () => true;")
    page.click('#rm-edit [data-act="del"]')
    assert page.evaluate('window.__sent') == [{'type': 'delete_room', 'name': 'kouzina'}]


def test_go_sends_the_robot_to_the_room_centre(page):
    _open(page)
    page.click('#rm-cards .rm-card[data-room="saloni"]')
    page.evaluate("window.__sent = [];")
    page.click('#rm-edit [data-act="go"]')
    assert page.evaluate('window.__sent') == [
        {'type': 'nav_goal', 'x': 1.0, 'y': 4.0}]


def test_divide_does_not_propose_the_original_rooms_name(page):
    """The name in the form is the NEW piece's; carrying "Σαλόνι" over from the
    editor of the room being cut is a duplicate the server rejects."""
    _open(page)
    page.click('#rm-cards .rm-card[data-room="saloni"]')
    assert page.evaluate("$('rm-name').value") == 'saloni'
    page.click('#rm-edit [data-act="split"]')
    assert page.evaluate("$('rm-name').value") != 'saloni'


# ── merge ──────────────────────────────────────────────────────────────────

def test_merge_keeps_the_first_room_and_folds_the_second_in(page):
    _open(page, 'merge')
    page.click('#rm-cards .rm-card[data-room="saloni"]')
    page.evaluate("window.__sent = [];")
    page.click('#rm-cards .rm-card[data-room="kouzina"]')
    assert page.evaluate('window.__sent') == [
        {'type': 'merge_rooms', 'names': ['saloni', 'kouzina']}]


def test_merge_needs_two_different_rooms(page):
    _open(page, 'merge')
    page.click('#rm-cards .rm-card[data-room="saloni"]')
    page.evaluate("window.__sent = [];")
    page.click('#rm-cards .rm-card[data-room="saloni"]')
    assert page.evaluate('window.__sent') == []


# ── the map's own taps ─────────────────────────────────────────────────────

def test_a_plain_map_click_still_navigates(page):
    _load_rooms(page)
    page.evaluate("setMapView('2d'); setRoomMgr(false); window.__sent = [];")
    page.click('#map-canvas', position=_TAP)
    sent = page.evaluate('window.__sent')
    assert sent and sent[0]['type'] == 'nav_goal'


def test_add_mode_paints_the_room_the_tap_landed_in(page):
    _open(page, 'add')
    page.evaluate("$('rm-name').value = 'μπάνιο'; $('rm-color').value = '#0000ff';")
    page.evaluate("window.__sent = [];")
    page.click('#map-canvas', position=_TAP)
    sent = page.evaluate('window.__sent')
    assert len(sent) == 1 and sent[0]['type'] == 'place_room'
    assert sent[0]['name'] == 'μπάνιο' and sent[0]['color'] == [0, 0, 255]


def test_add_mode_refuses_a_nameless_room(page):
    _open(page, 'add')
    page.evaluate("$('rm-name').value = ''; window.__sent = [];")
    page.click('#map-canvas', position=_TAP)
    assert page.evaluate('window.__sent') == []
    assert page.evaluate("$('rm-msg').textContent")


def test_select_mode_picks_the_room_under_the_tap(page):
    """With no map image to sample (image:"" never decodes), the nearest room
    centroid wins — _TAP is up in saloni's corner."""
    _open(page)
    page.click('#map-canvas', position=_TAP)
    assert page.evaluate("selectedRoomName") == 'saloni'
    assert page.evaluate('window.__sent') == [], 'selecting must not move the robot'


def test_a_tap_inside_a_room_picks_it_off_the_map_picture(page):
    """The real path (the centroid fallback above only runs with no picture):
    the server blends each room's colour into the floor at a known ratio, so
    the pixel under the finger is enough to name the room."""
    Image = pytest.importorskip('PIL.Image', reason='pillow not installed')
    tint = 0.62
    def mix(rgb):
        return tuple(int(255 * (1 - tint) + c * tint) for c in rgb)
    img = Image.new('RGB', (100, 100), (128, 128, 128))
    img.paste(mix(_ROOMS['saloni']), (2, 2, 48, 98))      # left half
    img.paste(mix(_ROOMS['kouzina']), (52, 2, 98, 98))    # right half
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    page.evaluate('a => HANDLERS.map({type:"map", width:100, height:100,'
                  'resolution:0.05, origin:[0,0], image:a[2], rooms:a[0],'
                  'centers:a[1], tinted:true})',
                  [_ROOMS, {'saloni': [3.6, 1.0, 12.5], 'kouzina': [0.5, 1.0, 8.0]},
                   base64.b64encode(buf.getvalue()).decode()])
    page.wait_for_timeout(200)
    page.evaluate("setMapView('2d'); setRoomMgr(true); selectedRoomName = null;"
                  "rmSetMode('select'); window.__sent = [];")
    # Top of the canvas: clear of the panel, and far from both centroids (parked
    # at the bottom on purpose) so the fallback would answer "no room here" —
    # a pass can only come from the colour under the finger.
    page.click('#map-canvas', position={'x': 280, 'y': 60})
    assert page.evaluate("selectedRoomName") == 'kouzina'
    _load_rooms(page)   # restore the image-less map the other tests expect


def test_divide_takes_two_taps_and_then_sends_the_line(page):
    _open(page, 'split')
    page.evaluate("rmSelect('saloni'); $('rm-name').value = 'γραφείο';"
                  "$('rm-color').value = '#00ff00'; window.__sent = [];")
    page.click('#map-canvas', position=_TAP)
    assert page.evaluate('window.__sent') == [], 'sent before the line was drawn'
    assert page.evaluate('rmSplitPt !== null')
    page.click('#map-canvas', position=_TAP2)
    sent = page.evaluate('window.__sent')
    assert len(sent) == 1 and sent[0]['type'] == 'split_room'
    assert sent[0]['name'] == 'saloni' and sent[0]['new_name'] == 'γραφείο'
    assert sent[0]['color'] == [0, 255, 0]
    assert {'x1', 'y1', 'x2', 'y2'} <= set(sent[0])
    assert (sent[0]['x1'], sent[0]['y1']) != (sent[0]['x2'], sent[0]['y2'])


# ── acks and cross-tab updates ─────────────────────────────────────────────

def test_room_saved_ack_updates_the_status_line(page):
    _open(page)
    page.evaluate('HANDLERS.room_saved({type:"room_saved", ok:true})')
    assert 'θηκε' in page.evaluate("$('rm-msg').textContent").lower()
    page.evaluate('HANDLERS.room_saved({type:"room_saved", ok:false, error:"boom"})')
    assert page.evaluate("$('rm-msg').textContent") == 'boom'


def test_a_finished_add_falls_back_to_selection(page):
    """Otherwise the next tap on the map paints a second room with the same
    name, on top of a map that already moved on."""
    _open(page, 'add')
    page.evaluate('HANDLERS.room_saved({type:"room_saved", ok:true})')
    assert page.evaluate('rmMode') == 'select'


def test_a_save_from_another_tab_refreshes_the_cards(page):
    _open(page)
    page.evaluate("""HANDLERS.map_rooms({type:'map_rooms',
        rooms:{'diadromos':[68,68,255]}, centers:{'diadromos':[0.2,0.2,3.0]}})""")
    page.wait_for_timeout(30)
    assert page.evaluate(
        "document.querySelectorAll('#rm-cards .rm-card').length") == 1
    assert 'diadromos' in page.evaluate("$('rm-cards').textContent")
    _load_rooms(page)   # restore


def test_a_room_that_disappeared_clears_the_selection(page):
    _open(page)
    page.click('#rm-cards .rm-card[data-room="kouzina"]')
    page.evaluate("""HANDLERS.map_rooms({type:'map_rooms',
        rooms:{'saloni':[204,68,255]}, centers:{'saloni':[0.45,0.45,12.5]}})""")
    assert page.evaluate("selectedRoomName") is None
    _load_rooms(page)
