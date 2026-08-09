"""The safety tab, rendered in a real browser engine.

‼️ Every other dashboard test in this directory reads the page as TEXT. That is
how 21 tabs once shipped to a phone as a blank screen with 1137 tests green: a
top-level throw is an HTTP 200 and a dead page, and no amount of string
matching sees it. See [[feedback_dashboard_js_never_tested]].

So this file runs the page in WebKit — the engine on the user's phone — with a
listener on `pageerror`, and then checks the things that would actually be
broken for someone standing next to the robot:

  * the rows are built (they are generated from SAFETY_SPECS, so a typo in a
    data-key produces a card of labels with no controls at all);
  * a slider is disabled, not merely greyed, while nothing owns the parameter —
    a control that moves but writes to nobody is a lie about what is guarding
    the robot;
  * moving a slider sends exactly one message, on release rather than per pixel;
  * the doorway warning appears past the radius where the jambs meet.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_safety_tab.py -q
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

# The same substitutions index() makes, so what the browser parses here is what
# it parses in the flat. Values are stand-ins except the safety ones, which are
# the real table — this file is about that table.
def _usb_devices():
    """USB_DEVICES out of the node module, without importing rclpy."""
    body = re.search(r'USB_DEVICES = (\{.*?\n\})', _SRC, re.S).group(1)
    return dict(re.findall(r"'(\w+)':\s*'([^']*)'", body))


_SUBS = {
    '__ROOMS__': json.dumps(['saloni', 'kouzina']),
    '__TOKEN_QS__': json.dumps(''),
    # Pulled out of the module rather than retyped: the arm pane indexes every
    # joint at load, so a short table here throws before the safety rows build.
    '__ARM_LIMITS__': re.search(r'ARM_LIMITS = (\{.*?\n\})', _SRC, re.S).group(1)
                        .replace("'", '"').replace(',\n}', '\n}'),
    '__ARM_JOINTS__': re.search(r'ARM_JOINTS = (\[.*?\])', _SRC, re.S).group(1)
                        .replace("'", '"'),
    '__HAS_NOVNC__': 'false',
    # Kept in step with the node's own list rather than retyped: a device
    # added there and missing here is a page that throws at load.
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
    """One WebKit page with the dashboard loaded and its socket stubbed.

    connect() is left to fail on its own (there is no server here); what the
    tests need is the synchronous top-level wiring, which runs before any of
    that. send() is replaced so a slider's message can be inspected.
    """
    with playwright.sync_playwright() as pw:
        browser = pw.webkit.launch()
        pg = browser.new_page(viewport={'width': 390, 'height': 664})
        errors = []
        pg.on('pageerror', lambda e: errors.append(str(e)))
        pg.set_content(_page_html())
        pg.wait_for_timeout(400)
        # Only the socket may fail; anything else is a real load error.
        fatal = [e for e in errors if 'WebSocket' not in e and 'ws' not in e.lower()]
        assert not fatal, 'the dashboard threw on load:\n  ' + '\n  '.join(fatal)
        pg.evaluate('window.__sent = []; window.send = m => window.__sent.push(m);')
        yield pg
        browser.close()


def _show_safety(page):
    page.evaluate("showTab('safe')")
    page.wait_for_timeout(60)


# ── the rows exist at all ────────────────────────────────────────────────────

def test_the_safety_rows_do_not_reuse_map_tab_class_names(page):
    """‼️ They did, and the card rendered broken on a phone.

    `.srow`/`.slab` were already the map tab's speed sliders — a
    96px/1fr/74px grid. Every safety label was squeezed into that 96px column
    with its help paragraph beside it as a second column. Nothing threw, every
    test was green, and the tab was unusable at 390px. The check is on computed
    layout rather than on class names, because the failure was visual: the
    label must span the row, not sit in a narrow first column.
    """
    _show_safety(page)
    width = page.evaluate(
        "document.querySelector('#p-safe .sfrow[data-key=\"stop_distance\"] "
        ".sflab').getBoundingClientRect().width")
    row = page.evaluate(
        "document.querySelector('#p-safe .sfrow[data-key=\"stop_distance\"]')"
        ".getBoundingClientRect().width")
    assert width > row * 0.8, (
        f'the label is {width:.0f}px inside a {row:.0f}px row — it is being '
        'laid out by somebody else\'s grid again')


def test_the_help_text_is_below_the_label_not_beside_it(page):
    _show_safety(page)
    boxes = page.evaluate(
        "['.sflab', '.sfhelp'].map(s => document.querySelector("
        "'#p-safe .sfrow[data-key=\"stop_distance\"] ' + s)"
        ".getBoundingClientRect()).map(r => [r.top, r.left])")
    (lab_top, lab_left), (help_top, help_left) = boxes
    assert help_top > lab_top, 'the help text is on the label\'s line'
    assert abs(help_left - lab_left) < 2, 'the help text is in its own column'


def test_the_tab_is_in_the_bar(page):
    labels = page.evaluate(
        "Array.from(document.querySelectorAll('.tab')).map(t => t.dataset.pane)")
    assert 'safe' in labels, 'the safety tab did not render into the tab bar'


def test_every_spec_has_a_row_and_a_control(page):
    """A data-key typo leaves a card of labels with nothing to drag."""
    _show_safety(page)
    for spec in ss.SPECS:
        found = page.evaluate(
            "k => !!document.querySelector('[data-input=\"' + k + '\"]')", spec.key)
        assert found, f'{spec.key} has no control in the safety tab'


def test_no_row_is_left_without_a_spec(page):
    """The other direction: a row whose key was renamed sits there inert."""
    _show_safety(page)
    keys = page.evaluate(
        "Array.from(document.querySelectorAll('#p-safe .sfrow')).map(r => r.dataset.key)")
    assert set(keys) == {s.key for s in ss.SPECS}


def test_sliders_carry_the_spec_range(page):
    _show_safety(page)
    for spec in ss.SPECS:
        if spec.kind != 'double':
            continue
        got = page.evaluate(
            "k => { const e = document.querySelector('[data-input=\"'+k+'\"]');"
            "return [e.min, e.max, e.step]; }", spec.key)
        assert [float(v) for v in got] == [spec.lo, spec.hi, spec.step], spec.key


def test_the_contact_guards_are_checkboxes_not_sliders(page):
    _show_safety(page)
    for key in ('bump_stop', 'cliff_stop', 'wheel_drop_stop'):
        kind = page.evaluate(
            "k => document.querySelector('[data-input=\"'+k+'\"]').type", key)
        assert kind == 'checkbox', key


def test_the_fixed_limits_are_listed(page):
    _show_safety(page)
    text = page.evaluate("document.getElementById('sf-info').textContent")
    for row in ss.INFO_ONLY:
        assert row['source'] in text
        assert f"{row['value']:.2f}" in text


# ── what the page does with a state update ───────────────────────────────────

def _state(**over):
    payload = {}
    for spec in ss.SPECS:
        payload[spec.key] = {'set': spec.default, 'live': spec.default,
                             'nodes': len(spec.targets),
                             'total': len(spec.targets)}
    for key, patch in over.items():
        payload[key].update(patch)
    return payload


def test_a_knob_nobody_owns_is_disabled(page):
    """‼️ obstacle_safety_node is off in half the launch configurations. A
    slider that still moves would claim a guard that is not running."""
    _show_safety(page)
    page.evaluate('s => onSafety(s)', _state(stop_distance={'nodes': 0, 'live': None}))
    assert page.evaluate(
        "document.querySelector('[data-input=\"stop_distance\"]').disabled") is True
    assert page.evaluate(
        "document.querySelector('#p-safe .sfrow[data-key=\"stop_distance\"]')"
        ".classList.contains('off')") is True


def test_a_knob_that_is_owned_stays_usable(page):
    _show_safety(page)
    page.evaluate('s => onSafety(s)', _state())
    assert page.evaluate(
        "document.querySelector('[data-input=\"stop_distance\"]').disabled") is False


def test_a_half_applied_costmap_is_not_shown_as_applied(page):
    """One costmap configured and one not is the state the robot is in for the
    first seconds of every boot; the panel must not call that done."""
    _show_safety(page)
    page.evaluate('s => onSafety(s)',
                  _state(inflation_radius={'nodes': 1, 'live': None}))
    text = page.evaluate(
        "document.querySelector('[data-val=\"inflation_radius\"]').textContent")
    assert '→' not in text


def test_a_live_value_that_disagrees_is_shown_beside_the_asked_one(page):
    """The launch argument wins over the saved file, and the user has to be
    able to see that rather than trust the slider."""
    _show_safety(page)
    page.evaluate('s => onSafety(s)',
                  _state(stop_distance={'set': 0.80, 'live': 0.50}))
    text = page.evaluate(
        "document.querySelector('[data-val=\"stop_distance\"]').textContent")
    assert '0.80' in text and '0.50' in text and '→' in text


def test_agreeing_values_are_not_printed_twice(page):
    _show_safety(page)
    page.evaluate('s => onSafety(s)', _state())
    text = page.evaluate(
        "document.querySelector('[data-val=\"stop_distance\"]').textContent")
    assert '→' not in text


def test_the_badges_report_which_nodes_are_up(page):
    _show_safety(page)
    page.evaluate('s => onSafety(s)', _state(
        stop_distance={'nodes': 0, 'live': None},
        center_width={'nodes': 0, 'live': None},
        detector_timeout={'nodes': 0, 'live': None}))
    assert page.evaluate("document.getElementById('sf-cam-badge').textContent") \
        != page.evaluate("document.getElementById('sf-base-badge').textContent")


def test_a_checkbox_follows_the_reported_state(page):
    _show_safety(page)
    page.evaluate('s => onSafety(s)', _state(bump_stop={'set': False, 'live': False}))
    assert page.evaluate(
        "document.querySelector('[data-input=\"bump_stop\"]').checked") is False
    page.evaluate('s => onSafety(s)', _state())
    assert page.evaluate(
        "document.querySelector('[data-input=\"bump_stop\"]').checked") is True


# ── the doorway warning ──────────────────────────────────────────────────────

def test_the_doorway_warning_appears_past_the_threshold(page):
    """‼️ ~0.78 m doorways: past 0.30 m of inflation the two jambs meet and the
    planner reports no path through an opening that is clear on the map."""
    _show_safety(page)
    page.evaluate('s => onSafety(s)', _state(inflation_radius={'set': 0.45,
                                                              'live': 0.45}))
    assert page.evaluate(
        "document.querySelector('#p-safe .sfrow[data-key=\"inflation_radius\"]')"
        ".classList.contains('warned')") is True


def test_the_doorway_warning_is_absent_at_the_default(page):
    _show_safety(page)
    page.evaluate('s => onSafety(s)', _state())
    assert page.evaluate(
        "document.querySelector('#p-safe .sfrow[data-key=\"inflation_radius\"]')"
        ".classList.contains('warned')") is False


def test_too_short_a_block_time_warns_on_the_low_side(page):
    """bump_block_s is dangerous SMALL, not large: at 0 the robot rebounds and
    drives straight back into the same furniture."""
    _show_safety(page)
    page.evaluate('s => onSafety(s)', _state(bump_block_s={'set': 0.0, 'live': 0.0}))
    assert page.evaluate(
        "document.querySelector('#p-safe .sfrow[data-key=\"bump_block_s\"]')"
        ".classList.contains('warned')") is True


# ── what gets sent ───────────────────────────────────────────────────────────

def test_dragging_sends_nothing_until_release(page):
    """One parameter write per gesture. 'input' repaints, 'change' commits."""
    _show_safety(page)
    page.evaluate('window.__sent = []')
    page.evaluate(
        "const e = document.querySelector('[data-input=\"stop_distance\"]');"
        "e.value = 0.9; e.dispatchEvent(new Event('input'));")
    assert page.evaluate('window.__sent.length') == 0
    page.evaluate(
        "document.querySelector('[data-input=\"stop_distance\"]')"
        ".dispatchEvent(new Event('change'));")
    sent = page.evaluate('window.__sent')
    assert sent == [{'type': 'safety_set', 'key': 'stop_distance', 'value': 0.9}]


def test_a_checkbox_sends_a_bool(page):
    _show_safety(page)
    page.evaluate('window.__sent = []')
    page.evaluate(
        "const e = document.querySelector('[data-input=\"bump_stop\"]');"
        "e.checked = false; e.dispatchEvent(new Event('change'));")
    assert page.evaluate('window.__sent') == \
        [{'type': 'safety_set', 'key': 'bump_stop', 'value': False}]


def test_an_update_does_not_yank_a_slider_out_of_a_finger(page):
    """A 3 s broadcast landing mid-drag must not snap the handle back."""
    _show_safety(page)
    page.evaluate(
        "const e = document.querySelector('[data-input=\"stop_distance\"]');"
        "e.focus(); e.value = 1.2;")
    page.evaluate('s => onSafety(s)', _state())
    assert float(page.evaluate(
        "document.querySelector('[data-input=\"stop_distance\"]').value")) == 1.2


def test_the_reset_button_asks_first(page):
    _show_safety(page)
    page.evaluate('window.__sent = []')
    page.once('dialog', lambda d: d.dismiss())
    page.click('#b-sf-reset')
    page.wait_for_timeout(80)
    assert page.evaluate('window.__sent') == [], \
        'reset fired without a confirmation'


def test_the_reset_button_sends_when_confirmed(page):
    _show_safety(page)
    page.evaluate('window.__sent = []')
    page.once('dialog', lambda d: d.accept())
    page.click('#b-sf-reset')
    page.wait_for_timeout(80)
    assert page.evaluate('window.__sent') == [{'type': 'safety_reset'}]


# ── it survives the language switch ──────────────────────────────────────────

def test_the_rows_are_translated_not_blanked(page):
    """The rows are built before applyLang() runs; switching language re-walks
    the page and must not wipe the generated controls."""
    _show_safety(page)
    page.evaluate("setLang('en')")
    page.wait_for_timeout(80)
    try:
        assert page.evaluate(
            "!!document.querySelector('[data-input=\"stop_distance\"]')")
        text = page.evaluate(
            "document.querySelector('#p-safe .sfrow[data-key=\"bump_stop\"]')"
            ".textContent")
        assert 'bumper' in text.lower()
    finally:
        page.evaluate("setLang('el')")
