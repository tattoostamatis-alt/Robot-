"""Three things the log tab was reporting on 2026-08-05, and what they were.

Read off a live `robot max` at 07:26:

    WARN  [amcl] Failed to transform initial pose in time (Lookup would require
          extrapolation into the future ... 1785907599.219939 but the latest
          data is at 1785907599.165194)
    ERROR [camera_watchdog] Camera never streamed in 45s — the robot is BLIND
    WARN  [camera_watchdog] Restarting the camera (attempt 1/3)
    WARN  [camera.camera] Incomplete video frame detected! Size 226056 out of
          814335 bytes (27%), Frame Corrupted

The first is a dropped localization fix, not a warning: AMCL transforms an
initial pose AT the stamp it carries, and a now() stamp is 55 ms in the future
under boot load. The second recovered by itself 17 s later and the tab never
said so. The third is USB frame loss under load — real, unfixable from here,
and repeating often enough to bury everything else in the tab.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_log_tab_noise.py -q
"""
import json
import re
from pathlib import Path

import pytest

playwright = pytest.importorskip('playwright.sync_api',
                                 reason='playwright not installed')

_ROOT = Path(__file__).resolve().parents[1]
_NODES = _ROOT / 'home_robot' / 'nodes'
_DASH = (_NODES / 'web_dashboard_node.py').read_text()
_WATCHDOG = (_NODES / 'camera_watchdog_node.py').read_text()

_INITIALPOSE_PUBLISHERS = ['global_localizer_node.py', 'apriltag_relocalizer.py',
                           'pose_saver_node.py']


# ── 1. the initial pose AMCL refused ─────────────────────────────────────────

@pytest.mark.parametrize('name', _INITIALPOSE_PUBLISHERS)
def test_initialpose_is_stamped_zero_not_now(name):
    """‼️ `now()` asks TF for a transform in the future; AMCL answers "Failed to
    transform initial pose in time" and DROPS it. The scan match, the tag fix or
    the saved pose is thrown away and nothing here ever hears about it."""
    src = (_NODES / name).read_text()
    block = re.search(
        r'msg = PoseWithCovarianceStamped\(\)(.*?)frame_id', src, re.S)
    assert block, f'{name} no longer builds a PoseWithCovarianceStamped'
    stamp = re.search(r'header\.stamp\s*=\s*(.+)', block.group(1))
    assert stamp, f'{name} sets no stamp at all'
    assert 'Time()' in stamp.group(1), (
        f'{name} stamps /initialpose with {stamp.group(1).strip()!r} — a future '
        'stamp is refused by AMCL and the fix is lost in silence')


@pytest.mark.parametrize('name', _INITIALPOSE_PUBLISHERS)
def test_the_stamp_choice_is_explained_where_it_is_made(name):
    """This looks like a bug to the next reader ("why not now()?") and gets
    'fixed' back. The reason lives next to the line."""
    src = (_NODES / name).read_text()
    window = src.split('header.stamp')[0][-700:]
    assert 'extrapolation into the future' in window, \
        f'{name} no longer explains why the stamp is zero'


# ── 2. the camera that came back without saying so ───────────────────────────

def test_a_camera_that_never_streamed_can_report_dead_again():
    """‼️ The old if/elif meant a never-started camera brought back by a restart
    left `_reported_dead` True forever: no "Camera is back", and the NEXT
    failure would have been reported by nobody."""
    body = _WATCHDOG.split('def _on_frame', 1)[1].split('def _check', 1)[0]
    assert 'elif self._reported_dead' not in body, \
        'the recovery branch is unreachable again for a never-started camera'
    assert '_reported_dead = False' in body, \
        'nothing clears _reported_dead when frames come back'


def test_restarts_stop_counting_once_the_camera_stays_up():
    """Three hiccups spread over a day used to exhaust max_restarts and park
    the watchdog for an hour on a camera that recovered every single time."""
    assert 'restart_count_reset' in _WATCHDOG, \
        'the restart budget never recovers'
    body = _WATCHDOG.split('def _on_frame', 1)[1].split('def _check', 1)[0]
    assert 'self._restarts = 0' in body, \
        'the restart count is not cleared after a healthy run'


def test_the_reset_window_is_longer_than_the_restart_grace():
    """Otherwise a restart that only half worked buys itself a fresh budget."""
    reset = float(re.search(r"'restart_count_reset', ([\d.]+)", _WATCHDOG).group(1))
    grace = float(re.search(r"'restart_grace', ([\d.]+)", _WATCHDOG).group(1))
    assert reset > grace * 2, (reset, grace)


def test_the_restart_time_is_recorded_when_the_restart_happens():
    body = _WATCHDOG.split('self._restarts += 1', 1)[1][:300]
    assert '_last_restart_at' in body, \
        'the restart is counted but not timed, so the reset window never opens'


# ── 3. the repeating USB warning that buried the tab ─────────────────────────

def _usb_devices():
    """USB_DEVICES out of the node module, without importing rclpy."""
    body = re.search(r'USB_DEVICES = (\{.*?\n\})', _DASH, re.S).group(1)
    return dict(re.findall(r"'(\w+)':\s*'([^']*)'", body))


_SUBS = {
    '__ROOMS__': json.dumps(['saloni']),
    '__TOKEN_QS__': json.dumps(''),
    '__ARM_LIMITS__': re.search(r'ARM_LIMITS = (\{.*?\n\})', _DASH, re.S).group(1)
                        .replace("'", '"').replace(',\n}', '\n}'),
    '__ARM_JOINTS__': re.search(r'ARM_JOINTS = (\[.*?\])', _DASH, re.S).group(1)
                        .replace("'", '"'),
    '__HAS_NOVNC__': 'false',
    # Kept in step with the node's own list rather than retyped: a device
    # added there and missing here is a page that throws at load.
    '__USB_DEVICES__': json.dumps(_usb_devices(), ensure_ascii=False),
    '__I18N__': '{}',
    '__LANGS__': json.dumps([['el', 'Ελληνικά'], ['en', 'English'],
                             ['de', 'Deutsch']]),
}


def _page_html():
    from home_robot import safety_settings as ss
    html = _DASH.split('HTML_TEMPLATE = r"""', 1)[1].split('"""', 1)[0]
    subs = dict(_SUBS)
    subs['__SAFETY_SPECS__'] = json.dumps(
        {s.key: {'kind': s.kind, 'def': s.default, 'lo': s.lo, 'hi': s.hi,
                 'step': s.step, 'warn_above': s.warn_above,
                 'warn_below': s.warn_below} for s in ss.SPECS})
    subs['__SAFETY_INFO__'] = json.dumps(ss.INFO_ONLY)
    for token, value in subs.items():
        html = html.replace(token, value)
    assert not re.findall(r'__[A-Z_]+__', html)
    return html


@pytest.fixture(scope='module')
def page():
    with playwright.sync_playwright() as pw:
        browser = pw.webkit.launch()
        pg = browser.new_page()
        pg.set_content(_page_html())
        pg.wait_for_timeout(300)
        yield pg
        browser.close()


def _incomplete_frame(page, size):
    """The exact warning librealsense emits, byte count and all."""
    page.evaluate('m => addLog(m)', {
        'level': 30, 'name': 'camera.camera', 't': 1785908959.0,
        'text': f'XXX Hardware Notification:Incomplete video frame detected! '
                f'Size {size} out of 814335 bytes, Frame Corrupted'})


def _clear(page):
    page.evaluate("$('b-log-clear').onclick()")


def test_a_repeating_warning_folds_into_one_line(page):
    """‼️ The byte count differs every time, so folding on exact text would
    never match — five of these in 45 s were measured on the live robot."""
    _clear(page)
    for size in (686856, 256776, 355080, 588552, 226056):
        _incomplete_frame(page, size)
    assert page.evaluate("$('log-list').children.length") == 1


def test_the_folded_line_carries_the_count(page):
    _clear(page)
    for size in (686856, 256776, 355080):
        _incomplete_frame(page, size)
    assert '×3' in page.evaluate("$('log-list').lastChild.textContent")


def test_the_folded_line_shows_the_latest_numbers_not_the_first(page):
    """A run frozen on its first occurrence hides what is happening now."""
    _clear(page)
    _incomplete_frame(page, 686856)
    _incomplete_frame(page, 226056)
    text = page.evaluate("$('log-list').lastChild.textContent")
    assert '226056' in text and '686856' not in text


def test_the_counter_still_counts_every_message(page):
    """Folding is a display choice; the badge must not under-report."""
    _clear(page)
    for size in (1, 2, 3, 4):
        _incomplete_frame(page, size)
    assert page.evaluate("$('log-count').textContent") == '4'


def test_a_different_message_ends_the_run(page):
    _clear(page)
    _incomplete_frame(page, 686856)
    page.evaluate('m => addLog(m)', {
        'level': 40, 'name': 'camera_watchdog', 't': 1785908960.0,
        'text': 'Camera never streamed in 45s — the robot is BLIND'})
    _incomplete_frame(page, 226056)
    assert page.evaluate("$('log-list').children.length") == 3


def test_the_same_text_from_a_different_node_does_not_fold(page):
    _clear(page)
    for name in ('camera.camera', 'camera_watchdog'):
        page.evaluate('m => addLog(m)', {
            'level': 30, 'name': name, 't': 1785908959.0, 'text': 'same words'})
    assert page.evaluate("$('log-list').children.length") == 2


def test_a_warning_and_an_error_with_the_same_text_do_not_fold(page):
    _clear(page)
    for level in (30, 40):
        page.evaluate('m => addLog(m)', {
            'level': level, 'name': 'n', 't': 1785908959.0, 'text': 'same words'})
    assert page.evaluate("$('log-list').children.length") == 2


def test_folding_still_works_once_the_tab_is_full(page):
    """The tab keeps 300 lines and drops the oldest. Folding has to keep
    working across that trim — this is the state the tab is in after a few
    minutes of a `robot max`, not a corner case."""
    _clear(page)
    _incomplete_frame(page, 111)
    # The node name is part of the fold key and is NOT digit-masked, so these
    # stay 320 distinct lines. (Varying only the text would not: 'message 1'
    # and 'message 2' share a shape and would fold into one, which is exactly
    # the behaviour the other tests are here for.)
    for i in range(320):
        page.evaluate('m => addLog(m)', {
            'level': 30, 'name': f'filler{i}', 't': 1785908959.0,
            'text': 'unique message'})
    before = page.evaluate("$('log-list').children.length")
    assert before == 300, 'the trim never kicked in — this test proves nothing'
    _incomplete_frame(page, 222)
    after = page.evaluate("$('log-list').children.length")
    assert after == before, 'the new line did not reach the list'
    assert '222' in page.evaluate("$('log-list').lastChild.textContent")


def test_clearing_the_tab_forgets_the_open_run(page):
    """Otherwise the first message after a clear folds into nothing."""
    _clear(page)
    _incomplete_frame(page, 111)
    _clear(page)
    _incomplete_frame(page, 111)
    assert page.evaluate("$('log-list').children.length") == 1
    assert '×' not in page.evaluate("$('log-list').lastChild.textContent")
