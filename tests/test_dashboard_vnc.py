"""Tests for the dashboard's VNC pane (web_dashboard_node).

2026-08-01: the RViz/MoveIt tabs showed a bare "Script error." — noVNC's own
`vnc.html` reporting a failure it could not name.  Two things caused that, and
both are properties of that page, so the dashboard stopped using it:

  * it loads its UI as `<script type="module" crossorigin="anonymous">`, and a
    module fetched in CORS mode has its errors MUTED unless the response
    carries Access-Control-Allow-Origin.  StaticFiles sends none, so anything
    thrown inside reaches window.onerror as the string "Script error." with no
    file, no line, no stack;
  * `app/webutil.js` reads settings straight out of localStorage with no
    try/catch, so a browser that denies site storage (Safari "Block All
    Cookies" / Lockdown Mode, private windows, strict tracking protection)
    kills the UI before it paints.

`/vncview/{app}` replaces it: core/rfb.js only — the protocol half, which
touches no storage — no crossorigin attribute, and every failure named out
loud and posted to the parent frame.

Source text is asserted on rather than imported: importing the node pulls in
rclpy, cv2, uvicorn and FastAPI.  Same approach as tests/test_dashboard_auth.py.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_vnc.py -q
"""
import os
import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()

NOVNC_DIR = '/usr/share/novnc'

_VIEW = re.search(r'VNC_VIEW_HTML = r"""(.*?)"""', _SRC, re.S)
assert _VIEW, 'VNC_VIEW_HTML is gone or changed shape'
VIEW = _VIEW.group(1)

_ROUTE = re.search(r'async def vnc_view\(.*?(?=\n\n# )', _SRC, re.S)
assert _ROUTE, 'the /vncview route is gone or changed shape'
ROUTE = _ROUTE.group(0)


def _fn(name):
    m = re.search(r'\nfunction %s\(.*?\n\}' % name, _SRC, re.S)
    assert m, f'{name}() is gone or changed shape'
    return m.group(0)


# ── the page that produced "Script error." is no longer loaded ───────────────

def _iframe_src():
    m = re.search(r'f\.src = .*?;', _fn('renderVnc'), re.S)
    assert m, 'renderVnc no longer sets an iframe src'
    return m.group(0)


def test_the_pane_loads_our_viewer_not_novncs_ui():
    src = _iframe_src()
    assert "'/vncview/'" in src, 'the pane must use our own viewer'
    assert 'vnc.html' not in src, 'this is the page that says "Script error."'


def test_the_viewer_never_asks_for_novncs_ui_layer():
    """app/ui.js is where the localStorage crash lives; core/rfb.js is not."""
    assert 'core/rfb.js' in VIEW
    assert 'app/ui.js' not in VIEW and 'webutil' not in VIEW


def test_the_viewer_touches_no_storage_itself():
    """Re-introducing localStorage here would rebuild the exact bug: on a
    browser that denies site storage the read throws and nothing paints."""
    assert 'localStorage' not in VIEW
    assert 'sessionStorage' not in VIEW


def test_no_script_is_fetched_in_cors_mode():
    """`crossorigin` is what mutes the error text into "Script error."."""
    assert 'crossorigin' not in VIEW


def test_the_novnc_import_is_dynamic_and_caught():
    """A static import that 404s leaves a dead frame and no message; `await
    import()` inside try/catch names the missing file instead."""
    assert 'await import(' in VIEW
    m = re.search(r'try \{\s*RFB = \(await import\(.*?\)\).*?catch \(e\) \{(.*?)\}',
                  VIEW, re.S)
    assert m, 'the noVNC import is not wrapped'
    assert '__vncReport' in m.group(1), 'a failed import says nothing'


# ── whatever goes wrong has to reach the user in words ───────────────────────

def test_the_error_handler_is_installed_before_anything_can_throw():
    """It must be a classic script, and first: a module is deferred, so a
    module-level throw would happen with no handler attached."""
    handler = VIEW.index("addEventListener('error'")
    module = VIEW.index('<script type="module">')
    assert handler < module, 'the module can throw before the handler exists'
    # The <script> that opens the handler block must carry no type=module: a
    # module is deferred and would run after the page's other scripts.
    opening = VIEW.rfind('<script', 0, handler)
    tag = VIEW[opening:VIEW.index('>', opening) + 1]
    assert tag == '<script>', f'the error handler must not be a module: {tag}'


def test_every_failure_mode_reports():
    """Each of these was a silent black rectangle before."""
    for hook in ("addEventListener('error'",
                 "addEventListener('unhandledrejection'",
                 "'securityfailure'",
                 "'credentialsrequired'",
                 "'disconnect'",
                 'nomodule'):
        assert hook in VIEW, f'{hook} goes unreported'


def test_a_connection_that_never_opens_still_says_so():
    """RFB retries internally and fires nothing the page can see, so a refused
    websocket used to be indistinguishable from a slow one."""
    assert re.search(r'setTimeout\(\(\) => \{ if \(!connected\)', VIEW), \
        'no watchdog for a connection that never opens'


def test_the_report_names_the_display_and_port():
    """"It does not work" is not actionable; ":2 (θύρα 5902)" is."""
    assert '__DISP__' in VIEW and '__PORT__' in VIEW
    assert 'gui_session.sh status' in VIEW, 'no way to check the session'


def test_only_the_first_error_is_reported():
    """A dropped connection fires several events; the later ones are fallout
    and would overwrite the one that explains the failure."""
    assert re.search(r"kind === 'error' && sent", VIEW), 'later noise wins'


# ── parent/frame handoff ─────────────────────────────────────────────────────

def test_the_frame_posts_to_its_own_origin_only():
    assert 'postMessage' in VIEW
    assert re.search(r'postMessage\(.*?,\s*\n?\s*location\.origin\)', VIEW, re.S), \
        'a wildcard target origin would leak the message'


def test_the_dashboard_checks_the_origin_of_what_it_renders():
    """The banner is written from a message; an unchecked listener would let
    any page that can reach this tab paint text into it."""
    m = re.search(r"window\.addEventListener\('message'.*?\n\}\);", _SRC, re.S)
    assert m, 'the message listener is gone'
    assert 'ev.origin !== location.origin' in m.group(0)
    assert "m.source !== 'vncview'" in m.group(0)


def test_a_working_connection_clears_the_banner():
    """Without this the reconnect message stays on screen over a live RViz."""
    assert "__vncReport('ok'" in VIEW
    m = re.search(r"window\.addEventListener\('message'.*?\n\}\);", _SRC, re.S)
    assert 'clearVncError' in m.group(0)


# ── the banner must not become its own bug ───────────────────────────────────

def test_the_message_is_inserted_as_text_not_markup():
    """The text can be a raw stack trace; innerHTML would hand the page
    whatever markup happened to be in it."""
    body = _fn('showVncError')
    assert 'textContent' in body
    assert 'innerHTML' not in body


def test_the_banner_is_not_stacked_on_every_message():
    body = _fn('showVncError')
    assert "querySelector('.vnc-err')" in body, 'duplicates would pile up'


def test_the_style_exists_for_the_banner():
    assert '.vnc-err{' in _SRC, 'the reported text would be invisible'


# ── reconnect ────────────────────────────────────────────────────────────────

def test_reconnect_is_bounded():
    """vnc.html had reconnect=1 and a phone that sleeps needs it, but a session
    that is really gone must not reload forever."""
    assert 'MAX_RETRY' in VIEW
    assert re.search(r'retry >= MAX_RETRY', VIEW), 'the retry loop is unbounded'


def test_the_retry_counter_does_not_live_in_storage():
    """It survives a reload, and the browsers this fix is for are exactly the
    ones that deny sessionStorage."""
    assert 'location.hash' in VIEW


def test_a_successful_reconnect_rearms_the_counter():
    """Otherwise the fourth drop of a long-lived tab is the last one retried."""
    m = re.search(r"addEventListener\('connect'.*?\n\}\);", VIEW, re.S)
    assert m and 'retry = 0' in m.group(0)


# ── the route ────────────────────────────────────────────────────────────────

def test_the_viewer_page_is_behind_the_token():
    """It carries the VNC password in its body."""
    assert '_authorised(t, request.cookies)' in ROUTE
    assert 'status_code=401' in ROUTE


def test_only_the_known_apps_are_served():
    assert 'app_name not in VNC_PORTS' in ROUTE


def test_the_password_no_longer_rides_in_the_iframe_url():
    """?password= ended up in history and in every referrer."""
    assert 'password' not in _iframe_src()
    assert 'VNC_PASSWORD' in ROUTE, 'the page must supply it server-side'


def test_the_page_is_not_cached():
    assert 'no-store' in ROUTE, 'a page holding a credential must not be cached'


def test_values_are_escaped_into_the_javascript():
    """A password with a quote in it would otherwise break the page — which,
    with no crossorigin, now reports as a syntax error rather than silence."""
    assert 'json.dumps' in ROUTE
    assert "'<\\\\/'" in ROUTE or '<\\\\/' in ROUTE, 'no </script> guard'


def test_the_frames_token_comes_from_TOKEN_not_the_query():
    """A caller let in by the cookie has no `t`, and the frame's own websocket
    would then arrive unauthenticated."""
    assert re.search(r"token_qs = '' if NO_AUTH else '\?t=' \+ quote\(TOKEN", ROUTE)


def test_a_missing_novnc_is_reported_not_blank():
    assert 'isdir(NOVNC_DIR)' in ROUTE


# ── what we depend on in the installed package ───────────────────────────────

@pytest.mark.skipif(not os.path.isdir(NOVNC_DIR), reason='novnc not installed')
def test_the_module_we_import_exists():
    """core/rfb.js is the whole dependency now; a package upgrade that moves it
    should fail here rather than in the browser."""
    assert Path(NOVNC_DIR, 'core', 'rfb.js').is_file()


@pytest.mark.skipif(not os.path.isdir(NOVNC_DIR), reason='novnc not installed')
def test_rfb_still_exports_a_default():
    src = Path(NOVNC_DIR, 'core', 'rfb.js').read_text()
    assert 'export default' in src


@pytest.mark.skipif(not os.path.isdir(NOVNC_DIR), reason='novnc not installed')
def test_the_installed_ui_really_has_the_unguarded_storage_read():
    """The reason this rewrite exists. If a future noVNC guards it, the stock
    page becomes usable again — but the CORS-muted errors would remain."""
    src = Path(NOVNC_DIR, 'app', 'webutil.js').read_text()
    m = re.search(r'export function readSetting.*?\n\}', src, re.S)
    assert m, 'webutil.js changed shape — re-check whether the bug is gone'
    assert 'localStorage.getItem' in m.group(0)
    assert 'try' not in m.group(0), 'noVNC now guards it; revisit this note'


@pytest.mark.skipif(not os.path.isdir(NOVNC_DIR), reason='novnc not installed')
def test_the_stock_page_still_mutes_its_own_errors():
    html = Path(NOVNC_DIR, 'vnc.html').read_text()
    assert re.search(r'<script type="module" crossorigin="anonymous"', html), \
        'vnc.html no longer mutes errors — revisit this note'
