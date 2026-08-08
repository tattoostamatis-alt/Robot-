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
    """RFB fires nothing at all while a socket sits open and quiet, so that
    case used to be indistinguishable from a slow one."""
    assert re.search(r'setTimeout\(\(\) => \{ if \(!connected', VIEW), \
        'no watchdog for a connection that never opens'


def test_the_watchdog_does_not_talk_over_the_disconnect_report():
    m = re.search(r'setTimeout\(\(\) => \{ if \(!connected.*?\}, 15000\);', VIEW, re.S)
    assert m and 'diagnosed' in m.group(0), 'both paths would report'


def test_a_failed_connection_reads_the_websocket_close_code():
    """1008 (token), 1011 (session down) and 1006 (something in the path ate
    the socket) need three different fixes, and RFB reports none of them."""
    assert 'function diagnose(' in VIEW
    for code in ('1008', '1011', '1006'):
        assert code in VIEW, f'close code {code} is not explained'


def test_the_diagnostic_socket_opens_at_most_once():
    """It runs on the failure path, which can fire repeatedly."""
    assert 'if (diagnosed)' in VIEW and 'diagnosed = true' in VIEW


def test_the_diagnostic_socket_cannot_hang():
    """A socket that neither opens nor closes would leave the pane silent —
    which is the symptom this whole page exists to remove."""
    m = re.search(r'function diagnose\(.*?\n\}', VIEW, re.S)
    assert m and re.search(r'setTimeout\(\(\) => finish\(', m.group(0)), \
        'the probe has no timeout'
    assert 'settled' in m.group(0), 'open/close/timeout could all report'


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


# ── the RFB socket handshake ─────────────────────────────────────────────────
# 2026-08-02, found by driving the page in a real Chromium: every VNC pane died
# at close code 1006 with
#
#   Response must not include 'Sec-WebSocket-Protocol' header
#   if not present in request: binary
#
# The bridge accepted with a hardcoded subprotocol='binary', but neither our
# viewer nor rfb.js ever offers one, and RFC 6455 §4.1 lets the server name
# only a subprotocol the client sent.

def _bridge():
    m = re.search(r'async def vnc_bridge\(.*?(?=\n\n# )', _SRC, re.S)
    assert m, 'the /vnc bridge is gone or changed shape'
    return m.group(0)


def test_the_bridge_never_invents_a_subprotocol():
    """Accepting with one the client did not offer fails the handshake."""
    accept = re.search(r'await ws\.accept\((.*?)\)', _bridge(), re.S)
    assert accept, 'the bridge no longer accepts the socket'
    args = accept.group(1)
    assert "subprotocol='binary')" not in args + ')', \
        'subprotocol is hardcoded again — Chromium rejects the handshake'
    assert 'offered' in args or 'subprotocols' in args, \
        'the accepted subprotocol must be derived from what the client offered'


def test_the_offered_subprotocols_come_from_the_request():
    assert re.search(r"ws\.scope\.get\(\s*'subprotocols'", _bridge()), \
        'the bridge must read the client\'s offer off the ASGI scope'


def test_binary_is_still_echoed_when_it_is_offered():
    """Older noVNC builds (and websockify clients) do ask for 'binary'."""
    assert re.search(r"'binary'\s+if\s+'binary'\s+in\s+offered", _bridge()), \
        'a client that offers binary must still get it back'


@pytest.mark.skipif(not os.path.isdir(NOVNC_DIR), reason='novnc not installed')
def test_the_installed_rfb_really_offers_no_subprotocol():
    """The fact that made the hardcoded 'binary' wrong. If a future noVNC goes
    back to offering one, the echo above keeps working either way."""
    src = Path(NOVNC_DIR, 'core', 'rfb.js').read_text()
    m = re.search(r'this\._wsProtocols = (.*?);', src)
    assert m, 'rfb.js changed shape — re-check what it offers'
    assert m.group(1).strip().endswith('|| []'), \
        f'rfb.js now defaults to {m.group(1)} — re-check the bridge'


def test_our_viewer_passes_no_wsProtocols():
    """The other half: the RFB options we hand rfb.js leave the default in."""
    assert 'wsProtocols' not in VIEW, \
        'the viewer now sets wsProtocols — the bridge echo must agree with it'


# ── fullscreen ───────────────────────────────────────────────────────────────
# 2026-08-08, asked from the phone: "στο tab rviz θέλω να μου το δείχνεις πιο
# μεγάλο το παράθυρο να το βλέπω καλύτερα στο κινητό".
#
# Measured first, in WebKit, before touching anything — RViz runs 1600x900, so
# the 16:9 image letterboxes into a portrait pane and fills only 43% of it
# (374x210 inside 374x486 on a 390pt iPhone). ‼️ The height was NEVER the
# constraint: 486px of pane sat unused. The WIDTH caps the scale, which is why
# "make the window bigger" cannot be answered by growing the pane, and why
# fullscreen is worth x1.62 in landscape but only x1.04 in portrait.

def test_fullscreen_is_css_first_not_the_native_api():
    """‼️ THE TRAP. iPhone Safari implements Element.requestFullscreen for
    <video> and nothing else, so a native-only button is dead on exactly the
    device this was requested for. The class is what does the work."""
    assert '.vnc-host.fs{' in _SRC, 'the CSS fullscreen rule is gone'
    rule = re.search(r'\.vnc-host\.fs\{(.*?)\}', _SRC, re.S).group(1)
    flat = re.sub(r'\s+', '', rule)
    assert 'position:fixed' in flat, 'fullscreen no longer pins to the viewport'
    assert 'inset:0' in flat


def test_the_native_call_is_optional_and_its_rejection_swallowed():
    """It is a bonus (it also hides the URL bar). A browser that refuses must
    leave the CSS fullscreen working, not throw into the console."""
    fn = _fn('enterVncFs')
    assert 'webkitRequestFullscreen' in fn, 'the Safari-prefixed form is gone'
    assert 'catch' in fn, 'a rejected requestFullscreen would surface as an error'


def test_there_is_a_way_back_out():
    """The tab bar is covered while fullscreen, so the pane must carry its own
    exit or the dashboard is unreachable until reload."""
    assert "'vnc-fs-exit'" in _fn('enterVncFs'), 'the exit button is never built'
    assert 'Escape' in _fn('enterVncFs'), 'Esc no longer exits'


def test_exit_undoes_everything_enter_did():
    fn = _fn('exitVncFs')
    assert "classList.remove('fs')" in fn
    assert 'removeEventListener' in fn, 'the Esc handler would leak per open'
    assert '.vnc-fs-exit' in fn, 'the exit button is never removed'


def test_leaving_fullscreen_by_the_browser_unpins_the_pane():
    """iOS swipe / Android back / Esc handled natively all fire
    fullscreenchange. Without this the host stays position:fixed over the
    whole dashboard with its exit button gone."""
    assert 'fullscreenchange' in _SRC, 'nothing listens for the browser exiting'
    assert 'webkitfullscreenchange' in _SRC, 'the Safari-prefixed event is gone'


def test_fullscreen_is_only_offered_once_something_paints():
    """Fullscreening the 'press start' placeholder is a black screen."""
    m = re.search(r"if \(mode === 'live'\) mk\(t\('⛶ Πλήρης οθόνη'\)", _SRC)
    assert m, 'the fullscreen button is no longer gated on a live session'


def test_the_exit_button_is_thumb_sized():
    """44pt is the iOS minimum that a thumb hits reliably; this button sits
    over a live RViz where a miss is a click into the 3D view."""
    rule = re.search(r'\.vnc-fs-exit\{(.*?)\}', _SRC, re.S)
    assert rule, 'the exit button lost its style'
    flat = re.sub(r'\s+', '', rule.group(1))
    assert 'min-width:44px' in flat and 'min-height:44px' in flat
