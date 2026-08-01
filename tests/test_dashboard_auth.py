"""Tests for the dashboard's token/cookie auth (web_dashboard_node).

Added 2026-08-01 after the URL's `?t=...` got lost (typed by hand, or the
terminal wrapped the line when it was copied) and the reply was a bare 401 —
which reads as "the dashboard is down", not "you dropped a query parameter".

`_authorised` is extracted rather than imported: importing the node pulls in
rclpy, cv2, uvicorn and FastAPI. Same approach as tests/test_tool_router.py.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_auth.py -q
"""
import re
import secrets
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()

_fn = re.search(r'^def _authorised.*?(?=\n\ndef |\Z)', _SRC, re.M | re.S)
assert _fn, 'could not extract _authorised from web_dashboard_node.py'

TOKEN = 'a-real-looking-token'


def _make(no_auth=False, token=TOKEN):
    ns = {'secrets': secrets, 'NO_AUTH': no_auth, 'TOKEN': token,
          'COOKIE_NAME': 'hr_dash', 'Optional': __import__('typing').Optional}
    exec(compile(_fn.group(0), 'dash_auth', 'exec'), ns)  # noqa: S102
    return ns['_authorised']


authorised = _make()


# ── the query token keeps working exactly as before ──────────────────────────

def test_the_query_token_still_authorises():
    assert authorised(TOKEN) is True


def test_a_wrong_token_is_refused():
    assert authorised('nope') is False


def test_no_token_at_all_is_refused():
    assert authorised('') is False
    assert authorised(None) is False


# ── the cookie is the new part ───────────────────────────────────────────────

def test_the_cookie_authorises_without_a_query_token():
    """The whole point: a bare http://host:8080/ works on the second visit."""
    assert authorised('', {'hr_dash': TOKEN}) is True


def test_a_wrong_cookie_is_refused():
    assert authorised('', {'hr_dash': 'forged'}) is False


def test_an_unrelated_cookie_is_ignored():
    assert authorised('', {'session': TOKEN}) is False


def test_an_empty_cookie_jar_is_refused():
    assert authorised('', {}) is False


def test_a_good_cookie_rescues_a_bad_query_token():
    """Stale bookmark with a rotated token in the URL, valid cookie in the
    browser — the browser wins rather than locking the user out."""
    assert authorised('old-token', {'hr_dash': TOKEN}) is True


# ── the escape hatch still escapes ───────────────────────────────────────────

def test_no_auth_mode_lets_everything_through():
    open_dash = _make(no_auth=True, token='')
    assert open_dash('') is True
    assert open_dash(None, {}) is True


# ── how it is wired into the app ─────────────────────────────────────────────

def test_every_entry_point_passes_the_cookie_jar():
    """A single endpoint left on the token-only check would 401 the tab that
    every other endpoint just let in."""
    calls = re.findall(r'_authorised\(([^)]*)\)', _SRC)
    calls = [c for c in calls if not c.startswith('supplied')]
    assert calls, 'no call sites found'
    for call in calls:
        assert 'cookies' in call, f'_authorised({call}) ignores the cookie'


def test_the_cookie_is_samesite_strict_and_httponly():
    """SameSite=strict is what stops another page on the LAN from opening a
    websocket to /ws with the browser's cookie and driving the robot —
    websockets are not subject to CORS."""
    setter = re.search(r'set_cookie\((.*?)\)', _SRC, re.S)
    assert setter, 'the cookie is never set'
    args = setter.group(1)
    assert "samesite='strict'" in args
    assert 'httponly=True' in args


def test_the_401_body_says_what_is_missing():
    """A blank 'Unauthorized' is what made this look like a dead server."""
    body = re.search(r"return HTMLResponse\(\s*'Unauthorized(.*?)status_code=401",
                     _SRC, re.S)
    assert body, 'the 401 body changed shape'
    assert 'token' in body.group(1)
