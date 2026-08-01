"""Tests for the dashboard's noVNC pane (web_dashboard_node).

noVNC hides its failures *inside* the iframe: uncaught script errors go to a
red `#noVNC_fallback_error` banner, refused connections to `#noVNC_status`.
Neither is visible on a scaled-down view, so the user reports "the VNC has an
error" and there is nothing to go on. The pane now mirrors that text out.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_vnc.py -q
"""
import os
import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()

NOVNC_DIR = '/usr/share/novnc'


def _fn(name):
    m = re.search(r'\nfunction %s\(.*?\n\}' % name, _SRC, re.S)
    assert m, f'{name}() is gone or changed shape'
    return m.group(0)


# ── the error has to escape the iframe ───────────────────────────────────────

def test_the_frame_is_watched_once_it_loads():
    """Without this the only symptom is a blank pane."""
    body = _fn('renderVnc')
    assert 'vncPeekError' in body, 'nothing reads noVNC back'
    assert "addEventListener('load'" in body, 'the peek must wait for the frame'


def test_both_of_novncs_error_surfaces_are_read():
    """A script error and a refused connection land in different elements;
    reading only one of them loses half the failures."""
    body = _fn('vncPeekError')
    assert 'noVNC_fallback_errormsg' in body, 'script errors are missed'
    assert 'noVNC_status' in body, 'connection errors are missed'


def test_only_an_open_banner_counts():
    """Both elements exist in the DOM at all times; their text is only real
    once noVNC adds the open class, or every healthy session reports an error."""
    body = _fn('vncPeekError')
    assert body.count('noVNC_open') >= 2
    assert 'noVNC_status_error' in body, 'a plain status is not a failure'


@pytest.mark.skipif(not os.path.isdir(NOVNC_DIR), reason='novnc not installed')
def test_the_element_ids_match_the_installed_novnc():
    """These ids are noVNC's, not ours — a package upgrade can rename them and
    the mirror would silently go quiet."""
    html = Path(NOVNC_DIR, 'vnc.html').read_text()
    for el in ('noVNC_fallback_error', 'noVNC_fallback_errormsg', 'noVNC_status'):
        assert f'id="{el}"' in html, f'{el} is not in this noVNC'


# ── the mirror must not become its own bug ───────────────────────────────────

def test_the_message_is_inserted_as_text_not_markup():
    """noVNC pastes a raw stack trace in there; innerHTML would eat the angle
    brackets and hand the page a script tag from whatever threw."""
    body = _fn('showVncError')
    assert 'textContent' in body
    assert 'innerHTML' not in body


def test_the_banner_is_not_stacked_on_every_poll():
    body = _fn('showVncError')
    assert "querySelector('.vnc-err')" in body, 'duplicates would pile up'


def test_polling_gives_up_instead_of_running_forever():
    """A tab left open for a day must not keep a 2 Hz timer alive."""
    body = _fn('vncPeekError')
    assert 'tries' in body and 'clearInterval' in body
    assert re.search(r'tries\s*>\s*\d+', body), 'the poll is unbounded'


def test_a_rebuilt_pane_cancels_the_old_poll():
    """renderVnc throws the iframe away; a surviving interval would then peek
    into a detached document forever."""
    assert 'clearInterval(st.peek)' in _fn('renderVnc')


def test_a_blocked_document_stops_the_poll():
    """contentDocument throwing means we can never read it — spinning at 2 Hz
    on an exception for 20 s is pure waste."""
    body = _fn('vncPeekError')
    catch = body.split('catch')[1]
    assert 'clearInterval' in catch and 'return' in catch


def test_the_style_exists_for_the_banner():
    assert '.vnc-err{' in _SRC, 'the mirrored text would be invisible'
