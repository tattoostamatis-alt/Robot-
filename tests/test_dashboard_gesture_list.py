"""Tests for the gesture-binding list (web_dashboard_node).

2026-08-04, from the owner: "στις χειρονομίες πάω να πατήσω μια χειρονομία να
επιλέξω τι θα κάνει και τρεμοπαίζει και δεν μπορώ να επιλέξω".

/gesture_status arrives at 15 Hz — measured — because it carries the live hold
progress for the ring. renderGestureBindings rebuilt `gb-list.innerHTML` on
every message, which destroys and recreates the <select> elements fifteen times
a second: an open dropdown is gone before you can reach an option, and the list
strobes. Nothing was wrong with the bindings themselves.

Fixed by building the rows once and only mutating them afterwards. Verified in
Chromium: the same DOM element survives 5 s of traffic, and a real
select_option still reads back four seconds (≈60 messages) later.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_gesture_list.py -q
"""
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()


def _fn(name):
    return _SRC.split(f'function {name}(')[1].split('\nfunction ')[0]


def test_the_hot_path_does_not_rebuild_the_list():
    """‼️ THE BUG. innerHTML at 15 Hz is what ate the dropdown."""
    body = _fn('renderGestureBindings')
    assert 'innerHTML' not in body, \
        'renderGestureBindings writes innerHTML again — the list will strobe'


def test_rebuilding_is_keyed_on_the_structure():
    """Only a changed set of gestures or actions justifies new DOM. Anything
    keyed on the live fields (holding, progress) fires at 15 Hz again."""
    body = _fn('renderGestureBindings')
    assert 'gbKey' in body
    assert 'JSON.stringify([rows, Object.keys(labels)])' in body
    for live in ('holding', 'progress'):
        assert f'{live}]' not in body.split('const key =')[1].split('\n')[0], \
            f'the rebuild key depends on {live}, which changes constantly'


def test_the_update_path_only_touches_style_and_values():
    body = _fn('gbUpdate')
    assert 'innerHTML' in body, 'the risky-badge span is set via innerHTML'
    assert 'gb-list' in body
    assert "$('gb-list').innerHTML =" not in body, 'the update path rebuilds the list'


def test_a_focused_select_is_never_overwritten():
    """‼️ Writing .value on a focused <select> closes its open dropdown — the
    same bug in miniature, and the reason the naive per-row update also fails."""
    body = _fn('gbUpdate')
    assert 'document.activeElement !== sel' in body


def test_the_value_is_only_written_when_it_differs():
    """An unconditional assignment at 15 Hz is a repaint per message even when
    nothing changed."""
    body = _fn('gbUpdate')
    assert 'sel.value !== cur' in body


def test_the_change_handler_is_attached_once_at_build():
    """Rebinding onchange per frame is the other half of the same waste."""
    build = _fn('gbBuild')
    update = _fn('gbUpdate')
    assert 'onchange' in build
    assert 'onchange' not in update


def test_the_binding_still_reaches_the_server():
    build = _fn('gbBuild')
    assert "type:'gesture_bind'" in build
    assert 'sel.dataset.gesture' in build
    assert 'sel.value' in build


def test_the_live_highlight_still_works():
    """The 15 Hz data is the point of the pane — it must still show which
    gesture is being held and how far along the hold is."""
    body = _fn('gbUpdate')
    assert 'holding' in body
    assert 'progress' in body
    assert 'gb-bar' in body


def test_unknown_gestures_still_appear():
    """A gesture a node added but GESTURE_ORDER does not list must not vanish."""
    body = _fn('renderGestureBindings')
    assert 'GESTURE_ORDER.indexOf(g) < 0' in body
