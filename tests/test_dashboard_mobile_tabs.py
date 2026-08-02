"""Tests for the dashboard's tab bar on phones (web_dashboard_node).

2026-08-02, reported from an iPhone: "μου δείχνει μόνο την αρχική". The page was
fine — 12 tabs built, websocket open, map drawing, zero errors, confirmed in
WebKit (the engine Safari actually uses). It was the tab bar.

The mobile rule was one row with `overflow-x:auto`. On iOS Safari that is
invisible: the scrollbar is not drawn until a scroll is already under way, so
nothing suggests the row continues past the edge. Only 7 of 12 tabs fit on an
iPhone, which left Gazebo, Σύστημα, Log and Ρυθμίσεις unreachable — and the
dashboard looked like it only had a first page.

Wrapping to two rows shows all 12 with no gesture to discover. Verified by
tapping every tab in WebKit at 428x746, 390x664 and 320x568.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_mobile_tabs.py -q
"""
import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()

_MEDIA = re.search(r'@media\(max-width:760px\)\{(.*?)\n\}', _SRC, re.S)
assert _MEDIA, 'the mobile media query is gone or changed shape'
MOBILE = _MEDIA.group(1)


def _rule(selector):
    m = re.search(r'(?:^|\n)\s*%s\{([^}]*)\}' % re.escape(selector), MOBILE, re.S)
    assert m, f'the mobile rule for {selector} is gone'
    return m.group(1)


def test_the_tab_bar_wraps_instead_of_scrolling():
    """A horizontal scroller is undiscoverable on iOS — no visible scrollbar."""
    tabs = _rule('#tabs')
    assert 'flex-wrap:wrap' in tabs.replace(' ', ''), \
        'the tab bar no longer wraps — tabs past the edge become unreachable'


def test_the_tab_bar_does_not_scroll_horizontally():
    tabs = _rule('#tabs')
    assert 'overflow-x:auto' not in tabs.replace(' ', ''), \
        'overflow-x:auto is back; on iOS this hides the remaining tabs entirely'


def test_the_bar_can_grow_to_a_second_row():
    tabs = _rule('#tabs')
    assert re.search(r'height:\s*auto', tabs), \
        'a fixed height clips the wrapped row out of sight'


def test_the_tabs_are_sized_to_fit_six_per_row():
    """12 tabs / 2 rows. A wider basis reflows to 5 and leaves a ragged row;
    a narrower one fits 7 and wastes the wrap."""
    tab = _rule('.tab')
    m = re.search(r'flex:\s*1\s+0\s+([\d.]+)%', tab)
    assert m, '.tab no longer declares a flex basis, so widths are content-driven'
    basis = float(m.group(1))
    assert 15.0 <= basis <= 17.0, \
        f'basis {basis}% does not give six per row'


def test_long_labels_cannot_widen_a_cell():
    """Βραχίονας and Ρυθμίσεις are wider than 1/6 of an iPhone SE."""
    assert 'min-width:0' in _rule('.tab').replace(' ', ''), \
        'without min-width:0 a long label refuses to shrink and forces overflow'
    label = _rule('.tab>span:last-child')
    assert 'text-overflow:ellipsis' in label.replace(' ', ''), \
        'long labels must ellipsis rather than push the row wide'
    assert 'white-space:nowrap' in label.replace(' ', '')


def test_every_tab_is_still_in_the_bar():
    """The fix must not have been "show fewer tabs on mobile"."""
    m = re.search(r'const TABS = \[(.*?)\];', _SRC, re.S)
    assert m, 'the TABS table is gone'
    assert len(re.findall(r"\['", m.group(1))) == 12, \
        'the tab count changed — re-check that six-per-row still gives two even rows'
    assert 'display:none' not in _rule('.tab'), 'tabs are being hidden on mobile'


def test_the_active_tab_is_still_marked_on_top():
    """The desktop marks the active tab on its left edge; mobile on top."""
    assert 'border-top-color:#3b82f6' in _rule('.tab.active').replace(' ', '')


# ── the bar has to be inside the part of the screen you can see ─────────────
# Follow-up the same day: with the tabs wrapped and reachable in every headless
# iPhone viewport, the real phone still showed none of them. Playwright's iPhone
# viewports are the FULL screen; Safari's address bar is not modelled. And
# `100vh` on iOS is deliberately the height the page would have WITHOUT that bar
# — taller than what is visible. With the tab bar at the bottom
# (flex-direction:column-reverse) it sat underneath Safari's chrome, and
# html{overflow:hidden} meant it could not be scrolled to either.

def test_the_shell_is_sized_to_the_visible_viewport():
    m = re.search(r'#shell\{[^}]*\}', _SRC)
    assert m, 'the #shell rule is gone'
    assert '100dvh' in m.group(0), \
        '100vh on iOS includes the area behind Safari\'s bar — the tab bar hides there'


def test_the_vh_fallback_is_kept_and_comes_first():
    """Anything older than iOS 15.4 has no dvh; the earlier declaration wins
    there and the later one wins everywhere else."""
    rule = re.search(r'#shell\{([^}]*)\}', _SRC).group(1)
    assert '100vh' in rule, 'the vh fallback was dropped'
    assert rule.index('100vh') < rule.index('100dvh'), \
        'the dvh declaration must come second or it is the one that gets dropped'


def test_the_tab_bar_clears_the_home_indicator():
    tabs = _rule('#tabs').replace(' ', '')
    assert 'env(safe-area-inset-bottom' in tabs, \
        'the bottom row sits under the home indicator on a notched iPhone'


def test_safe_area_insets_can_actually_resolve():
    """env(safe-area-inset-*) is 0 unless the viewport opts into the full screen.

    NB: the file holds two viewport metas — this one and the VNC viewer page's.
    Match on user-scalable=no, which only the dashboard sets.
    """
    metas = re.findall(r'<meta name="viewport" content="([^"]*)"', _SRC)
    assert metas, 'the viewport meta is gone'
    dash = [m for m in metas if 'user-scalable=no' in m]
    assert len(dash) == 1, f'expected one dashboard viewport meta, found {len(dash)}'
    assert 'viewport-fit=cover' in dash[0], \
        'without viewport-fit=cover the safe-area inset above is always 0'


def test_the_page_is_not_cached():
    """The CSS and JS are inlined in this page, so a cached copy is a cached
    build — a layout fix looks like no fix at all until website data is cleared."""
    m = re.search(r'async def index\(.*?\n\n\n', _SRC, re.S)
    assert m, 'the index route changed shape'
    assert "'Cache-Control': 'no-store'" in m.group(0), \
        'the dashboard page must not be heuristically cached by Safari'
