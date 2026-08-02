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
import math
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


def _tab_count():
    m = re.search(r'const TABS = \[(.*?)\];', _SRC, re.S)
    assert m, 'the TABS table is gone'
    return len(re.findall(r"\['", m.group(1)))


# Narrowest a cell may get on a 320pt iPhone SE before the icon and the label
# collide. This is what caps a row at 7 tabs.
MIN_CELL_PT = 45
SE_WIDTH_PT = 320


def _expected_basis(n):
    """Fewest even rows whose cells still clear MIN_CELL_PT."""
    max_per_row = int(SE_WIDTH_PT // MIN_CELL_PT)      # 7
    rows = math.ceil(n / max_per_row)
    per_row = math.ceil(n / rows)
    return per_row, 100.0 / per_row


def test_the_tabs_are_sized_to_fill_even_rows():
    """Derived from the tab count rather than hard-coded, so adding a tab fails
    loudly with the basis to use instead of silently stranding one tab alone on
    a ragged last row (13 tabs at the old 16.66% wrapped 6+6+1) or squeezing a
    row so tight the labels collide (15 tabs in two rows is 40pt a cell)."""
    tab = _rule('.tab')
    m = re.search(r'flex:\s*1\s+0\s+([\d.]+)%', tab)
    assert m, '.tab no longer declares a flex basis, so widths are content-driven'
    basis = float(m.group(1))

    n = _tab_count()
    per_row, want = _expected_basis(n)
    assert abs(basis - want) < 0.6, (
        f'{n} tabs want {per_row} per row (basis ~{want:.2f}%), not {basis}% — '
        f'at {basis}% they wrap {math.floor(100 / basis)} per row')
    assert SE_WIDTH_PT / per_row >= MIN_CELL_PT, \
        f'{per_row} per row is {SE_WIDTH_PT/per_row:.0f}pt on an iPhone SE'


def test_the_rows_come_out_even():
    """A last row holding one lonely tab reads as a rendering fault."""
    n = _tab_count()
    per_row, _ = _expected_basis(n)
    rows = math.ceil(n / per_row)
    last = n - per_row * (rows - 1)
    assert last >= per_row - 1 or rows == 1, \
        f'{n} tabs at {per_row} per row leaves a last row of {last}'


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
    # Every tab in the table must also have a pane to show, or tapping it blanks
    # the page — the failure mode a bare count check would not catch.
    m = re.search(r'const TABS = \[(.*?)\];', _SRC, re.S)
    ids = re.findall(r"\['(\w+)'", m.group(1))
    assert len(ids) >= 12, 'tabs were removed rather than made to fit'
    for tab_id in ids:
        assert f'id="p-{tab_id}"' in _SRC, f'tab {tab_id!r} has no pane'
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
