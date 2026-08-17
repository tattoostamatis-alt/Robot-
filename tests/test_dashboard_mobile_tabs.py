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


def _all_tab_ids():
    m = re.search(r'const ALL_TABS = \[(.*?)\];', _SRC, re.S)
    assert m, 'the ALL_TABS table is gone'
    return re.findall(r"\['(\w+)'", m.group(1))


def _core_tab_ids():
    """The curated ids actually rendered into the bar — see CORE_TAB_IDS in
    web_dashboard_node.py. TABS itself is now `ALL_TABS.filter(...)`, not a
    literal array, so it can no longer be parsed directly the way ALL_TABS is."""
    m = re.search(r"const CORE_TAB_IDS = \[(.*?)\];", _SRC, re.S)
    assert m, 'CORE_TAB_IDS is gone — the curated bar is built some other way now'
    return re.findall(r"'(\w+)'", m.group(1))


# Narrowest a cell may get on a 320pt iPhone SE before the icon and the label
# collide. This is what caps a row at 7 tabs.
MIN_CELL_PT = 45
SE_WIDTH_PT = 320


def _basis():
    tab = _rule('.tab')
    m = re.search(r'flex:\s*(\d)\s+0\s+([\d.]+)%', tab)
    assert m, '.tab no longer declares a flex basis, so widths are content-driven'
    return int(m.group(1)), float(m.group(2))


def test_the_cells_do_not_stretch_to_fill_a_short_last_row():
    """‼️ flex-grow must be 0.

    With grow:1 a partly-filled last row spreads its cells across the full
    width. At 16 tabs the bottom four came out half again as wide as the twelve
    above them, and the bar read as a rendering fault rather than a tidy
    left-aligned remainder. Fixed basis keeps every cell identical."""
    grow, _ = _basis()
    assert grow == 0, \
        'flex-grow is back — a short last row will stretch and look broken'


def test_a_tab_stays_wide_enough_to_read_on_a_phone():
    """Below ~45pt on a 320pt iPhone SE the icon and the label collide. This is
    the only constraint the basis actually has to satisfy, so adding tabs grows
    the bar downward instead of squeezing it sideways."""
    _, basis = _basis()
    per_row = math.floor(100.0 / basis)
    cell = SE_WIDTH_PT * basis / 100.0
    assert cell >= MIN_CELL_PT, \
        f'basis {basis}% is {cell:.0f}pt on an iPhone SE — labels will collide'
    assert per_row >= 4, f'{per_row} tabs per row wastes the width'


def test_the_bar_does_not_grow_past_a_readable_number_of_rows():
    """Rows are free vertically only up to a point: past four the tab bar owns
    more of a phone screen than the content does. Counts the curated bar
    (CORE_TAB_IDS), not every tab the dashboard has — the rest live in
    Ρυθμίσεις → "Περισσότερα εργαλεία" precisely so the bar does not have to
    grow with every new tab."""
    n = len(_core_tab_ids())
    _, basis = _basis()
    per_row = math.floor(100.0 / basis)
    rows = math.ceil(n / per_row)
    assert rows <= 4, (
        f'{n} tabs at {per_row} per row needs {rows} rows — either widen the '
        'basis or the bar has outgrown a flat tab bar')


def test_long_labels_cannot_widen_a_cell():
    """Βραχίονας and Ρυθμίσεις are wider than 1/6 of an iPhone SE."""
    assert 'min-width:0' in _rule('.tab').replace(' ', ''), \
        'without min-width:0 a long label refuses to shrink and forces overflow'
    label = _rule('.tab>span:last-child')
    assert 'text-overflow:ellipsis' in label.replace(' ', ''), \
        'long labels must ellipsis rather than push the row wide'
    assert 'white-space:nowrap' in label.replace(' ', '')


def test_every_tab_still_has_a_reachable_pane():
    """The 2026-08-02 fix this file is named after was "wrap them all in the
    bar". A later pass deliberately supersedes that: only CORE_TAB_IDS render
    into the bar now, the rest moved to Ρυθμίσεις → "Περισσότερα εργαλεία"
    (MORE_TABS/renderMoreTools in web_dashboard_node.py) so the bar does not
    read as a cluttered 25-icon wall. What must still hold is the ORIGINAL
    guarantee this test protects — no tab silently becomes unreachable — just
    checked against "has a pane, and is in the bar or the more-tools list"
    instead of "is in the bar". A bare count check would not catch a tab
    quietly dropped from both.
    """
    all_ids = _all_tab_ids()
    assert len(all_ids) >= 12, 'tabs were removed rather than relocated'
    for tab_id in all_ids:
        assert f'id="p-{tab_id}"' in _SRC, f'tab {tab_id!r} has no pane'
    core_ids = _core_tab_ids()
    assert core_ids, 'the curated bar is empty'
    assert set(core_ids) <= set(all_ids), \
        'CORE_TAB_IDS references a tab id that does not exist in ALL_TABS'
    # MORE_TABS must be everything ALL_TABS minus CORE_TAB_IDS, by construction
    # — pinning the actual filter expression means a tab dropped from the core
    # list is provably still reachable, not just probably.
    assert re.search(
        r'const MORE_TABS = ALL_TABS\.filter\(\(\[id\]\) => '
        r'!CORE_TAB_IDS\.includes\(id\)\);', _SRC), \
        'MORE_TABS is no longer derived from ALL_TABS minus the bar — a tab ' \
        'moved out of the bar could now be reachable from nowhere'
    assert 'display:none' not in _rule('.tab'), 'tabs are being hidden on mobile'


def test_the_active_tab_is_still_marked_on_top():
    """The desktop marks the active tab with a pill highlight; mobile on top."""
    assert 'border-top-color:var(--accent)' in _rule('.tab.active').replace(' ', '')


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
