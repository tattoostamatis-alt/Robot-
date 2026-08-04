"""Tests for how the dashboard's canvas panels are sized (web_dashboard_node).

2026-08-04, reported from a laptop: "στο tab arm τον βραχίονα δεν τον βλέπω …
το βλέπω πολύ στενό", and "στο tab costmap μου τα βγάζει πολύ μεγάλα".

Both were layout, not rendering. Measured in Chromium at 1366x768:

  * The arm pane holds six cards in a flex column. Cards shrink by default, so
    the card holding the 300px viewer was squeezed to its 130px floor and
    showed the top quarter of the arm. ‼️ getBoundingClientRect on the CANVAS
    still returned the full 300 — the canvas was the right size, its card was
    not — which is why every check of the canvas itself came back clean.
  * The same crop hit #imu-rose (200px) and #pt-ring (120px), found by sweeping
    all 22 tabs for `canvas.bottom > card.bottom`.
  * armDraw fitted the arm to `min(w,h) * 1.25`, while its own comment said
    80%. A fully extended arm was therefore drawn taller than the canvas.
  * #cost-wrap was flex:1, so a 60x60 grid of a 3x3 m window was stretched
    across 1196x450: every cell became a ~20x7 px slab and a square window
    rendered as a wide rectangle.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_viewer_sizes.py -q
"""
import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()


def _rule(selector):
    """The body of a CSS rule, whitespace collapsed."""
    m = re.search(re.escape(selector) + r'\{(.*?)\}', _SRC, re.S)
    assert m, f'{selector} is gone from the stylesheet'
    return re.sub(r'\s+', '', m.group(1))


# ── cards must not be squeezed below their contents ──────────────────────────

def test_cards_do_not_shrink_below_their_contents():
    """‼️ THE BUG. The pane scrolls; a card that holds a viewer must not give
    away pixels it does not have."""
    assert 'flex-shrink:0' in _rule('.pane>.card:not(.grow)')


def test_the_growing_card_is_exempt():
    """.card.grow deliberately hands its space back — pinning it would undo the
    fix that keeps the map list visible under the network cards."""
    assert ':not(.grow)' in _SRC, 'the shrink floor swallowed .card.grow'
    assert 'flex:11auto' in _rule('.card.grow')      # 'flex:1 1 auto', despaced


# ── the arm viewer ───────────────────────────────────────────────────────────

def test_the_arm_canvas_is_sized_to_the_viewport():
    """A fixed 300px is a strip on a laptop. vh with a cap scales to the screen
    and still leaves the drag grip in charge."""
    m = re.search(r'id="arm3d"[^>]*style="([^"]*)"', _SRC, re.S)
    assert m, 'the arm canvas lost its inline size'
    style = re.sub(r'\s+', '', m.group(1))
    assert 'height:min(52vh,560px)' in style
    assert 'min-height:260px' in style, 'it must still be usable on a phone'


def test_the_arm_is_scaled_to_fit_inside_the_canvas():
    """The factor was 1.25 — 125% — while the comment claimed 80%. Anything
    over 1.0 draws a fully extended arm past the edge, whatever the centring
    pass does afterwards."""
    m = re.search(r'const scale = \(Math\.min\(w, h\) \* ([\d.]+) / armReach\)', _SRC)
    assert m, 'the arm scale line moved — check it still fits the canvas'
    assert float(m.group(1)) <= 1.0, 'the arm is drawn larger than the canvas'


# ── the costmap ──────────────────────────────────────────────────────────────

def test_the_costmap_stays_square():
    """3x3 m of 60x60 cells is a square. Stretching it to the pane's aspect
    lies about the shape of the space the planner sees."""
    css = _rule('#cost-wrap')
    assert 'aspect-ratio:1' in css
    assert 'width:auto' in css, 'width must follow height, or the grip squashes it'


def test_the_costmap_is_capped_and_centred():
    css = _rule('#cost-wrap')
    assert 'height:min(58vh,440px)' in css
    assert 'align-self:center' in css
    assert 'max-width:100%' in css, 'it must still fit a phone'
    assert 'flex:1;' not in css, 'flex:1 is what stretched it in the first place'
