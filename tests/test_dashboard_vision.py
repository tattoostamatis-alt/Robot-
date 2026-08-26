"""Tests for the camera overlay and the costmap tab (web_dashboard_node).

Added 2026-08-02 with the features. Two of these exist because the same mistake
was made twice while writing them, and neither one breaks anything — they just
draw the wrong colour, which no test would otherwise notice and no user could
report as anything more useful than "it looks odd":

  cv2 takes BGR, CSS takes RGB. The detection box for non-people was written
  (255,170,60) meaning orange and rendered pale BLUE, under a caption promising
  orange. The costmap legend then took the BGR inflation colour across into CSS
  and showed it as GREEN — which is also the robot marker's colour, so the
  legend pointed at the wrong thing twice.

The overlay itself was verified by publishing synthetic detections (YOLO found
nothing in the real scene — confirmed by running the model directly on a
captured frame at conf>=0.25, so a blank picture proved nothing either way).

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_vision.py -q
"""
import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()


def _fn(name):
    m = re.search(r'\n    def %s\(self.*?(?=\n    def |\n    # ──)' % re.escape(name),
                  _SRC, re.S)
    assert m, f'{name}() is gone'
    return m.group(0)


# ── colours: the bug class that renders wrong and tests nothing ─────────────

def test_the_object_box_is_actually_orange():
    """cv2 is BGR. (255,170,60) is what 'orange' looks like written down wrong,
    and it paints pale blue."""
    m = re.search(r"colour = \(80, 220, 80\) if person else \((\d+), (\d+), (\d+)\)", _SRC)
    assert m, 'the detection box colours are gone or changed shape'
    b, g, r = (int(x) for x in m.groups())
    assert r > b, (f'BGR({b},{g},{r}) has more blue than red — that is a blue '
                   'box, but the caption under the video says orange')


def test_the_person_box_is_actually_green():
    m = re.search(r"colour = \((\d+), (\d+), (\d+)\) if person", _SRC)
    assert m, 'the person box colour is gone'
    b, g, r = (int(x) for x in m.groups())
    assert g > b and g > r, f'BGR({b},{g},{r}) is not green'


def test_the_legend_matches_what_the_costmap_paints():
    """The legend is CSS/RGB, the map is cv2/BGR. Inflation is painted with a
    high blue channel, so its swatch must be blue-ish, not green — and never
    the robot marker's green."""
    m = re.search(r"\['(#\w{6})', 'inflation'\]", _SRC)
    assert m, 'the inflation legend entry is gone'
    hexc = m.group(1)
    r, g, b = (int(hexc[i:i+2], 16) for i in (1, 3, 5))
    assert b > r, f'{hexc} is not the blue-ish inflation colour the map draws'
    assert not (g > r and g > b and b < 100), f'{hexc} reads as green'


# ── the overlay only claims what it can currently see ──────────────────────

def test_stale_detections_are_not_drawn():
    """The detector is behind use_perception and can simply stop. Without an
    age check the last boxes stay burned over a scene the person left."""
    body = _fn('_draw_overlay')
    assert body.count('_OVERLAY_MAX_AGE') >= 2, \
        'boxes and/or skeletons are drawn without checking how old they are'
    m = re.search(r'_OVERLAY_MAX_AGE\s*=\s*([\d.]+)', _SRC)
    assert m and 0.4 <= float(m.group(1)) <= 3.0, \
        'the overlay staleness window is not a sane fraction of a second'


def test_boxes_are_scaled_by_one_shared_factor():
    """The frame is resized aspect-preserving before drawing, so x and y share a
    ratio. Deriving them separately from img_w/img_h invites a few-percent
    stretch that reads as a mis-calibrated camera rather than a bug here."""
    body = _fn('_draw_overlay')
    assert 'def sc(' in body, 'the shared scale helper is gone'
    # Comments stripped: this rule is about the arithmetic, and the comment
    # right above it explains the rule by naming img_h.
    code = re.sub(r'#[^\n]*', '', body)
    assert 'img_h' not in code, \
        'img_h is back in the scaling maths — x and y must share one factor'


def test_the_skeleton_uses_the_full_coco_edge_list():
    m = re.search(r'_KP_EDGES = \[(.*?)\]\n', _SRC, re.S)
    assert m, 'the skeleton edge list is gone'
    edges = re.findall(r'\((\d+), (\d+)\)', m.group(1))
    assert len(edges) == 16, f'{len(edges)} edges — COCO-17 draws 16'
    assert max(int(a) for e in edges for a in e) == 16, 'edges index past keypoint 16'


def test_low_confidence_joints_are_skipped():
    """YOLO reports occluded joints at (0,0) with v~0; drawing them staples the
    skeleton to the top-left corner."""
    body = _fn('_draw_overlay')
    assert re.search(r'v[ab]? < 0\.5', body), \
        'keypoint visibility is not checked before drawing'


def test_camera_click_starts_goto_object_not_arm_only_pick():
    """A camera object is usually outside arm reach; the base must approach
    and stop short of it, not try to grasp immediately."""
    dispatch = re.search(r"elif t == 'pick':(.*?)(?=\n        elif t ==)", _SRC, re.S)
    assert dispatch, 'camera pick dispatch is gone'
    body = dispatch.group(1)
    assert "f'goto_object:{label}'" in body
    assert '_mission_pub.publish' in body
    assert '_pick_pub.publish' not in body


def test_camera_click_refuses_stale_boxes_and_living_targets():
    assert 'performance.now() - latestDetectionsAt > 1500' in _SRC
    click = re.search(r"\$\('cam'\)\.addEventListener\('click'.*?\n\}\);", _SRC, re.S)
    assert click, 'camera click handler is gone'
    assert "'person'" in click.group(0) and "'dog'" in click.group(0)
    # The server repeats the guard; a forged WebSocket must not bypass UI safety.
    dispatch = re.search(r"elif t == 'pick':(.*?)(?=\n        elif t ==)", _SRC, re.S)
    assert 'unsafe_grasps' in dispatch.group(1)


# ── costmap: cost is not occupancy ─────────────────────────────────────────

def test_the_cost_gradient_is_rendered_not_thresholded():
    """0..100 in a Nav2 costmap is a gradient, not free/occupied. Collapsing it
    loses the one thing the tab is for: seeing how far inflation reaches into a
    doorway."""
    body = _fn('_cb_costmap')
    assert re.search(r'grid > 0\) & \(grid < 99', body), \
        'the 1..98 inflation band is no longer coloured as a gradient'
    for special in ('grid == 99', 'grid == 100', 'grid == -1'):
        assert special in body, f'{special} is not given its own colour'


def test_the_costmap_encode_is_skipped_when_nobody_is_watching():
    body = _fn('_cb_costmap')
    assert 'if not self._costmap_on' in body, \
        'the costmap is encoded to PNG even with the tab closed'


def test_a_closed_tab_releases_the_costmap_stream():
    """A browser closed on the tab never sends its 'off'."""
    m = re.search(r'def release_client\(self, client\):(.*?)\n    def ', _SRC, re.S)
    assert m, 'release_client is gone'
    assert '_costmap_on.discard' in m.group(1), \
        'a vanished browser leaves the costmap encoding for nobody'


def test_the_robot_marker_comes_from_tf_not_the_window_centre():
    """The local costmap rolls with the robot but is not exactly centred on it;
    assuming the middle puts the marker centimetres out, which is the difference
    between a near miss and a clear pass."""
    body = _fn('_cb_costmap')
    assert 'lookup_transform' in body, \
        'the robot position is not taken from TF'
    assert 'origin.position.x' in body, \
        'the costmap origin is ignored when placing the robot'


# ── following ──────────────────────────────────────────────────────────────

def test_the_follow_button_publishes_the_tool_topic():
    """Same topic the llm_bridge 'follow' tool uses, so the button and
    "ακολούθησέ με" cannot drift apart."""
    m = re.search(r"_follow_pub = self\.create_publisher\(Bool, '([^']+)'", _SRC)
    assert m and m.group(1) == '/follow_command', \
        'the follow button no longer drives /follow_command'


def test_stopping_the_follow_also_zeroes_the_wheels():
    """follow_command=False asks the follower to stop, but the last twist it
    sent is still the standing order until its own watchdog fires."""
    m = re.search(r"elif t == 'follow':(.*?)elif t ==", _SRC, re.S)
    assert m, 'the follow dispatch is gone'
    assert '_vel_pub.publish(Twist())' in m.group(1), \
        'stopping a follow does not command zero velocity'
