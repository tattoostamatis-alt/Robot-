"""Which detection the arm commits to — pick_place_node._select_target.

Written after 2026-08-06, when «μπροστά σου είναι μια παντόφλα, άπλωσε τον
βραχίονα να την πιάσεις» produced «δεν βλέπω κάτι να σηκώσω» with the slipper
in plain view. The arm was never the problem: it had obeyed an `arm` command two
minutes earlier. Nothing could see a slipper, because object_detector only ever
emits the COCO 80 and footwear is not among them.

The method is exercised unbound, against a stand-in object, so target selection
stays testable without rclpy, a camera or an arm.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.nodes.pick_place_node import PickPlaceNode  # noqa: E402


def _node(coco=None, open_vocab=None):
    """A stand-in carrying only what _select_target/_graspable read."""
    return types.SimpleNamespace(
        _latest_objects=coco,
        _latest_open_vocab=open_vocab,
        _graspable=lambda objs: PickPlaceNode._graspable(None, objs),
    )


def _det(label, clutter=False, xyz=(0.1, 0.0, 0.4)):
    x, y, z = xyz
    return {'label': label, 'clutter': clutter, 'x': x, 'y': y, 'z': z}


select = PickPlaceNode._select_target


# ── the reported failure ──────────────────────────────────────────────────────

def test_an_open_vocabulary_hit_is_graspable():
    """The regression. open_vocab_detector marks its hits clutter=False (its
    confidences are not comparable to COCO's), and the old filter kept only
    clutter=True — so a slipper the detector was actively reporting could never
    be selected."""
    node = _node(coco=[_det('chair')], open_vocab=[_det('slipper')])
    target = select(node, 'slipper')
    assert target is not None, 'the slipper was visible and still not selected'
    assert target['label'] == 'slipper'


def test_a_named_miss_is_reported_as_a_miss():
    """It used to fall back to "any clutter object": asking for a slipper could
    have the robot confidently pick up an unrelated cup. Saying it cannot see
    the thing is the honest answer."""
    node = _node(coco=[_det('cup', clutter=True)])
    assert select(node, 'slipper') is None


def test_a_named_coco_object_still_wins():
    node = _node(coco=[_det('book', clutter=True), _det('cup', clutter=True)])
    assert select(node, 'cup')['label'] == 'cup'


# ── no label: "whatever clutter", used by sort/tidy ───────────────────────────

def test_without_a_label_it_takes_clutter_only():
    """Unchanged behaviour: {} means "tidy something up", and a chair or a
    sofa is not something to pick up and carry away."""
    node = _node(coco=[_det('chair'), _det('bottle', clutter=True)])
    assert select(node, None)['label'] == 'bottle'


def test_without_a_label_and_no_clutter_there_is_no_target():
    assert select(_node(coco=[_det('chair')]), None) is None


def test_an_unprompted_open_vocab_hit_is_not_tidied_away():
    """Open-vocab hits answer a specific request. Sweeping them into the
    no-label case would have `tidy` grab whatever the last `find` looked for."""
    node = _node(coco=[_det('chair')], open_vocab=[_det('slipper')])
    assert select(node, None) is None


# ── detections without depth ──────────────────────────────────────────────────

def test_a_detection_without_depth_is_never_a_target():
    """open_vocab_detector reports x/y/z as None when depth was missing for the
    box. Passing one on would either fail in the TF lookup or, worse, be read
    as a point at the camera origin and send the arm there."""
    blind = {'label': 'slipper', 'clutter': False,
             'x': None, 'y': None, 'z': None}
    assert select(_node(open_vocab=[blind]), 'slipper') is None


def test_a_located_duplicate_is_preferred_over_a_blind_one():
    blind = {'label': 'slipper', 'clutter': False,
             'x': None, 'y': None, 'z': None}
    node = _node(open_vocab=[blind, _det('slipper', xyz=(0.2, 0.0, 0.5))])
    assert select(node, 'slipper')['x'] == 0.2


def test_nothing_seen_at_all_is_not_an_error():
    assert select(_node(), 'slipper') is None
    assert select(_node(), None) is None


# ── ranking: the phantoms seen on the live 2026-08-06 run ─────────────────────

def test_the_most_confident_detection_wins():
    """Live run: the real slipper came back at 0.90 m with conf 0.33, alongside
    four phantoms on a far wall at ~3 m. First-seen would have aimed the arm at
    a wall; most-confident picks the slipper."""
    phantom = _det('slipper', xyz=(0.41, -0.49, 2.99))
    phantom['conf'] = 0.23
    real = _det('slipper', xyz=(-0.15, 0.20, 0.90))
    real['conf'] = 0.33
    assert select(_node(open_vocab=[phantom, real]), 'slipper')['z'] == 0.90


def test_ranking_also_applies_without_a_label():
    low = _det('cup', clutter=True, xyz=(0.1, 0.0, 0.4))
    low['conf'] = 0.2
    high = _det('book', clutter=True, xyz=(0.2, 0.0, 0.3))
    high['conf'] = 0.9
    assert select(_node(coco=[low, high]), None)['label'] == 'book'


def test_a_detection_without_a_confidence_still_selects():
    """conf is absent from some detector paths; it must not raise."""
    assert select(_node(open_vocab=[_det('slipper')]), 'slipper') is not None
