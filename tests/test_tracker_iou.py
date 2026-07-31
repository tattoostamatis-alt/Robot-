"""The IoU matrix must survive an empty side — that is the normal first frame.

tracker_node builds an (N tracks × M detections) IoU matrix. On the very first
frame that contains anything there are detections but no tracks yet, so N=0.
np.array([]) is one-dimensional, so indexing it as (N,4) raised

    IndexError: too many indices for array: array is 1-dimensional

and killed the node outright. It died that way on 2026-07-31 the moment the
first object came into view, which is to say the tracker had never actually
run with anything in front of the camera.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_tracker_iou.py -q
"""
import os
import re
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKER = f'{PKG}/home_robot/nodes/tracker_node.py'


def _iou_batch():
    """Pull the helper out of the node so no ROS or camera is needed."""
    src = open(TRACKER).read()
    fn = re.search(r'^def _iou_batch.*?(?=^\w|\Z)', src, re.S | re.M).group(0)
    ns = {'np': np}
    exec(compile(fn, 'iou', 'exec'), ns)  # noqa: S102
    return ns['_iou_batch']


def test_no_tracks_yet_gives_an_empty_matrix():
    m = _iou_batch()([], [[0, 0, 10, 10]])
    assert m.shape == (0, 1)


def test_no_detections_gives_an_empty_matrix():
    m = _iou_batch()([[0, 0, 10, 10]], [])
    assert m.shape == (1, 0)


def test_both_empty_is_still_well_formed():
    assert _iou_batch()([], []).shape == (0, 0)


def test_identical_boxes_score_one():
    m = _iou_batch()([[0, 0, 10, 10]], [[0, 0, 10, 10]])
    assert m[0, 0] == pytest.approx(1.0, abs=1e-4)


def test_disjoint_boxes_score_zero():
    m = _iou_batch()([[0, 0, 10, 10]], [[20, 20, 30, 30]])
    assert m[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_half_overlap_scores_a_third():
    # Two 10×10 boxes sharing half their area: intersection 50, union 150.
    m = _iou_batch()([[0, 0, 10, 10]], [[5, 0, 15, 10]])
    assert m[0, 0] == pytest.approx(1 / 3, abs=1e-3)


def test_shape_is_tracks_by_detections():
    m = _iou_batch()([[0, 0, 1, 1], [2, 2, 3, 3]],
                     [[0, 0, 1, 1], [5, 5, 6, 6], [7, 7, 8, 8]])
    assert m.shape == (2, 3)
