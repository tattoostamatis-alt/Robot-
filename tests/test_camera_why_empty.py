"""A working camera pointed at a wall must not look like a dead one.

Reported twice as "δεν δείχνει" while the stream ran at 23 fps: the robot was
parked 40 cm from a white wall, and every measure the dashboard had said the
frame was fine — brightness 116.7, a perfectly normal exposure. What gave it
away was texture: Laplacian variance 5.0, against >100 for an ordinary room.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_camera_why_empty.py -q
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

# ‼️ Two collisions live in these four lines, both of which only appear when
# the suite runs as a whole.
#
# First: whichever test imports cv2 FIRST decides which OpenCV every later one
# gets. Under a sourced ROS environment a plain `import cv2` picks up the
# system 4.6, and 4.6 cannot load the YuNet model — so importing it the lazy
# way made five face-detection tests fail with "Layer with requested id=-1 not
# found" while they passed in isolation. Hence the venv (4.11) first, the same
# as tests/test_face_detection.py does.
#
# Second: the venv must come straight back OUT. It carries its own playwright,
# pinned to a browser build that is not installed (it went looking for
# webkit-1530 against the 2336 on this machine), and leaving it on the path
# broke all 33 browser tests. Nothing here needs 4.11 — this file just must not
# be the one that chooses 4.6 for everybody else.
_VENV = '/home/dimi/ryzenai_venv/lib/python3.12/site-packages'
_added = os.path.isdir(_VENV) and _VENV not in sys.path
if _added:
    sys.path.insert(0, _VENV)
try:
    cv2 = pytest.importorskip('cv2')
finally:
    if _added:
        sys.path.remove(_VENV)

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()

_BODY = re.search(r'    JUDGE_SIZE.*?\n    def _cb_camera', _SRC, re.S).group(0)
_NS = {'cv2': cv2, 'np': np, 'time': __import__('time')}
exec(compile('class D:\n' + _BODY.rsplit('\n    def _cb_camera', 1)[0],  # noqa: S102
             'judge', 'exec'), _NS)
Judge = _NS['D']


class Fake(Judge):
    def __init__(self):
        self._cam_judge_at = 0.0
        self._cam_state_last = None
        self.sent = []

        class S:
            @staticmethod
            def broadcast(msg, remember=True):
                Fake._last.sent.append(msg)
        Fake._last = self
        self._state = S()

    def verdict(self, img):
        self._cam_judge_at = 0.0      # defeat the once-a-second throttle
        self.sent.clear()
        self._judge_frame(img)
        return self.sent[-1] if self.sent else None


def solid(value):
    return np.full((480, 640, 3), value, np.uint8)


def textured(mean=120):
    rng = np.random.default_rng(1)
    noise = rng.integers(-60, 60, (480, 640, 1))
    return np.clip(np.repeat(noise, 3, axis=2) + mean, 0, 255).astype(np.uint8)


def wall():
    """A white wall up close: normal exposure, gentle gradient, no texture.
    This is the real failure case — everything looks healthy except detail."""
    grad = np.linspace(105, 130, 640, dtype=np.float32)
    img = np.repeat(grad[None, :], 480, axis=0)
    return np.repeat(img[:, :, None], 3, axis=2).astype(np.uint8)


def test_a_normal_scene_says_nothing():
    assert Fake().verdict(textured())['state'] == 'ok'


def test_a_blank_wall_is_reported_as_a_wall_not_a_fault():
    v = Fake().verdict(wall())
    assert v['state'] == 'flat'
    # ‼️ And the exposure must read as fine — that is the whole trap. If this
    # frame tripped the dark or blown branch the message would send someone
    # after a camera fault instead of moving the robot.
    assert 18.0 < v['mean'] < 238.0


def test_darkness_is_told_apart_from_a_wall():
    v = Fake().verdict(solid(5))
    assert v['state'] == 'dark'


def test_a_blown_out_frame_is_told_apart_too():
    v = Fake().verdict(solid(250))
    assert v['state'] == 'blown'


def test_the_measured_wall_numbers_land_in_the_flat_band():
    """The live measurements that started this: brightness 116.7, and a
    Laplacian variance of 21.3 AT JUDGE_SIZE."""
    assert Judge.DARK_MEAN < 116.7 < Judge.BLOWN_MEAN
    assert 21.3 < Judge.FLAT_DETAIL


def test_the_threshold_is_calibrated_at_the_size_it_is_measured_at():
    """‼️ Laplacian variance scales with resolution — the same wall scored
    6.0 at 640x480, 21.3 at 320x240 and 73.3 at 160x120. The first attempt
    took the 640-wide number and applied it to a 160-wide copy, so a blank
    wall scored 73 against a threshold of 40 and the banner never fired."""
    g = cv2.cvtColor(cv2.resize(wall(), Judge.JUDGE_SIZE), cv2.COLOR_BGR2GRAY)
    at_judge_size = cv2.Laplacian(g, cv2.CV_64F).var()
    assert at_judge_size < Judge.FLAT_DETAIL, (
        f'a wall scores {at_judge_size:.1f} at {Judge.JUDGE_SIZE}, '
        f'threshold {Judge.FLAT_DETAIL}')
    # And the same wall at full resolution would score far lower still, which
    # is exactly why the two must not be mixed.
    full = cv2.cvtColor(wall(), cv2.COLOR_BGR2GRAY)
    assert cv2.Laplacian(full, cv2.CV_64F).var() < at_judge_size


def test_an_ordinary_room_is_well_clear_of_the_flat_threshold():
    v = Fake().verdict(textured())
    assert v['detail'] > Judge.FLAT_DETAIL * 2


def test_it_only_speaks_when_the_verdict_changes():
    f = Fake()
    f.verdict(wall())
    n = len(f.sent)
    f._cam_judge_at = 0.0
    f._judge_frame(wall())            # same verdict again
    assert len(f.sent) == n, 'a latched banner would be rebroadcast every second'
