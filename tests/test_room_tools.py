"""The map tab's room tools, server side: delete, merge, divide.

A room is a set of exactly-coloured pixels in maps/<map>_room_mask.png plus a
name -> [r,g,b] line in <map>_room_colors.yaml, and everything the dashboard's
room manager does is a rewrite of those two files. These check the rewrites
against a real mask on disk — the failure that matters is not an exception, it
is a mask that quietly comes out with the wrong pixels in it (room4 lost 5 of
its 8 rooms that way in production, see _save_rooms' comment).

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_room_tools.py -q
"""
import numpy as np
import pytest
import yaml
from PIL import Image

from home_robot import room_files
from home_robot.nodes.web_dashboard_node import DashboardNode

MAP = 'testmap'
# 20x20 px at 10 cm/px, origin at (0,0) — so map metres and mask columns are
# the same numbers divided by ten, which keeps the split lines below readable.
RES, ORIGIN = 0.1, (0.0, 0.0)
SALONI, KOUZINA, NEW = [200, 0, 0], [0, 200, 0], [0, 0, 200]


class _Node(DashboardNode):
    """The real room tools, without a ROS node under them. Only what those
    methods touch is stubbed: which map is active, the logger, and the
    broadcast/refresh tail (exercised by the browser-side suite instead)."""

    def __init__(self):     # deliberately not DashboardNode.__init__
        self._full_map_info = (ORIGIN[0], ORIGIN[1], RES)
        self.acks = []
        self.notes = []

    def active_map(self):
        return MAP

    def get_logger(self):
        return self

    def warn(self, *a):
        pass

    def info(self, *a):
        pass

    def _rooms_changed(self, note):
        self.notes.append(note)
        self.acks.append({'ok': True})

    @property
    def _state(self):
        node = self

        class _S:
            @staticmethod
            def broadcast(msg):
                node.acks.append(msg)
        return _S


@pytest.fixture
def rooms(tmp_path, monkeypatch):
    """A map whose left half is saloni and right half is kouzina."""
    mask = np.zeros((20, 20, 4), np.uint8)
    mask[:, :10, :3] = SALONI
    mask[:, 10:, :3] = KOUZINA
    mask[:, :, 3] = 255
    mask_path = tmp_path / f'{MAP}_room_mask.png'
    colours_path = tmp_path / f'{MAP}_room_colors.yaml'
    Image.fromarray(mask).save(mask_path)
    colours_path.write_text(yaml.safe_dump({'saloni': SALONI, 'kouzina': KOUZINA}))
    monkeypatch.setattr(room_files, 'paths_for',
                        lambda name: (str(mask_path), str(colours_path)))
    return mask_path, colours_path


def _read(paths):
    mask_path, colours_path = paths
    return (np.array(Image.open(mask_path).convert('RGBA')),
            yaml.safe_load(colours_path.read_text()))


def _count(mask, rgb):
    return int((((mask[:, :, 0] == rgb[0]) & (mask[:, :, 1] == rgb[1])
                 & (mask[:, :, 2] == rgb[2])) & (mask[:, :, 3] > 50)).sum())


def _failure(node):
    bad = [a for a in node.acks if a.get('ok') is False]
    return bad[0]['error'] if bad else None


# ── delete ─────────────────────────────────────────────────────────────────

def test_delete_unassigns_only_that_rooms_pixels(rooms):
    node = _Node()
    node._delete_room('saloni')
    mask, colours = _read(rooms)
    assert list(colours) == ['kouzina']
    assert _count(mask, SALONI) == 0
    assert _count(mask, KOUZINA) == 200, 'the other room was touched'


def test_delete_of_an_unknown_room_is_refused(rooms):
    node = _Node()
    node._delete_room('δεν υπάρχει')
    assert 'δεν υπάρχει' in _failure(node)
    assert set(_read(rooms)[1]) == {'saloni', 'kouzina'}, 'file was rewritten'


# ── merge ──────────────────────────────────────────────────────────────────

def test_merge_folds_the_second_room_into_the_first(rooms):
    node = _Node()
    node._merge_rooms(['saloni', 'kouzina'])
    mask, colours = _read(rooms)
    assert list(colours) == ['saloni']
    assert _count(mask, SALONI) == 400, 'the merged room is not the whole floor'
    assert _count(mask, KOUZINA) == 0


def test_merge_needs_two_rooms(rooms):
    node = _Node()
    node._merge_rooms(['saloni'])
    assert _failure(node)
    assert set(_read(rooms)[1]) == {'saloni', 'kouzina'}


def test_merge_refuses_an_unknown_room(rooms):
    node = _Node()
    node._merge_rooms(['saloni', 'ξένο'])
    assert 'ξένο' in _failure(node)
    assert _count(_read(rooms)[0], KOUZINA) == 200


# ── divide ─────────────────────────────────────────────────────────────────

def test_divide_cuts_the_room_along_the_line(rooms):
    """A horizontal line across saloni at y=1.0 m: what is above it becomes the
    new room, the rest keeps the old name. Nothing outside saloni may change.
    The line's own row of pixels stays with the original (side == 0), which is
    why the halves are 90/110 and not 100/100."""
    node = _Node()
    node._split_room('saloni', 0.0, 1.0, 1.0, 1.0, 'γραφείο', NEW)
    mask, colours = _read(rooms)
    assert _failure(node) is None
    assert set(colours) == {'saloni', 'kouzina', 'γραφείο'}
    assert colours['γραφείο'] == NEW
    assert _count(mask, NEW) == 90
    assert _count(mask, SALONI) == 110
    assert _count(mask, KOUZINA) == 200, 'the neighbouring room was cut too'


def test_divide_refuses_a_line_that_leaves_a_sliver(rooms):
    """A line along the very edge is a misdrawn gesture, not a room."""
    node = _Node()
    node._split_room('saloni', 0.0, 0.05, 1.0, 0.05, 'γραφείο', NEW)
    assert _failure(node)
    assert set(_read(rooms)[1]) == {'saloni', 'kouzina'}


def test_divide_refuses_a_name_that_is_taken(rooms):
    node = _Node()
    node._split_room('saloni', 0.0, 1.0, 1.0, 1.0, 'kouzina', NEW)
    assert 'kouzina' in _failure(node)
    assert _count(_read(rooms)[0], NEW) == 0


def test_divide_refuses_the_same_colour_twice(rooms):
    """Two rooms painted the same colour are one room as far as every reader of
    the mask is concerned — the name would simply never be found again."""
    node = _Node()
    node._split_room('saloni', 0.0, 1.0, 1.0, 1.0, 'γραφείο', SALONI)
    assert _failure(node)
    assert set(_read(rooms)[1]) == {'saloni', 'kouzina'}


def test_divide_needs_a_line_with_length(rooms):
    node = _Node()
    node._split_room('saloni', 0.5, 0.5, 0.5, 0.5, 'γραφείο', NEW)
    assert _failure(node)
    assert set(_read(rooms)[1]) == {'saloni', 'kouzina'}


def test_a_map_with_no_rooms_yet_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(room_files, 'paths_for',
                        lambda name: (str(tmp_path / 'missing.png'),
                                      str(tmp_path / 'missing.yaml')))
    node = _Node()
    node._delete_room('saloni')
    assert _failure(node)
