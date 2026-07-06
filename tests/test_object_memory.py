"""Tests for the spatial object memory core (see home_robot/object_memory.py).

Robot-free unit tests of the merge/query/prune bookkeeping with an injected
fake clock, plus a static wiring check that the node exposes the query/answer
and memory/store contract.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_object_memory.py -q
"""
import os

from home_robot.object_memory import ObjectMemory

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = f'{PKG}/home_robot/nodes'


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


# ── merge behaviour ───────────────────────────────────────────────────

def test_nearby_same_label_merges_into_one_instance():
    mem = ObjectMemory(merge_distance=0.6, ema_alpha=0.5, min_conf=0.4)
    mem.observe('cup', 1.0, 1.0, 0.5, conf=0.8)
    mem.observe('cup', 1.1, 1.05, 0.5, conf=0.9)   # ~0.11 m away → same cup
    assert len(mem.all()) == 1
    inst = mem.all()[0]
    assert inst['count'] == 2
    assert inst['conf'] == 0.9                       # keeps the max
    assert 1.0 <= inst['x'] <= 1.1                   # EMA between the two


def test_far_same_label_is_a_separate_instance():
    mem = ObjectMemory(merge_distance=0.6)
    mem.observe('cup', 1.0, 1.0, 0.5, conf=0.8)
    mem.observe('cup', 3.0, 3.0, 0.5, conf=0.8)      # far → second cup
    assert len(mem.all()) == 2


def test_different_labels_never_merge():
    mem = ObjectMemory(merge_distance=5.0)
    mem.observe('cup', 1.0, 1.0, 0.5, conf=0.9)
    mem.observe('bottle', 1.0, 1.0, 0.5, conf=0.9)   # same spot, different label
    assert len(mem.all()) == 2


def test_low_confidence_is_dropped():
    mem = ObjectMemory(min_conf=0.5)
    assert mem.observe('cup', 0, 0, 0.5, conf=0.3) is None
    assert mem.all() == []


# ── query ──────────────────────────────────────────────────────────────

def test_query_returns_most_recent_instance():
    clk = FakeClock(100.0)
    mem = ObjectMemory(merge_distance=0.3, clock=clk)
    mem.observe('cup', 0.0, 0.0, 0.5, conf=0.9)      # cup A at t=100
    clk.t = 200.0
    mem.observe('cup', 5.0, 5.0, 0.5, conf=0.9)      # cup B at t=200 (far)
    hit = mem.query('cup')
    assert hit['x'] == 5.0                            # newest wins
    assert mem.query('banana') is None


# ── prune ──────────────────────────────────────────────────────────────

def test_prune_forgets_stale_instances():
    clk = FakeClock(0.0)
    mem = ObjectMemory(clock=clk)
    mem.observe('cup', 0, 0, 0.5, conf=0.9)
    clk.t = 100.0
    assert mem.prune(max_age=50.0) == 1              # 100 s old > 50 s
    assert mem.all() == []


def test_prune_disabled_keeps_everything():
    clk = FakeClock(0.0)
    mem = ObjectMemory(clock=clk)
    mem.observe('cup', 0, 0, 0.5, conf=0.9)
    clk.t = 1e9
    assert mem.prune(max_age=0.0) == 0
    assert len(mem.all()) == 1


# ── persistence round-trip ─────────────────────────────────────────────

def test_to_list_load_list_round_trips_and_bumps_next_id():
    mem = ObjectMemory()
    mem.observe('cup', 1.0, 1.0, 0.5, conf=0.9)
    saved = mem.to_list()

    restored = ObjectMemory()
    restored.load_list(saved)
    assert restored.all()[0]['label'] == 'cup'
    # a new object gets a fresh id, not a clash with the loaded one
    new = restored.observe('bottle', 9.0, 9.0, 0.5, conf=0.9)
    assert new['id'] != saved[0]['id']


# ── static wiring check ─────────────────────────────────────────────────

def test_node_wires_query_answer_and_rag_store():
    with open(f'{NODES}/object_memory_node.py') as f:
        src = f.read()
    assert "'detected_objects'" in src
    assert "'object_memory/query'" in src
    assert "'object_memory/answer'" in src
    assert "'memory/store'" in src        # feeds RAG recall path
