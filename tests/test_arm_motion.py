"""Tests for moving the arm by voice (home_robot/arm_motion.py).

2026-08-04, from the owner: «μιλάω στο Gemini και του λέω κάνε τον βραχίονα
λίγο μπροστά και δεν κάνει τίποτα, ή κάνω κάτι λάθος;»

Not a phrasing problem — there was no arm tool at all. The schema had `pick`
and `fetch`, which drive the whole pick-and-place pipeline at a named object,
and nothing that simply moved the arm. The dashboard sliders and the PS5 stick
could; that was the entire list.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_arm_motion.py -q
"""
import os

import pytest

from home_robot.arm_motion import AMOUNTS, DIRECTIONS, deltas_for, describe
from home_robot.tool_router import select_tools

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRIDGE = open(f'{PKG}/home_robot/nodes/llm_bridge_node.py').read()


# ── the directions a person actually says ────────────────────────────────────

@pytest.mark.parametrize('direction', [
    'μπροστά', 'πίσω', 'πάνω', 'κάτω', 'αριστερά', 'δεξιά', 'μέσα', 'έξω',
])
def test_every_spoken_direction_moves_something(direction):
    d = deltas_for(direction)
    assert d, f'{direction} maps to nothing'
    assert all(isinstance(v, float) for v in d.values())


def test_an_unknown_direction_is_refused_not_guessed():
    assert deltas_for('πλάγια') is None


def test_shoulder_up_is_negative():
    """‼️ Measured on the hardware — see project_robot_arm_joy. Backwards here
    means the arm dives into the chassis when someone asks for up."""
    assert deltas_for('πάνω')['shoulder'] < 0
    assert deltas_for('κάτω')['shoulder'] > 0


def test_opposites_cancel():
    """"λίγο μπροστά" then "λίγο πίσω" has to come back to where it started, or
    the arm walks itself out of the envelope over a conversation."""
    for a, b in [('μπροστά', 'πίσω'), ('πάνω', 'κάτω'),
                 ('αριστερά', 'δεξιά'), ('μέσα', 'έξω')]:
        fwd, back = deltas_for(a), deltas_for(b)
        assert set(fwd) == set(back)
        for joint in fwd:
            assert fwd[joint] == pytest.approx(-back[joint])


def test_forward_moves_shoulder_and_elbow_together():
    """Dropping the shoulder alone swings the gripper down as much as out,
    which does not read as "μπροστά" to anyone watching."""
    d = deltas_for('μπροστά')
    assert 'shoulder' in d and 'elbow' in d
    assert d['shoulder'] * d['elbow'] < 0, 'the elbow must open as the shoulder drops'


# ── how much ─────────────────────────────────────────────────────────────────

def test_the_default_is_a_small_step():
    """"λίγο" is what people say when adjusting something they can see."""
    assert deltas_for('πάνω') == deltas_for('πάνω', 'λίγο')


def test_amounts_are_ordered():
    step = [AMOUNTS[a] for a in ('λίγο', 'κανονικά', 'πολύ')]
    assert step == sorted(step)
    assert step[0] > 0


def test_an_unknown_amount_falls_back_to_the_default():
    """The model can invent a word. Refusing the whole command over it would
    be worse than moving a little."""
    assert deltas_for('πάνω', 'τελείως') == deltas_for('πάνω')


def test_a_single_step_stays_well_inside_the_envelope():
    """MECH_LIMITS gives shoulder ±1.57. One "πολύ" must not span it — the
    driver clamps, but a command that always clamps is not a control."""
    for direction in DIRECTIONS:
        for joint, delta in deltas_for(direction, 'πολύ').items():
            assert abs(delta) <= 1.0, f'{direction}/{joint} is a whole-range swing'


def test_it_says_back_what_it_was_asked():
    assert describe('μπροστά') == 'λίγο μπροστά'
    assert describe('πάνω', 'πολύ') == 'πολύ πάνω'


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_bridge_has_an_arm_tool():
    """‼️ THE BUG: it did not."""
    assert "'name': 'arm'" in _BRIDGE
    assert "elif name == 'arm':" in _BRIDGE


def test_the_arm_tool_reads_the_live_pose():
    """The wire format is absolute positions, so a relative "λίγο μπροστά"
    without joint_states could only guess a starting point — and a guess flings
    the arm from wherever it is to wherever the guess said."""
    body = _BRIDGE.split('def _move_arm')[1].split('\n    def ')[0]
    assert '_arm_pose' in body
    assert 'cur + delta' in body
    assert "'/arm/joint_states'" in _BRIDGE


def test_it_refuses_rather_than_guessing_without_a_pose():
    body = _BRIDGE.split('def _move_arm')[1].split('\n    def ')[0]
    assert 'if not self._arm_pose' in body


def test_it_publishes_on_the_topic_the_driver_listens_to():
    """arm_driver subscribes to arm/joint_cmd and arm/gripper_cmd. A publisher
    on any other name is a command into the void."""
    assert "'/arm/joint_cmd'" in _BRIDGE
    assert "'/arm/gripper_cmd'" in _BRIDGE


def test_moving_the_arm_does_not_need_confirmation():
    """The arm is in front of you and the robot stays put. Only the tools that
    drive it across the house ask first."""
    from home_robot.motion_confirm import MOTION_TOOLS
    assert 'arm' not in MOTION_TOOLS


# ── routing: the phrase has to reach the tool ────────────────────────────────

_FAKE = [{'function': {'name': n}} for n in
         ('arm', 'pick', 'fetch', 'goto', 'stop', 'sort')]


def _routed(text):
    return [t['function']['name'] for t in select_tools(text, _FAKE)]


@pytest.mark.parametrize('utterance', [
    'Κάνε τον βραχίονα λίγο μπροστά.',
    'Άπλωσε τον βραχίονα.',
    'Σήκωσε το χέρι σου.',
    'Τέντωσε τον αγκώνα.',
    'Γύρνα τον καρπό.',
])
def test_arm_phrases_reach_the_arm_tool(utterance):
    assert 'arm' in _routed(utterance)


@pytest.mark.parametrize('utterance', [
    'Πιάσε το μπουκάλι.',
    'Φέρε μου το ποτήρι.',
])
def test_grasping_still_goes_to_pick(utterance):
    """"κάνε τον βραχίονα μπροστά" is not a request to grab anything, and
    routing it to pick would send the robot hunting an object nobody named."""
    assert 'pick' in _routed(utterance)
