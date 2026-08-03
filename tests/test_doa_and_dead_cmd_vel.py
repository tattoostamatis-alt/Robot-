"""The XVF3800's direction-of-arrival, and the /cmd_vel trap in general.

Two bugs found live on 2026-08-03, on a fully running stack:

1. `doa_node` NEVER STARTED under `robot max`. bringup declares `use_doa`
   default false, and localize.launch.py — the file `robot max` actually runs —
   never forwarded it. `ros2 node list` on the live robot showed every other
   voice node and zero doa_node. So the hardware DoA and the DSP's VAD, both
   already implemented, were simply never running. This is the FOURTH
   unforwarded `use_*` in this package (use_situational, use_planner,
   llm_backend were the others), which is why the check below is generic.

2. Even started, it published its "turn toward the speaker" twist to
   /cmd_vel — a topic with several publishers and ZERO subscribers here.
   roomba_driver listens on cmd_vel_safe alone. Fifth time in this repo:
   PS5 teleop, keyboard teleop, the web D-pad and person_follower all made it.

The dead-topic test scans EVERY node rather than just this one, because the
mistake keeps arriving in new files.
"""
import ast
import os
import re

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_DIR = os.path.join(PKG, 'home_robot', 'nodes')
DOA = os.path.join(NODES_DIR, 'doa_node.py')
LOCALIZE = os.path.join(PKG, 'launch', 'localize.launch.py')
BRINGUP = os.path.join(PKG, 'launch', 'bringup.launch.py')


# ── bug 1: the node that never ran ──────────────────────────────────────────

def test_use_doa_is_forwarded_to_bringup():
    """localize.launch.py is what `robot max` runs; bringup defaults it off.

    An unforwarded use_* only reaches bringup when the command line happens to
    set it — which is exactly how this feature stayed dark.
    """
    src = open(LOCALIZE).read()
    assert re.search(r"'use_doa'\s*:", src), (
        "localize.launch.py does not forward use_doa — bringup will fall back "
        "to its default of false and doa_node will never start under "
        "`robot max`, exactly as it did before 2026-08-03")


def test_use_doa_defaults_to_on_in_the_file_that_runs():
    src = open(LOCALIZE).read()
    m = re.search(r"DeclareLaunchArgument\(\s*'use_doa',\s*default_value='(\w+)'", src)
    assert m, 'use_doa is not a declared argument of localize.launch.py'
    assert m.group(1) == 'true', (
        'use_doa declared false here would reintroduce the original bug in a '
        'new place')


def test_features_localize_decides_on_are_forwarded_not_inherited():
    """The generic form of the bug that has now happened four times.

    Inheritance DOES work for a flag the caller passes on the command line —
    `robot max` spells out use_wake_word, use_stt, use_tts, use_llm, use_arm,
    so those reach bringup fine. The trap is a feature localize is supposed to
    DECIDE: nobody types it, bringup's own default wins, and the feature is
    silently off for ever. use_situational, use_planner, llm_backend and
    use_doa all died exactly this way.

    So the rule enforced here is narrow: anything localize.launch.py mentions
    in a comment or condition as its decision must appear in the forwarding
    dict, not be left to inheritance.
    """
    loc = open(LOCALIZE).read()
    forwarded = set(re.findall(r"'(use_\w+)'\s*:", loc))
    # The four that have already bitten, kept as a permanent regression guard.
    for flag in ('use_dashboard', 'use_planner', 'use_doa'):
        assert flag in forwarded, (
            f'{flag} is not forwarded by localize.launch.py. It has been '
            'broken this way before: bringup falls back to its own default '
            'and `robot max` cannot reach the feature at all.')


# ── bug 2: the dead topic ───────────────────────────────────────────────────

def test_doa_turns_the_base_on_the_topic_the_roomba_reads():
    src = open(DOA).read()
    m = re.search(r"_cmd_vel_pub\s*=\s*self\.create_publisher\(\s*Twist,\s*'([^']+)'",
                  src)
    assert m, 'the velocity publisher is gone or changed shape'
    assert m.group(1).lstrip('/') == 'cmd_vel_safe', (
        f'publishing to {m.group(1)!r} — the Roomba will never see it, and '
        '"turn toward the speaker" will look broken while computing perfectly '
        'correct twists')


def test_no_node_publishes_twists_to_the_dead_cmd_vel_topic():
    """Scan every node, because this keeps arriving in new files.

    /cmd_vel has zero subscribers on this graph. A Twist publisher aimed at it
    is a feature that will silently do nothing. Nodes that are *remapped* in
    the launch file are the documented exception and are listed here.
    """
    # Remapped to cmd_vel_safe in bringup.launch.py, so publishing to the
    # relative name is correct for these.
    remapped = {'teleop_twist_keyboard', 'teleop_node'}
    offenders = []
    for fn in sorted(os.listdir(NODES_DIR)):
        if not fn.endswith('.py'):
            continue
        if fn[:-3] in remapped:
            continue
        src = open(os.path.join(NODES_DIR, fn)).read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'create_publisher'):
                continue
            if len(node.args) < 2:
                continue
            typ, topic = node.args[0], node.args[1]
            if not (isinstance(typ, ast.Name) and typ.id == 'Twist'):
                continue
            if not isinstance(topic, ast.Constant):
                continue
            if topic.value.lstrip('/') == 'cmd_vel':
                offenders.append(f'{fn}:{node.lineno}')
    assert not offenders, (
        '/cmd_vel is a dead topic here (zero subscribers — roomba_driver reads '
        'cmd_vel_safe). These publishers will never move the robot: '
        + ', '.join(offenders))


# ── the hardware VAD ────────────────────────────────────────────────────────

def test_the_dsp_voice_activity_flag_is_published():
    """The XVF3800 returns a speech flag with every DoA read, for free.

    Nothing published it, so the only voice-activity signal in the stack was
    stt_node's energy threshold — which is why a fan spike can open a
    recording. Exposing it costs one extra message per transition.
    """
    src = open(DOA).read()
    assert re.search(r"_vad_pub\s*=\s*self\.create_publisher\(\s*Bool,\s*'voice_activity'",
                     src), 'the hardware VAD is read but never published'
    assert 'Bool' in re.search(r'from std_msgs\.msg import ([^\n]+)', src).group(1), \
        'Bool is not imported — the node will not start'


def test_voice_activity_is_edge_triggered():
    """Polled at 10 Hz; publishing every poll would bury a bag in duplicates."""
    src = open(DOA).read()
    assert re.search(r'if speech != self\._last_speech', src), (
        'the VAD publishes on every poll instead of on transitions')
    assert re.search(r'_last_speech\s*=\s*None', src), (
        '_last_speech must start as None so the first reading is published')
