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

def _twist_publishers(path):
    """[(lineno, topic)] for every create_publisher(Twist, '...') in a file."""
    out = []
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return out
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
        if isinstance(topic, ast.Constant):
            out.append((node.lineno, topic.value))
    return out


def test_doa_turns_the_base_through_the_remappable_name():
    """Relative 'cmd_vel', so the launch file can route it.

    Not hardcoded either way on purpose: with use_obstacle_safety:=true the
    twist must pass THROUGH obstacle_safety_node (which zeroes forward motion
    when the camera sees something), and with it false nothing relays /cmd_vel
    at all so it must go straight to cmd_vel_safe. bringup picks per config.
    """
    pubs = dict((t, ln) for ln, t in _twist_publishers(DOA))
    assert pubs, 'the velocity publisher is gone or changed shape'
    assert 'cmd_vel' in pubs, (
        f'expected a relative cmd_vel publisher, found {sorted(pubs)} — an '
        'absolute /cmd_vel cannot be remapped by the launch file, and a '
        'hardcoded cmd_vel_safe walks straight past the safety net')


def test_raw_velocity_publishers_are_remapped_by_the_launch_file():
    """Every node that publishes a bare 'cmd_vel' must be routed in bringup.

    /cmd_vel has ZERO subscribers unless obstacle_safety_node is running to
    relay it, and localize.launch.py leaves that off. A Twist publisher on
    'cmd_vel' with no remapping is therefore a feature that silently does
    nothing — which is how spoken movement commands, the mission executor's
    final approach and doa_node's turn-to-speaker were all dead at once.
    """
    bring = open(BRINGUP).read()
    offenders = []
    for fn in sorted(os.listdir(NODES_DIR)):
        if not fn.endswith('.py'):
            continue
        topics = [t for _, t in _twist_publishers(os.path.join(NODES_DIR, fn))]
        if not any(t.lstrip('/') == 'cmd_vel' for t in topics):
            continue
        # obstacle_safety_node is the relay itself: it SUBSCRIBES to cmd_vel
        # and publishes cmd_vel_safe, so it is never remapped.
        if fn == 'obstacle_safety_node.py':
            continue
        # Find the node's block in bringup and check it carries a remapping.
        m = re.search(rf"executable='{re.escape(fn)}'(.*?)\n    \)", bring, re.S)
        if not m:
            offenders.append(f'{fn} (not launched by bringup at all)')
        elif 'remappings' not in m.group(1):
            offenders.append(f'{fn} (no remappings= in bringup)')
    assert not offenders, (
        'these publish raw Twists to cmd_vel but bringup never routes them, '
        'so they will move nothing whenever obstacle_safety_node is off '
        '(which is the default under `robot max`): ' + ', '.join(offenders))


def test_an_absolute_cmd_vel_is_never_used():
    """'/cmd_vel' with a leading slash ignores the node's namespace and reads
    as deliberate, which is exactly the wrong signal for a topic that only
    works when something happens to be relaying it."""
    offenders = []
    for fn in sorted(os.listdir(NODES_DIR)):
        if not fn.endswith('.py'):
            continue
        for ln, topic in _twist_publishers(os.path.join(NODES_DIR, fn)):
            if topic == '/cmd_vel':
                offenders.append(f'{fn}:{ln}')
    assert not offenders, (
        'absolute /cmd_vel publishers: ' + ', '.join(offenders))


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


# ── bug 3: the robot turned itself with no command given ────────────────────
#
# 2026-08-04, a plain `robot max` with nobody talking TO the robot. It started
# turning on its own; the user saw a machine driving off unasked. doa_node's
# log for those 100 s:
#
#   Wake word DoA: 205° / Rotating -155° toward speaker
#   Wake word DoA: 205° / Rotating -155° toward speaker   (7 s later, again)
#   Wake word DoA: 268° / Rotating  -92° toward speaker
#   Wake word DoA: 268° / Rotating  -92° toward speaker   (4 s later, again)
#   Wake word DoA: 202° / Rotating -158° toward speaker
#
# Two separate faults on one line of code:
#   a) it was armed by default, so a wake word — which is NOT a command, and
#      which the far-field mic picks up out of conversations nobody addressed
#      to the robot — moved the base before anyone said what to do. Every
#      other motion path got a yes/no confirmation after the "executes
#      overheard conversation" bug; this one walked straight past it.
#   b) no cooldown. The wake word re-fires every ~1.5 s while somebody keeps
#      talking, and a turn already in flight has not changed the measured DoA
#      yet, so each repeat started the SAME turn over again.

DASH = os.path.join(NODES_DIR, 'web_dashboard_node.py')


def _param_default(src, name):
    """The default in a declare_parameter(name, default) call."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'declare_parameter'
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == name):
            return ast.literal_eval(node.args[1])
    raise AssertionError(f'{name} is not declared at all')


def test_turning_toward_a_speaker_is_off_by_default():
    """A wake word is not a command, so it may not move the base unasked."""
    assert _param_default(open(DOA).read(), 'rotate_on_wake') is False, (
        'rotate_on_wake defaults to True again — `robot max` will have the '
        'robot turning itself at every wake word, including the ones it '
        'overhears from conversations that are not addressed to it')


def test_the_launch_files_do_not_arm_it_either():
    """The node default is worthless if bringup passes true over the top."""
    bring = open(BRINGUP).read()
    m = re.search(r"DeclareLaunchArgument\('doa_rotate_on_wake',\s*"
                  r"default_value='(\w+)'", bring)
    assert m, 'doa_rotate_on_wake is no longer a declared bringup argument'
    assert m.group(1) == 'false', (
        'bringup arms turn-toward-speaker for every launch; it must be opt-in '
        'from the dashboard')
    m = re.search(r"LaunchConfiguration\('doa_rotate_on_wake',\s*"
                  r"default='(\w+)'\)", bring)
    assert m and m.group(1) == 'false', (
        'the LaunchConfiguration default disagrees with the declared one — '
        'whichever is true wins and the feature is armed again')
    loc = open(LOCALIZE).read()
    assert not re.search(r"'doa_rotate_on_wake'\s*:\s*'true'", loc), (
        'localize.launch.py — the file `robot max` runs — arms it explicitly')


def test_a_repeat_wake_word_does_not_restart_the_same_turn():
    """One utterance, one turn. The wake word re-fires every ~1.5 s."""
    src = open(DOA).read()
    assert 'rotate_cooldown_s' in src, 'the repeat-wake-word cooldown is gone'
    assert re.search(r'_last_rotate\)\s*<\s*self\._rotate_cooldown', src), (
        'nothing checks the cooldown before starting a turn, so every repeat '
        'wake word starts the same rotation over again')


def test_the_turn_respects_the_e_stop():
    """A latched e-stop means nobody may ask the base to move — including us.

    The driver drops the wheel writes, so the robot did stay still, but this
    node kept publishing a turn for the whole sweep and the leftover twist was
    still in flight when the latch was released.
    """
    src = open(DOA).read()
    assert re.search(r"create_subscription\(\s*\n?\s*Bool,\s*'/emergency_stop'", src), (
        'doa_node does not subscribe to /emergency_stop')
    assert re.search(r'if self\._estop', src), (
        'the e-stop is subscribed but never consulted before turning')


def test_the_e_stop_subscription_is_latched():
    """Same trap as everywhere else: the e-stop is published TRANSIENT_LOCAL
    and a volatile subscriber joining later never hears the stop at all."""
    src = open(DOA).read()
    m = re.search(r"create_subscription\(\s*\n?\s*Bool,\s*'/emergency_stop',\s*"
                  r"self\._on_estop,\s*(\w+)\)", src)
    assert m, 'the /emergency_stop subscription changed shape'
    qos = m.group(1)
    assert re.search(rf'{qos}\s*=\s*QoSProfile\([^)]*TRANSIENT_LOCAL', src, re.S), (
        f'{qos} is not TRANSIENT_LOCAL — doa_node will miss an e-stop that was '
        'latched before it started')


# ── the switch the user actually asked for ──────────────────────────────────

def test_the_dashboard_can_arm_and_disarm_it():
    """Off by default is only usable if there is a way to turn it on."""
    doa = open(DOA).read()
    dash = open(DASH).read()
    assert "'doa/rotate_enable'" in doa, 'doa_node has no enable topic'
    assert "'/doa/rotate_enable'" in dash, 'the dashboard cannot arm it'
    assert "'doa/rotate_state'" in doa, (
        'doa_node never reports its state, so the dashboard checkbox can only '
        'guess')
    assert "'/doa/rotate_state'" in dash, 'the dashboard does not read the state'
    assert "t == 'doa_rotate'" in dash, 'the websocket command is not dispatched'
    assert "id=\"dr-rotate\"" in dash, 'there is no checkbox in the page'


def test_the_switch_survives_a_restart():
    """Arming it from the web must not have to be redone after `robot max`."""
    src = open(DOA).read()
    assert '_STATE_PATH' in src, 'the setting is not persisted anywhere'
    assert re.search(r'def _load_enabled', src) and re.search(r'def _save_enabled', src), (
        'the enable state is not loaded/saved')
    assert re.search(r'os\.replace\(tmp, _STATE_PATH\)', src), (
        'the state file is written in place — a crash mid-write leaves JSON '
        'that cannot be parsed and the setting silently reverts')


def test_the_state_topics_are_latched_on_both_ends():
    """doa_node starts long before the dashboard; volatile QoS would leave the
    checkbox showing OFF for a feature that is actually armed."""
    for path, name in ((DOA, 'doa_node'), (DASH, 'web_dashboard_node')):
        src = open(path).read()
        # publisher: (Bool, 'doa/rotate_state', QOS)
        # subscription: (Bool, '/doa/rotate_state', callback, QOS)
        m = re.search(r"rotate_state',\s*(?:self\.\w+,\s*)?(\w+)\)", src)
        assert m, f'{name}: the rotate_state publisher/subscription changed shape'
        assert re.search(rf'{m.group(1)}\s*=\s*QoSProfile\([^)]*TRANSIENT_LOCAL',
                         src, re.S), (
            f'{name}: /doa/rotate_state is not latched')
