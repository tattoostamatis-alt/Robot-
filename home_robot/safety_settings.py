"""The safety clearances, in one place, with names a person can read.

Every number that decides how close the robot lets itself get to something was
previously spread across three places with three different lifetimes: a launch
argument (`obstacle_safety_distance`), two costmap sections of nav2_params.yaml
(`inflation_layer.inflation_radius`, twice, and they have to agree), and the
Roomba driver's bumper/cliff parameters. Changing one meant editing YAML and
restarting the whole stack, which is why in practice nobody changed them.

This module is the description of those knobs — what they are called, what they
mean, what a sane range is, and which node owns which ROS parameter — plus the
file that remembers the user's choice. It deliberately holds NO ROS objects:
the dashboard node does the service calls, and everything here can be tested
without a running graph.

## What is live and what is not

Everything in SPECS applies immediately, at runtime, through
`rcl_interfaces/srv/SetParameters`, and each owning node was checked to honour
it (the costmap inflation layer has had a dynamic-parameters callback since
Nav2 Humble; obstacle_safety_node and roomba_driver got theirs alongside this
file). A knob that only takes effect on the next launch does not belong here —
it would show a value the robot is not using, which is worse than no knob.

That rules out one clearance on purpose, documented in INFO_ONLY so the
dashboard can still show it rather than pretend the ones it does show are the
whole story:

  * FootprintApproach's time_before_collision.

Nav2 reads polygon parameters once, in on_configure, so writing them at runtime
succeeds and changes nothing. It stays in nav2_params.yaml, genuinely
read-only from the web.

‼️ collision_monitor's Stop polygon (the hard skirt) used to be INFO_ONLY too,
and got asked about often enough that it now has its own editable control in
the Safety tab's "Σκληρά όρια" card — see home_robot/collision_skirt.py. It
isn't a live SetParameters write either (same on_configure-once limitation);
that control patches nav2_params.yaml's moving_forward/moving_backward points
directly and triggers the same full-stack restart map switching uses. What
this file's `inflation_radius` spec changes live, and what people usually mean
first when they ask, is a DIFFERENT clearance: how much room the *planner*
leaves around walls, not the hard emergency-stop distance.

## The trap this file exists to keep visible

The doorways in this flat are ~0.78 m. Raising inflation_radius makes the robot
prefer the middle of the room, and past roughly 0.30 m the inflated walls of a
doorway meet in the middle: the planner then reports no path through an opening
that is plainly clear on the map. That has cost real debugging time before, so
`warn_above` carries the number and the dashboard says so next to the slider
instead of leaving it to be rediscovered.
"""
import json
import os
from typing import NamedTuple

STATE_PATH = os.path.expanduser('~/.home_robot/safety_settings.json')


class Spec(NamedTuple):
    """One user-facing knob, and the ROS parameters it writes to.

    `targets` is a tuple because a knob is not always one parameter: the
    inflation radius has to be written to the local AND the global costmap or
    the planner and the controller disagree about where the walls are — the
    kind of half-applied setting that produces a robot which plans a path it
    then refuses to drive.
    """
    key: str
    kind: str                  # 'double' | 'bool'
    default: float | bool
    targets: tuple             # ((node_name, parameter_name), ...)
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.01
    warn_above: float | None = None
    warn_below: float | None = None


SPECS = (
    # ── how far ahead the camera stops the robot ─────────────────────────────
    # obstacle_safety_node is the only guard on the raw cmd_vel that teleop and
    # llm_bridge publish — Nav2's costmap does not see those at all.
    Spec('stop_distance', 'double', 0.50,
         (('/obstacle_safety_node', 'safety_distance'),),
         lo=0.20, hi=1.50, step=0.05, warn_above=1.00),
    # How much of the image width counts as "ahead". Widening it makes the
    # robot stop for things it would have driven past; 1.0 means anything
    # anywhere in frame blocks forward motion, which in a corridor is
    # permanent.
    Spec('center_width', 'double', 0.50,
         (('/obstacle_safety_node', 'center_fraction'),),
         lo=0.20, hi=1.00, step=0.05, warn_above=0.80),
    # Fail-safe: no detections for this long and forward motion is refused.
    # Raising it widens the window in which a dead camera looks like a clear
    # path, so the warning is on the HIGH side here too.
    Spec('detector_timeout', 'double', 2.0,
         (('/obstacle_safety_node', 'detection_timeout'),),
         lo=1.0, hi=10.0, step=0.5, warn_above=5.0),

    # ── how much room Nav2 leaves around walls ───────────────────────────────
    Spec('inflation_radius', 'double', 0.22,
         (('/local_costmap/local_costmap', 'inflation_layer.inflation_radius'),
          ('/global_costmap/global_costmap', 'inflation_layer.inflation_radius')),
         lo=0.10, hi=0.60, step=0.01, warn_above=0.30),
    # How sharply the cost falls off inside that radius. Higher = the robot
    # hugs the middle less eagerly and is willing to pass closer.
    Spec('cost_scaling', 'double', 3.0,
         (('/local_costmap/local_costmap', 'inflation_layer.cost_scaling_factor'),
          ('/global_costmap/global_costmap', 'inflation_layer.cost_scaling_factor')),
         lo=1.0, hi=10.0, step=0.5),

    # ── the sensors that stop the robot after it has already touched ─────────
    # All three default ON and there is no good reason to turn them off except
    # a stuck sensor. oi_mode is 'full', which means the Roomba itself does NOT
    # act on its bumper or its cliff sensors — these flags are the only thing
    # between the robot and the furniture, or the stairs.
    Spec('bump_stop', 'bool', True, (('/roomba_driver', 'bump_stop'),)),
    Spec('cliff_stop', 'bool', True, (('/roomba_driver', 'cliff_stop'),)),
    Spec('wheel_drop_stop', 'bool', True, (('/roomba_driver', 'wheel_drop_stop'),)),
    # How long forward stays blocked after the bumper/cliff clears, so the
    # rebound does not drive straight back into the same thing.
    Spec('bump_block_s', 'double', 1.0, (('/roomba_driver', 'bump_block_s'),),
         lo=0.0, hi=5.0, step=0.5, warn_below=0.5),
    Spec('cliff_block_s', 'double', 1.5, (('/roomba_driver', 'cliff_block_s'),),
         lo=0.0, hi=5.0, step=0.5, warn_below=0.5),
)

BY_KEY = {s.key: s for s in SPECS}

# Clearances the dashboard shows but cannot set live — see the module
# docstring. `value` is what nav2_params.yaml actually says, so the panel is
# not quoting a number from memory; the tests check it against the file.
INFO_ONLY = (
    {'key': 'time_before_collision', 'value': 0.9, 'unit': 's',
     'source': 'collision_monitor / FootprintApproach'},
)


def clamp(key: str, value):
    """Coerce a value from the browser into something the spec allows.

    Returns None for an unknown key or a value that is not a number/bool, so
    the caller can drop it rather than write nonsense into a safety parameter.
    A slider is not the only way into this code path — anything on the network
    that has the dashboard token can send whatever it likes over the websocket.
    """
    spec = BY_KEY.get(key)
    if spec is None:
        return None
    if spec.kind == 'bool':
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if v != v:                     # NaN survives every comparison below
        return None
    return round(min(spec.hi, max(spec.lo, v)), 4)


def defaults() -> dict:
    return {s.key: s.default for s in SPECS}


def load() -> dict:
    """Saved settings, clamped, with anything missing filled from defaults.

    A file written by an older version (or by hand, or corrupted by a power cut
    mid-write) must not be able to take the guards away: unknown keys are
    dropped and out-of-range values are pulled back into range, so the worst a
    bad file can do is give the robot its defaults.
    """
    out = defaults()
    try:
        with open(STATE_PATH, encoding='utf-8') as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        return out
    if not isinstance(saved, dict):
        return out
    for key, value in saved.items():
        clamped = clamp(key, value)
        if clamped is not None:
            out[key] = clamped
    return out


def save(settings: dict) -> None:
    """Write atomically — a half-written file is one the next boot ignores."""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump({k: settings[k] for k in sorted(settings)}, fh, indent=2)
    os.replace(tmp, STATE_PATH)


def targets(settings: dict) -> dict:
    """Regroup {key: value} into {node: {parameter: value}} — one service call
    per node instead of one per knob, and the two costmaps land together."""
    out: dict = {}
    for key, value in settings.items():
        spec = BY_KEY.get(key)
        if spec is None:
            continue
        for node, param in spec.targets:
            out.setdefault(node, {})[param] = value
    return out


def from_ros(node: str, param: str, value):
    """Reverse lookup: which knob does this node parameter belong to?

    Used when reading the values back off the live nodes, so the panel shows
    what the robot IS doing rather than what the file last asked for — the two
    differ whenever a launch argument overrides the file, or a node came up
    after the settings were applied.
    """
    for spec in SPECS:
        if (node, param) in spec.targets:
            return spec.key, clamp(spec.key, value)
    return None, None
