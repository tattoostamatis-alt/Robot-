"""Arm envelope + speed, one place — mirrors safety_settings.py's design.

Split out from safety_settings.py rather than added to it: a safety knob is
always one number written to one or two nodes. Here a joint limit is a PAIR
([lo, hi]) written to arm_driver AND (for every joint arm_joy actually jogs)
arm_joy too, and the speed knob is one percentage fanned out to six real
parameters across two nodes with two different units (rad/s and a 0-254
accel code) — different enough shapes that folding them into safety_settings'
Spec/clamp/targets would have made that file harder to read for no shared
code.

## What "speed" means here

arm_driver's own `speed` parameter (steps/s cruise cap) is left alone. It
defaults to 0 ("firmware default") and nobody has measured what a *safe*
non-zero ceiling is on the real servos — turning that dial up would be
sending a made-up number at hardware nobody has characterised. What IS
well-understood, because it has been hand-tuned against the real arm before
(arm_joy's scale_* was doubled on 2026-07-29 after all five axes were
confirmed tracking on hardware), is:

  * arm_joy's scale_<joint> / scale_gripper — rad/s the joystick jog
    integrates per second of stick deflection.
  * arm_driver's accel — the T:102/T:106 ramp, 0-254 in 100 steps/s^2,
    0 = instant. LOWER accel reaches full speed sooner, so "faster" maps to
    a SMALLER accel — inversely, not directly.

One percentage slider (50%-150%, 100% = today's tuned defaults) scales the
first group directly and the second inversely, so the dashboard, the
joystick and any future arm/joint_cmd publisher all turn the same "how fast"
knob together.

## What "limits" means here

Two different envelopes exist per joint, and this module must never mix them
up:

  * MECH_LIMITS — the servo's physical travel (Waveshare wiki). A limit is
    NEVER allowed wider than this; see arm_driver.py's own `limits()`.
  * DEFAULT_LIMITS — the USABLE envelope, hand-measured 2026-07-31 with the
    servo torque cut (see config/arm_joy_ps5.yaml's header for how). This is
    what bringup.launch.py and arm_joy_ps5.yaml ship today, and what "reset"
    restores — resetting to MECH_LIMITS instead would open the arm back up to
    swinging into the chassis, the D435 or the LiDAR scan plane, which is
    exactly what the measured envelope exists to prevent.

arm_joy never jogs 'roll' (no stick axis is bound to it) and so never
declares a limit_roll parameter at all — writing one there fails outright.
"""
import json
import os

STATE_PATH = os.path.expanduser('~/.home_robot/arm_settings.json')

# Mirrors arm_driver.MECH_LIMITS. Duplicated for the same reason arm_joy
# duplicates it: these nodes install as standalone scripts, not an importable
# package, so there is no shared module boundary to import across.
MECH_LIMITS = {
    'base':     (-3.14, 3.14),
    'shoulder': (-1.57, 1.57),
    'elbow':    (0.0, 3.14),
    'wrist':    (-1.57, 1.57),
    'roll':     (-3.14, 3.14),
    'hand':     (1.08, 3.14),
}

# The hand-measured usable envelope — see the module docstring. Identical to
# bringup.launch.py's arm_node parameters and config/arm_joy_ps5.yaml's
# limit_*; kept here a third time for the same reason ARM_LIMITS is kept a
# third time in web_dashboard_node.py (no importable boundary between them).
DEFAULT_LIMITS = {
    'base':     (-3.015, 0.016),
    'shoulder': (-0.169, 1.570),
    'elbow':    (0.141, 3.079),
    'wrist':    (-1.143, 1.407),
    'roll':     (-1.165, 1.372),
    'hand':     (1.080, 3.140),
}

# arm_joy has no axis bound to 'roll' and never declares limit_roll.
_ARM_JOY_JOINTS = ('base', 'shoulder', 'elbow', 'wrist', 'hand')

# rad/s at 100% — arm_joy's own tuned defaults (config/arm_joy_ps5.yaml).
SPEED_BASE_SCALE = {
    'base': 1.20, 'shoulder': 1.00, 'elbow': 1.00, 'wrist': 1.00, 'gripper': 0.80,
}
SPEED_BASE_ACCEL = 10          # arm_driver's declared default for 'accel'
ACCEL_LO, ACCEL_HI = 1, 60     # never 0 (instant/no ramp) or a near-standstill crawl

SPEED_LO, SPEED_HI, SPEED_STEP, SPEED_DEFAULT = 0.5, 1.5, 0.05, 1.0


def clamp_speed(pct):
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        return None
    if pct != pct:                      # NaN survives every comparison above
        return None
    return round(min(SPEED_HI, max(SPEED_LO, pct)), 3)


def speed_from_accel(accel):
    """Reverse of speed_targets' accel formula — for showing what the arm is
    actually running, not just what was last asked for."""
    try:
        accel = float(accel)
    except (TypeError, ValueError):
        return None
    if accel <= 0:
        return None
    return clamp_speed(SPEED_BASE_ACCEL / accel)


def speed_targets(pct) -> dict:
    """{node: {param: value}} for one speed percentage."""
    clamped = clamp_speed(pct)
    if clamped is None:
        return {}
    joy = {f'scale_{j}': round(v * clamped, 3) for j, v in SPEED_BASE_SCALE.items()}
    accel = max(ACCEL_LO, min(ACCEL_HI, int(round(SPEED_BASE_ACCEL / clamped))))
    return {'/arm_joy': joy, '/arm_driver': {'accel': accel}}


def clamp_limit(joint: str, lo, hi):
    if joint not in MECH_LIMITS:
        return None
    lo_m, hi_m = MECH_LIMITS[joint]
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if lo != lo or hi != hi:            # NaN
        return None
    if lo > hi:
        lo, hi = hi, lo
    return (round(max(lo, lo_m), 4), round(min(hi, hi_m), 4))


def limit_targets(joint: str, lo, hi) -> dict:
    """{node: {param: [lo, hi]}} — arm_driver always, arm_joy only for a
    joint it actually jogs (see _ARM_JOY_JOINTS)."""
    clamped = clamp_limit(joint, lo, hi)
    if clamped is None:
        return {}
    out = {'/arm_driver': {f'limit_{joint}': list(clamped)}}
    if joint in _ARM_JOY_JOINTS:
        out['/arm_joy'] = {f'limit_{joint}': list(clamped)}
    return out


def defaults() -> dict:
    out = {'speed': SPEED_DEFAULT}
    for j, (lo, hi) in DEFAULT_LIMITS.items():
        out[f'limit_{j}'] = [lo, hi]
    return out


def load() -> dict:
    """Saved settings, clamped, with anything missing filled from defaults.

    A file written by an older version (or corrupted by a power cut mid-write)
    must not be able to widen a joint past MECH_LIMITS: unknown/malformed
    values are dropped in favour of the hand-measured default, so the worst a
    bad file can do is give the arm back its known-safe envelope.
    """
    out = defaults()
    try:
        with open(STATE_PATH, encoding='utf-8') as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        return out
    if not isinstance(saved, dict):
        return out
    if 'speed' in saved:
        v = clamp_speed(saved['speed'])
        if v is not None:
            out['speed'] = v
    for j in MECH_LIMITS:
        v = saved.get(f'limit_{j}')
        if isinstance(v, (list, tuple)) and len(v) == 2:
            clamped = clamp_limit(j, v[0], v[1])
            if clamped is not None:
                out[f'limit_{j}'] = list(clamped)
    return out


def save(settings: dict) -> None:
    """Write atomically — a half-written file is one the next boot ignores."""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump({k: settings[k] for k in sorted(settings)}, fh, indent=2)
    os.replace(tmp, STATE_PATH)


def all_targets(settings: dict) -> dict:
    """Regroup the whole saved dict into {node: {param: value}}, one service
    call per node — mirrors safety_settings.targets()."""
    out: dict = {}
    for j in MECH_LIMITS:
        key = f'limit_{j}'
        if key in settings:
            for node, params in limit_targets(j, *settings[key]).items():
                out.setdefault(node, {}).update(params)
    if 'speed' in settings:
        for node, params in speed_targets(settings['speed']).items():
            out.setdefault(node, {}).update(params)
    return out
