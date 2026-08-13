"""The microphone's tunable settings, in one place — mirrors safety_settings.py.

Every knob here maps to a single scalar ROS parameter on a single node, the
same shape safety_settings.py already handles, so this file reuses that
module's Spec/clamp/load/save/targets/from_ros wholesale rather than
redefining an identical set of functions under a new name. What's new is only
SPECS itself, plus a dedicated state file and an INFO_ONLY table.

## Why these and not the others

wake_word_node.py and stt_node.py declare a lot more parameters than this
(device/channel routing, model paths, watchdog timings). Two things ruled
most of them out:

  * Hardware routing (mic_channel, stt_channel, device_index, ...) is
    calibrated per physical unit against a measured SNR table (see
    wake_word_node.py's module docstring) — a slider that can silently pick
    the wrong channel is worse than no slider.
  * Watchdog/liveness timings (vad_stuck_s, audio_timeout, busy_timeout, ...)
    each exist because of a specific "the robot went silently deaf" bug fixed
    at that exact number — see stt_node.py's own comments. Retuning them from
    a slider is how you reintroduce the bug that number was chosen to fix.

## Live or it doesn't belong here

Every SPECS entry actually took effect on the real node the moment it was
checked (see [[project_robot]] session notes): wake_word_node.py had NO
parameter callback at all before this, so `threshold`/`allow_barge_in`/
`barge_in_threshold`/`highpass_hz` were each read exactly once at startup and
silently ignored every `ros2 param set` after — the callback added alongside
this file (`WakeWordNode._on_param_change`) is what makes them real. Same
story for stt_node's `use_hw_vad`, added to its existing `_on_param_change`.
`energy_thresh` was already live. A slider for a parameter that isn't wired
up to change anything is worse than no slider — see safety_settings.py's own
docstring on this.

## Two more Spec kinds than safety_settings has

`led_brightness` (0-255) and the five `led_color_*` knobs (0xRRGGBB, picked
from a colour swatch rather than dragged on a range slider) don't fit
safety_settings' 'double'/'bool' kinds — hence 'int' and 'color' below, and
why clamp() is a full copy rather than a thin wrapper around that module's.
Both still map to a single ROS parameter on a single node, same as
everything else here; doa_node declares them as plain Python ints, so the
dashboard has to send PARAMETER_INTEGER for these, not PARAMETER_DOUBLE (see
_mic_request in web_dashboard_node.py).

`wake_model` is stranger still — a closed set of strings — and isn't a Spec
at all; see the comment above WAKE_MODEL_CHOICES for why, and key_targets()
for how it slots into targets()/from_ros() alongside the real Specs anyway.
"""
import json
import os

# The Spec shape (key/kind/default/targets/lo/hi/step/warn_*) is generic
# enough to reuse as-is; clamp()/load()/save()/targets() are NOT reused from
# there, because safety_settings.clamp() closes over THAT module's own
# BY_KEY — calling it with a mic_settings key would look the key up in the
# wrong table and silently return None for every knob in this file.
from home_robot.safety_settings import Spec

STATE_PATH = os.path.expanduser('~/.home_robot/mic_settings.json')

SPECS = (
    # ── wake word ────────────────────────────────────────────────────────────
    Spec('wake_threshold', 'double', 0.50,
         (('/wake_word_node', 'threshold'),),
         lo=0.20, hi=0.90, step=0.05, warn_below=0.30, warn_above=0.80),
    # Barge-in measured 46/81 self-triggers with zero real speech following
    # (see wake_word_node.py's docstring) — off by default, and the node
    # disarms itself again after 3 more empty interruptions even if turned
    # back on from here.
    Spec('barge_in_enabled', 'bool', False,
         (('/wake_word_node', 'allow_barge_in'),)),
    Spec('barge_in_threshold', 'double', 0.70,
         (('/wake_word_node', 'barge_in_threshold'),),
         lo=0.50, hi=0.95, step=0.05, warn_below=0.55),
    # 0 disables the filter outright. Measured on this rig: 8-17 false wake
    # triggers per 20 s of silence at 0 Hz, zero at 250 Hz — see
    # wake_word_node.py's declare_parameter comment for the full measurement.
    Spec('highpass_hz', 'double', 250.0,
         (('/wake_word_node', 'highpass_hz'),),
         lo=0.0, hi=500.0, step=10.0),

    # ── speech-to-text ────────────────────────────────────────────────────────
    Spec('energy_thresh', 'double', 0.065,
         (('/stt_node', 'energy_thresh'),),
         lo=0.010, hi=0.200, step=0.005, warn_below=0.015),
    Spec('use_hw_vad', 'bool', True,
         (('/stt_node', 'use_hw_vad'),)),

    # ── privacy mute ─────────────────────────────────────────────────────────
    # The ALSA stream stays open (see wake_word_node.py) — this only gates
    # what leaves _audio_callback, so un-muting is instant.
    Spec('mic_muted', 'bool', False,
         (('/wake_word_node', 'muted'),)),

    # ── LED ring (XVF3800, via doa_node) ─────────────────────────────────────
    Spec('led_enabled', 'bool', True,
         (('/doa_node', 'led_enabled'),)),
    Spec('led_brightness', 'int', 150,
         (('/doa_node', 'led_brightness'),),
         lo=0, hi=255, step=5),
    Spec('led_color_idle_base', 'color', 0x111111,
         (('/doa_node', 'led_color_idle_base'),)),
    Spec('led_color_idle_pointer', 'color', 0x0000FF,
         (('/doa_node', 'led_color_idle_pointer'),)),
    Spec('led_color_listening', 'color', 0x0000FF,
         (('/doa_node', 'led_color_listening'),)),
    Spec('led_color_processing', 'color', 0xFF6600,
         (('/doa_node', 'led_color_processing'),)),
    # Shown on the ring whenever mic_muted is on, overriding whichever of the
    # three states above the voice pipeline thinks it's in (see doa_node's
    # _apply_led_state).
    Spec('led_color_muted', 'color', 0xFF0000,
         (('/doa_node', 'led_color_muted'),)),
)

BY_KEY = {s.key: s for s in SPECS}

# ── wake word MODEL choice ───────────────────────────────────────────────────
# NOT a Spec: it's a closed set of strings, and Spec/clamp() above only know
# double/bool/int/color. 'hey_robot' is the one actually trained for this
# house's wake phrase ("Έι ρομπότ" — see wake_word_node.py's module
# docstring); the rest are openWakeWord's bundled PRETRAINED English models,
# usable instantly with no training step, at the cost of being someone else's
# assistant name rather than this robot's. Waking to a genuinely new phrase
# needs a newly trained .onnx (training/wake_word_hey_robot/README) — that is
# why this is a fixed picker and not a free-text field: typing a name in
# does not make the detector recognise it.
WAKE_MODEL_CHOICES = ('hey_robot', 'alexa', 'hey_jarvis', 'hey_mycroft', 'hey_marvin')
WAKE_MODEL_DEFAULT = 'hey_robot'
WAKE_MODEL_TARGET = ('/wake_word_node', 'model_name')


def clamp_wake_model(value):
    return value if value in WAKE_MODEL_CHOICES else None

# calibrate_on_start (stt_node.py) overwrites energy_thresh with a fresh
# ambient-noise measurement on every boot — a saved slider value would look
# ignored the moment the robot restarts, which is the exact "looks live but
# isn't" trap this module exists to avoid. Shown for context, not editable.
INFO_ONLY = (
    {'key': 'auto_calibrated_energy', 'value': True,
     'source': 'stt_node / calibrate_on_start',
     'note': 'energy_thresh is re-measured from ambient noise on every boot; '
             'a value set here is overwritten at the next restart.'},
)


def clamp(key: str, value):
    """Coerce a value from the browser into something the spec allows —
    same logic as safety_settings.clamp(), against THIS module's BY_KEY."""
    if key == 'wake_model':
        return clamp_wake_model(value)
    spec = BY_KEY.get(key)
    if spec is None:
        return None
    if spec.kind == 'bool':
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        return None
    if spec.kind == 'int':
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(max(spec.lo, min(spec.hi, round(value))))
    if spec.kind == 'color':
        # The browser sends "#rrggbb" (an <input type=color>'s .value) or,
        # from a saved file, a plain int — accept either.
        if isinstance(value, str):
            try:
                value = int(value.lstrip('#'), 16)
            except ValueError:
                return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(max(0, min(0xFFFFFF, round(value))))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if v != v:                     # NaN survives every comparison below
        return None
    return round(min(spec.hi, max(spec.lo, v)), 4)


def defaults() -> dict:
    out = {s.key: s.default for s in SPECS}
    out['wake_model'] = WAKE_MODEL_DEFAULT
    return out


def load() -> dict:
    """Saved settings, clamped, with anything missing filled from defaults —
    same reasoning as safety_settings.load(): a bad or old file must not be
    able to push a value outside its Spec's range, only fall back to it."""
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
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump({k: settings[k] for k in sorted(settings)}, fh, indent=2)
    os.replace(tmp, STATE_PATH)


def key_targets(key: str, value) -> dict:
    """{node: {param: value}} for ONE key — targets() is just this, merged."""
    if key == 'wake_model':
        node, param = WAKE_MODEL_TARGET
        return {node: {param: value}}
    spec = BY_KEY.get(key)
    if spec is None:
        return {}
    out: dict = {}
    for node, param in spec.targets:
        out.setdefault(node, {})[param] = value
    return out


def targets(settings: dict) -> dict:
    out: dict = {}
    for key, value in settings.items():
        for node, params in key_targets(key, value).items():
            out.setdefault(node, {}).update(params)
    return out


def from_ros(node: str, param: str, value):
    if (node, param) == WAKE_MODEL_TARGET:
        return 'wake_model', clamp_wake_model(value)
    for spec in SPECS:
        if (node, param) in spec.targets:
            return spec.key, clamp(spec.key, value)
    return None, None
