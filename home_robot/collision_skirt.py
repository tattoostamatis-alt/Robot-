"""The collision_monitor hard-stop skirt, in one place, editable from the web.

config/nav2_params.yaml's `collision_monitor: Stop` polygon zeroes cmd_vel the
instant the lidar sees something inside it — the last-resort stop, independent
of Nav2's planner (see safety_settings.py for the live-adjustable planner-side
`inflation_radius`, which is a different clearance from this one). Nav2 reads
this polygon once at on_configure, so it can only be changed by rewriting the
YAML and restarting — this module is the rewrite, kept out of the dashboard
node so it can be tested without rclpy.

## Scope: moving rings only, never `stopped_or_rotating`

The Stop polygon has three sub-rings (`moving_forward`, `moving_backward`,
`stopped_or_rotating`). Only the two moving ones are generated here.
`stopped_or_rotating` stays exactly as nav2_params.yaml already has it — it is
the anti-deadlock exception fixed on 2026-08-08 after a live Gazebo A/B test
(see the comments above it in nav2_params.yaml and
tests/test_collision_monitor_fits.py's test_rotation_is_not_gated_like_translation):
a circle rotating about its own centre sweeps no new ground, so its ring only
has to catch actual contact and is deliberately much smaller than the moving
rings. Re-deriving it from a user-chosen margin would risk reintroducing that
exact deadlock for a ring nobody asked to change.

## The allowed range, and why it is not "obvious"

Both bounds come from tests/test_collision_monitor_fits.py's invariants, not
from taste — verified by a full sweep in tests/test_collision_skirt.py:

  * floor 30 mm: below ~25.4 mm, the moving ring's min_reach gets too close to
    the FIXED stopped_or_rotating ring's min_reach (0.180 m) to keep the 20 mm
    gap test_rotation_is_not_gated_like_translation requires — the intuitive
    "just don't go below the chassis" floor of ~0 mm is wrong here.
  * ceiling 60 mm: above ~66 mm the moving ring's outer reach exceeds
    TIGHTEST_PASSAGE (0.241 m, the toualeta doorway on malou2) and the robot
    would refuse doorways it physically fits through.
"""
import ast
import math
import os
import re

import yaml

MARGIN_MIN_MM = 30
MARGIN_MAX_MM = 60
MARGIN_STEP_MM = 5
MARGIN_DEFAULT_MM = 40
ALLOWED_MARGINS_MM = tuple(range(MARGIN_MIN_MM, MARGIN_MAX_MM + 1, MARGIN_STEP_MM))

# The physical chassis, Ø0.35 m (re-measured 2026-07-31) — matches
# local_costmap/global_costmap robot_radius in nav2_params.yaml. NOT
# adjustable from here: this module only moves the margin added on top.
CHASSIS_R = 0.175

_N_POINTS = 16
_BLOCKS = ('moving_forward', 'moving_backward')


def circle_points(radius: float, n: int = _N_POINTS) -> str:
    """The `points: "[[...]]"` string collision_monitor's velocity_polygon
    wants: an n-gon inscribed in `radius`, starting on the +x axis, matching
    the formatting (4 decimals) already in nav2_params.yaml exactly."""
    pts = []
    for k in range(n):
        angle = 2 * math.pi * k / n
        x = round(radius * math.cos(angle), 4)
        y = round(radius * math.sin(angle), 4)
        x = 0.0 if x == 0 else x   # -0.0000 from floating-point cos/sin noise
        y = 0.0 if y == 0 else y
        pts.append(f'[{x:.4f}, {y:.4f}]')
    return '[' + ', '.join(pts) + ']'


def margin_to_radius(margin_mm: float) -> float:
    return CHASSIS_R + margin_mm / 1000.0


def current_margin_mm(yaml_text: str) -> int | None:
    """The moving rings' margin the file currently has, in whole mm, or None
    if the file doesn't parse or is missing the block."""
    try:
        params = yaml.safe_load(yaml_text)
        pts_str = (params['collision_monitor']['ros__parameters']['Stop']
                   ['moving_forward']['points'])
        pts = [tuple(p) for p in ast.literal_eval(pts_str)]
    except Exception:
        return None
    max_reach = max(math.hypot(x, y) for x, y in pts)
    return round((max_reach - CHASSIS_R) * 1000)


def default_params_path() -> str:
    """Where nav2_params.yaml actually lives.

    Prefers the source tree (this workspace uses `colcon build
    --symlink-install`, so the installed share copy IS this file, not a
    separate one that a rebuild could silently revert to) — the same
    precedence web_dashboard_node.py's SRC_CONFIG_DIR already uses, kept
    here too rather than guessed a second time.
    """
    src = os.path.expanduser('~/robot_ws/src/home_robot/config/nav2_params.yaml')
    if os.path.exists(src):
        return src
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('home_robot'),
                        'config', 'nav2_params.yaml')


def _points_pattern(block_name: str) -> re.Pattern:
    # Anchored on the block's own key so moving_forward and moving_backward
    # (which currently hold IDENTICAL point strings) can each be replaced
    # independently, and so this never touches stopped_or_rotating or
    # anything outside the collision_monitor Stop polygon.
    return re.compile(
        r'(' + re.escape(block_name) + r':\s*\n\s*points:\s*")'
        r'\[\[[^"]*\]\]'
        r'(")')


def patch_moving_points(yaml_text: str, margin_mm: float) -> str:
    """Replace moving_forward's and moving_backward's `points:` strings with
    a circle of radius CHASSIS_R + margin_mm, leaving every other character —
    including the extensive history comments — untouched.

    Raises ValueError if a block isn't found exactly once, rather than
    silently leaving the file half-patched or patching the wrong occurrence.
    """
    new_points = circle_points(margin_to_radius(margin_mm))
    text = yaml_text
    for block in _BLOCKS:
        pattern = _points_pattern(block)
        matches = pattern.findall(text)
        if len(matches) != 1:
            raise ValueError(
                f'expected exactly one {block!r} points block, found '
                f'{len(matches)} — refusing to patch nav2_params.yaml')
        text = pattern.sub(lambda m: m.group(1) + new_points + m.group(2),
                           text, count=1)
    return text
