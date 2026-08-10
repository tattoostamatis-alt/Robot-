"""Turn the two costmap keepout_filter layers on/off in nav2_params.yaml.

Nav2's KeepoutFilter needs BOTH things true at once to actually apply, and
they live in two different places: `use_keepout:=true` at launch (starts
filter_mask_server + costmap_filter_info_server — see launch/bringup.launch.py)
AND `enabled: True` on each costmap's keepout_filter layer, in this file.
Passing the launch flag alone leaves the layer compiled in but inert; setting
`enabled: True` alone starts a layer that waits forever for filter info nobody
publishes and WARNs every costmap update. Both restart-only, like
collision_skirt.py's margin — Nav2 reads costmap plugin config once in
on_configure, so nothing short of a relaunch picks this up.

Kept as a text patch (not yaml.safe_dump the whole file) for the same reason
collision_skirt.py is: nav2_params.yaml carries load-bearing history comments
this module has no business rewriting.
"""
import re

from home_robot.collision_skirt import default_params_path  # noqa: F401

_ENABLED_LINE = re.compile(
    r'(keepout_filter:\n(?:[^\n]*\n)*?\s*enabled:\s*)(True|False)', re.M)


def current_enabled(yaml_text: str) -> bool | None:
    """Whether both keepout_filter blocks currently say `enabled: True`, or
    None if the file doesn't have exactly two (local + global costmap)."""
    matches = _ENABLED_LINE.findall(yaml_text)
    if len(matches) != 2:
        return None
    states = {m[1] for m in matches}
    if len(states) != 1:
        return None       # local/global disagree — treat as indeterminate
    return states.pop() == 'True'


def patch_enabled(yaml_text: str, on: bool) -> str:
    """Set both costmaps' keepout_filter `enabled:` to `on`, leaving every
    other character untouched. Raises ValueError if the file doesn't have
    exactly two such blocks (local_costmap + global_costmap) — refusing to
    patch a file that doesn't look like nav2_params.yaml is safer than
    silently patching the wrong number of them.
    """
    matches = _ENABLED_LINE.findall(yaml_text)
    if len(matches) != 2:
        raise ValueError(
            f'expected exactly 2 keepout_filter enabled: blocks, found '
            f'{len(matches)} — refusing to patch nav2_params.yaml')
    value = 'True' if on else 'False'
    return _ENABLED_LINE.sub(lambda m: m.group(1) + value, yaml_text)
