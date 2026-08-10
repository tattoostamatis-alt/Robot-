"""Where a map's room mask/colours live, and how to load them.

One mask used to serve every map (maps/room_mask.png + room_colors.yaml), so
switching maps from the web Settings tab silently left the OLD map's rooms
painted under the NEW map's frame — wrong size, wrong origin, and on maps
that happen to share a resolution, no error at all, just wrong colours in
the wrong places. Scoping the files by map name (maps/<name>_room_mask.png)
makes "which rooms" follow "which map" automatically; no rebuild.
"""
import os
import re
import subprocess

import numpy as np
import yaml

PKG = os.path.expanduser('~/robot_ws/src/home_robot')


def maps_dir() -> str:
    """Prefer the checked-out src tree. share/ under install/ is a
    symlink-install of files that existed at BUILD time, so a brand-new
    per-map filename would need a colcon build before any node could see
    it there — the src tree has no such lag.
    """
    src = os.path.join(PKG, 'maps')
    if os.path.isdir(src):
        return src
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('home_robot'), 'maps')


def paths_for(map_name: str):
    """(mask_path, colours_path) for a given saved map's name."""
    d = maps_dir()
    return (os.path.join(d, f'{map_name}_room_mask.png'),
            os.path.join(d, f'{map_name}_room_colors.yaml'))


def load_raw(map_name: str):
    """(rgba_arr, names) for map_name, or (None, None) if unset. rgba_arr is
    the mask exactly as painted — [row, col] -> (r, g, b, a), alpha>50 meaning
    painted. This is the shape room_markers_node/situational_awareness_node
    already index directly; see load() below for the dashboard's cv2-style
    (BGR + boolean-alpha) variant of the same file.
    """
    mask_path, colours_path = paths_for(map_name)
    try:
        from PIL import Image
        arr = np.array(Image.open(mask_path).convert('RGBA'))
    except Exception:
        return None, None
    names = {}
    if os.path.exists(colours_path):
        try:
            with open(colours_path) as f:
                names = yaml.safe_load(f) or {}
        except Exception:
            names = {}
    return arr, names


def load(map_name: str):
    """(bgr, alpha, names) for map_name, or (None, None, None) if unset."""
    arr, names = load_raw(map_name)
    if arr is None:
        return None, None, None
    rgb = arr[:, :, :3].astype(np.float32)
    bgr = rgb[:, :, ::-1]
    alpha = arr[:, :, 3] > 50
    return bgr, alpha, names


def active_map_name(timeout: float = 8.0):
    """The map map_server currently has loaded, or None. Blocking (shells out
    to `ros2 param get`) — fine at node startup or from a rare/latched
    callback, wrong to call from a hot per-frame path.
    """
    try:
        r = subprocess.run(
            ['ros2', 'param', 'get', '/map_server', 'yaml_filename'],
            capture_output=True, text=True, timeout=timeout)
        hit = re.search(r'([A-Za-z0-9_-]+)\.yaml', r.stdout or '')
        return hit.group(1) if hit else None
    except Exception:
        return None
