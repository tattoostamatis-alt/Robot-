"""Keepout zones: per-map vector source (maps/<name>_keepout_zones.yaml) plus
rasterization into the fixed mask Nav2's filter_mask_server actually loads
(config/keepout_mask.pgm/.yaml — see launch/bringup.launch.py's yaml_filename
override).

Zones are per-map for the same reason rooms are (see room_files.py's
docstring): map-frame coordinates from one map are meaningless painted onto
another map's frame. The RASTER mask stays a single fixed pair of files
because that is the path Nav2's launch-time filter_mask_server is wired to —
render_active_mask() below regenerates it for whichever map is active,
on every edit and again right before a keepout-enabled restart, so it never
goes stale relative to maps/<active>_keepout_zones.yaml.

Shared by scripts/draw_keepout.py (CLI) and web_dashboard_node.py (the map
tab's "click defines a zone" tool) — one rasterizer, not two that can drift.
"""
import os

import numpy as np
import yaml

PKG = os.path.expanduser('~/robot_ws/src/home_robot')

FREE = 254       # map_saver convention: passable
BLOCKED = 0      # occupied — what KeepoutFilter treats as a hard obstacle


def maps_dir() -> str:
    # Same src-preferring reasoning as room_files.maps_dir(): a per-map zones
    # file is only ever written here, and share/ under install/ is a
    # symlink-install of files that existed at BUILD time.
    src = os.path.join(PKG, 'maps')
    if os.path.isdir(src):
        return src
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('home_robot'), 'maps')


def zones_path(map_name: str) -> str:
    return os.path.join(maps_dir(), f'{map_name}_keepout_zones.yaml')


def load_zones(map_name: str) -> dict:
    path = zones_path(map_name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return (yaml.safe_load(f) or {}).get('zones') or {}
    except Exception:
        return {}


def save_zones(map_name: str, zones: dict) -> None:
    path = zones_path(map_name)
    with open(path, 'w') as f:
        f.write('# Keepout zones for this map — areas the robot must NOT '
                'navigate (map frame).\n'
                '# Edited from the dashboard\'s Χάρτες tab, or by hand here.\n'
                '# Shapes: rect (x,y=centre, width,height) | circle (x,y,radius), metres.\n')
        yaml.safe_dump({'zones': zones}, f, allow_unicode=True, sort_keys=False,
                       default_flow_style=False)


def _world_to_pixel(wx, wy, ox, oy, res, height):
    """Map-frame (x, y) -> pixel (col, row). Row 0 = top of PGM = max y —
    same convention as room_files.py / auto_rooms.py."""
    col = int(round((wx - ox) / res))
    row = height - 1 - int(round((wy - oy) / res))
    return col, row


def rasterize(zones: dict, width: int, height: int, origin_xy, res: float) -> np.ndarray:
    """zones -> a (height, width) uint8 grid, FREE everywhere except the
    zones (BLOCKED). Pure function, no I/O — used by both render_active_mask
    and tests.
    """
    ox, oy = origin_xy
    grid = np.full((height, width), FREE, dtype=np.uint8)
    for z in zones.values():
        shape = str(z.get('shape', 'rect')).lower()
        wx, wy = float(z['x']), float(z['y'])
        cx, cy = _world_to_pixel(wx, wy, ox, oy, res, height)
        if shape == 'circle':
            r_px = max(1, int(round(float(z['radius']) / res)))
            r0, r1 = max(0, cy - r_px), min(height, cy + r_px + 1)
            c0, c1 = max(0, cx - r_px), min(width, cx + r_px + 1)
            for row in range(r0, r1):
                for col in range(c0, c1):
                    if (col - cx) ** 2 + (row - cy) ** 2 <= r_px ** 2:
                        grid[row, col] = BLOCKED
        else:
            w_px = max(1, int(round(float(z['width']) / res)))
            h_px = max(1, int(round(float(z['height']) / res)))
            r0, r1 = max(0, cy - h_px // 2), min(height, cy + h_px // 2 + 1)
            c0, c1 = max(0, cx - w_px // 2), min(width, cx + w_px // 2 + 1)
            grid[r0:r1, c0:c1] = BLOCKED
    return grid


def mask_paths():
    """Where Nav2's filter_mask_server actually reads from — fixed, not
    per-map (see module docstring)."""
    from home_robot.collision_skirt import default_params_path
    # config/ is default_params_path()'s directory (nav2_params.yaml lives
    # next to keepout_mask.*) — reuse rather than re-deriving SRC_CONFIG_DIR.
    config_dir = os.path.dirname(default_params_path())
    return (os.path.join(config_dir, 'keepout_mask.pgm'),
            os.path.join(config_dir, 'keepout_mask.yaml'))


def render_active_mask(map_name: str, map_yaml_path: str, map_pgm_shape) -> None:
    """Regenerate config/keepout_mask.pgm/.yaml for map_name from its saved
    zones. map_pgm_shape is (width, height) — read by the caller, which
    already has the map's PIL/cv2 image open for other reasons.
    """
    from PIL import Image
    with open(map_yaml_path) as f:
        meta = yaml.safe_load(f)
    res = float(meta['resolution'])
    ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
    width, height = map_pgm_shape
    zones = load_zones(map_name)
    grid = rasterize(zones, width, height, (ox, oy), res)

    pgm_path, yaml_path = mask_paths()
    Image.fromarray(grid, mode='L').save(pgm_path)
    with open(yaml_path, 'w') as f:
        f.write(
            '# Auto-generated — do not edit directly.\n'
            f'# Rendered from {os.path.basename(zones_path(map_name))} for map "{map_name}".\n'
            'image: keepout_mask.pgm\n'
            'mode: scale\n'
            f'resolution: {res}\n'
            f'origin: [{ox}, {oy}, 0.0]\n'
            'negate: 0\n'
            'occupied_thresh: 0.65\n'
            'free_thresh: 0.25\n')
