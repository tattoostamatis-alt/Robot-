"""Smoke tests for the home_robot package — fast consistency checks that need
no robot and no running ROS graph:

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/ -q

(Source ROS first — the launch-parse test imports `launch`.)

These would have caught, at various points: nodes missing from the CMake
install list, the stale CHANGE_ME udev placeholder, the room_mask drawn on an
older map than the active one, locations pointing into walls, and launch-file
syntax/regression errors.
"""
import glob
import os
import py_compile
import re
import subprocess

import pytest
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _active_map() -> str:
    src = open(f'{PKG}/launch/localize.launch.py').read()
    m = re.search(r"'map', default_value='([^']+)'", src)
    assert m, 'could not find the default map in localize.launch.py'
    return m.group(1)


# ── Python & shell sanity ─────────────────────────────────────────────

def test_all_python_compiles():
    for f in glob.glob(f'{PKG}/**/*.py', recursive=True):
        rel_parts = os.path.relpath(f, PKG).split(os.sep)
        # colcon --symlink-install mirrors source files and their __pycache__
        # into generated trees. Compiling those mirrors can fail solely
        # because the destination .pyc is intentionally a symlink, and also
        # checks the same source more than once. Only compile the source tree.
        if any(part in {'__pycache__', 'build', 'install', 'log'}
               for part in rel_parts):
            continue
        py_compile.compile(f, doraise=True)


def test_node_scripts_are_executable():
    for f in glob.glob(f'{PKG}/home_robot/nodes/*.py'):
        assert os.access(f, os.X_OK), f'{os.path.basename(f)} is not chmod +x'


def test_shell_scripts_parse():
    for f in (glob.glob(f'{PKG}/scripts/*.sh') + glob.glob(f'{PKG}/deploy/*.sh')
              + glob.glob(f'{PKG}/deploy/bin/*')):
        r = subprocess.run(['bash', '-n', f], capture_output=True, text=True)
        assert r.returncode == 0, f'{f}: {r.stderr}'


def test_cmakelists_installs_every_node():
    cmake = open(f'{PKG}/CMakeLists.txt').read()
    for f in glob.glob(f'{PKG}/home_robot/nodes/*.py'):
        rel = f'home_robot/nodes/{os.path.basename(f)}'
        assert rel in cmake, f'{rel} missing from CMakeLists.txt install()'


# ── Config files ──────────────────────────────────────────────────────

def test_yaml_files_valid():
    for f in (glob.glob(f'{PKG}/config/*.yaml') + glob.glob(f'{PKG}/maps/*.yaml')
              + glob.glob(f'{PKG}/deploy/config/*.yaml')):
        with open(f) as fh:
            yaml.safe_load(fh)


def test_locations_schema():
    locs = yaml.safe_load(open(f'{PKG}/config/locations.yaml'))
    assert locs, 'locations.yaml is empty'
    for name, p in locs.items():
        for k in ('x', 'y', 'yaw'):
            assert isinstance(p.get(k), (int, float)), f'{name}.{k} not numeric'


def test_udev_rules_have_no_placeholders():
    for f in glob.glob(f'{PKG}/config/*.rules') + glob.glob(f'{PKG}/deploy/config/*.rules'):
        assert 'CHANGE_ME' not in open(f).read(), f'{f} still has a CHANGE_ME placeholder'


# ── Active-map artifact consistency ───────────────────────────────────

def _load_map():
    from PIL import Image
    name = _active_map()
    meta = yaml.safe_load(open(f'{PKG}/maps/{name}.yaml'))
    img = Image.open(f'{PKG}/maps/{name}.pgm')
    return name, meta, img


def test_active_map_files_exist():
    name = _active_map()
    for ext in ('yaml', 'pgm'):
        assert os.path.exists(f'{PKG}/maps/{name}.{ext}'), f'maps/{name}.{ext} missing'


def test_room_mask_matches_active_map():
    from PIL import Image
    name, _, img = _load_map()
    mask_path = f'{PKG}/maps/{name}_room_mask.png'
    if not os.path.exists(mask_path):
        pytest.skip(f'no room mask for {name} yet — placed from the dashboard, '
                    'not required to exist')
    mask = Image.open(mask_path)
    assert mask.size == img.size, (
        f'{name}_room_mask.png {mask.size} != active map {img.size} — regenerate '
        'with scripts/auto_rooms.py (situational_awareness falls back meanwhile)')


def test_locations_on_free_space_and_in_own_room():
    import numpy as np
    from PIL import Image
    name, meta, img = _load_map()
    grid = np.array(img)
    ox, oy = meta['origin'][:2]
    res = meta['resolution']
    h, w = grid.shape
    mask_path = f'{PKG}/maps/{name}_room_mask.png'
    colours_path = f'{PKG}/maps/{name}_room_colors.yaml'
    if not os.path.exists(mask_path) or not os.path.exists(colours_path):
        pytest.skip(f'no room mask/colours for {name} yet')
    mask = np.array(Image.open(mask_path).convert('RGBA'))
    colors = yaml.safe_load(open(colours_path))
    locs = yaml.safe_load(open(f'{PKG}/config/locations.yaml'))
    for name, p in locs.items():
        col = int((p['x'] - ox) / res)
        row = h - 1 - int((p['y'] - oy) / res)
        assert 0 <= row < h and 0 <= col < w, f'{name} outside the map'
        # `dock` is the seated pose ON the base, so it lands on occupied
        # pixels by definition — the base is an obstacle to the lidar. It is
        # never a Nav2 goal (that is `dock_staging`, checked like any other
        # pose), so free space is the wrong requirement for it. Demanding it
        # is how the old, wrong dock entry looked healthy: 0.96 m from the
        # real base, but out in the open, so this test passed.
        if name != 'dock':
            assert grid[row, col] >= 250, \
                f'{name} not on free space (pixel={grid[row, col]})'
        # Not every location is a room: the charger poses sit inside whichever
        # room holds the charger. room_colors.yaml is what defines a room, so
        # only those get the own-room check.
        if name not in colors:
            continue
        if mask.shape[:2] == grid.shape:  # mask lookup only when dims match
            r, g, b, a = mask[row, col]
            assert a > 50, f'{name} on a transparent mask pixel'
            nearest = min(colors, key=lambda n: (int(r) - colors[n][0]) ** 2
                          + (int(g) - colors[n][1]) ** 2 + (int(b) - colors[n][2]) ** 2)
            assert nearest == name, f'{name} resolves to room "{nearest}"'


# ── Launch files ──────────────────────────────────────────────────────

def test_launch_files_parse():
    pytest.importorskip('launch')
    import importlib.util
    for f in glob.glob(f'{PKG}/launch/*.launch.py'):
        spec = importlib.util.spec_from_file_location(
            os.path.basename(f).replace('.', '_'), f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()
        assert ld is not None, f'{f} returned no LaunchDescription'
