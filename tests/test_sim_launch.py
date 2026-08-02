"""Tests for launch/sim.launch.py's render-engine choice.

2026-08-02: the dashboard's Gazebo tab streamed a black screen. The VNC session
was up, the ROS nodes were up, noVNC was connected — and nothing was ever drawn,
because gz died on startup:

    libEGL warning: Not allowed to force software rendering when API
                    explicitly selects a hardware device.
    ...libgz-sim-sensors-system.so -> libgz-rendering-ogre2.so
       -> RenderSystem_GL3Plus.so -> libEGL_mesa -> driCreateNewScreen3
    Segmentation fault (Address not mapped to object [0x8])

Two near-misses are worth keeping straight, because both look like the fix:

  * LIBGL_ALWAYS_SOFTWARE=1 was already set. That warning is Mesa saying it
    ignored it — the variable steers GLX, and ogre2 renders through EGL.
  * --render-engine-gui ogre fixed only the GUI. The trace above is the
    SERVER's sensors system (it renders the sim's lidar and camera) and it
    keeps its own engine, so the tab stayed black in exactly the same way.

Ogre 1.x goes through GLX, honours the software flag, and renders on the same
headless Xvnc — verified on :3 with the full window painting the house.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_sim_launch.py -q
"""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / 'launch' / 'sim.launch.py').read_text()


def _gz_mode():
    m = re.search(r'gz_mode = PythonExpression\(\[(.*?)\]\)', SRC, re.S)
    assert m, 'the gz_args mode expression is gone or changed shape'
    return m.group(1)


def _declared(name):
    m = re.search(r"DeclareLaunchArgument\('%s',\s*default_value='([^']*)'" % name, SRC)
    assert m, f'{name} is no longer declared'
    return m.group(1)


def test_the_render_engine_is_a_launch_argument():
    assert _declared('render_engine'), 'render_engine must stay overridable'


def test_it_defaults_to_ogre_one():
    """ogre2 segfaults in Mesa on every display this robot's sim runs on."""
    assert _declared('render_engine') == 'ogre', \
        'ogre2 is the engine that crashed — do not make it the default here'


def test_the_engine_reaches_gz():
    assert '--render-engine ' in _gz_mode(), \
        'gz is no longer told which engine to use'


def test_it_is_not_the_gui_only_flag():
    """--render-engine-gui leaves the server's sensor rendering on ogre2, which
    is where the segfault actually was."""
    assert '--render-engine-gui' not in _gz_mode(), \
        'the GUI-only flag does not cover the sensors system that crashed'


def test_headless_gets_the_engine_too():
    """--headless-rendering IS the sensor path that crashed, so it needs the
    engine as much as the windowed mode does."""
    mode = _gz_mode()
    head = mode.index('--headless-rendering')
    eng = mode.index('--render-engine ')
    assert head < eng, \
        'the engine must be appended after the headless branch, not inside it'
    # The headless branch must be a self-contained conditional, or the engine
    # name lands inside it and gz gets '-s --headless-rendering ogre'.
    assert re.search(r"else ''\)", mode), \
        'the headless branch must fall back to an empty string, not the engine'


def test_the_engine_is_appended_once_for_both_modes():
    mode = _gz_mode()
    assert mode.count('--render-engine ') == 1, \
        'the engine flag is built twice — one branch will disagree with the other'


def test_the_note_still_names_the_crash():
    """This is the kind of fix that gets 'cleaned up' back to the default."""
    assert 'driCreateNewScreen3' in SRC, \
        'the comment naming the actual crash is gone; keep it or this regresses'
