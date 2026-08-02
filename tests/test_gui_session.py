"""Tests for scripts/gui_session.sh — the RViz/MoveIt/Gazebo VNC sessions the
dashboard's tabs start.

Three separate faults, all found on 2026-08-02 by clicking the Gazebo tab on a
running robot and watching what happened to the live ROS graph:

 1. THE SIM SHARED THE ROBOT'S ROS GRAPH. sim.launch.py is a whole second robot
    — its own AMCL, map_server, planner, controller, collision monitor. On the
    same domain that produced 18 duplicated node names, two publishers each on
    /scan and /odom, and the sim's startup /initialpose landed in the REAL AMCL,
    teleporting the robot's pose from (-7.01, 1.17) to (-6.09, 0.49). The UI
    note "don't open it while driving the real robot" was the only guard, and it
    was wrong twice: the damage needs no driving, and a note cannot stop a click.

 2. `stop` STOPPED NOTHING. It ran `pkill -9 -f "DISPLAY=:$DISP"`, but DISPLAY
    is an environment variable — pkill -f reads /proc/PID/cmdline, where it
    never appears. So it matched zero processes every time: the X display went
    away, "stopped" was printed, and every ROS node the launch started kept
    running, reparented to init.

 3. (in sim.launch.py, see test_sim_launch.py) the GUI segfaulted in Mesa.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_gui_session.py -q
"""
import re
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parents[1] / 'scripts' / 'gui_session.sh'
SRC = SH.read_text()


def _block(name):
    """The body of one `case` branch (start|stop|status)."""
    m = re.search(r'\n  %s\)\n(.*?)\n    ;;' % name, SRC, re.S)
    assert m, f'the {name}) branch is gone or changed shape'
    return m.group(1)


def _code(text):
    """Same text with comment lines dropped — these blocks carry long notes
    that quote the very commands the tests below forbid."""
    return '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('#'))


def _xstartup(app):
    m = re.search(r'\n    %s\)\n.*?cat > "\$f" <<.?EOF.?\n(.*?)\nEOF' % app, SRC, re.S)
    assert m, f'the {app} xstartup heredoc is gone or changed shape'
    return m.group(1)


# ── it is still a valid script ──────────────────────────────────────────────

def test_the_script_parses():
    r = subprocess.run(['bash', '-n', str(SH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ── 1. the simulation cannot touch the real robot ───────────────────────────

def test_the_sim_runs_on_its_own_ros_domain():
    """Without this the sim's AMCL, planner and /initialpose reach the robot."""
    assert re.search(r'^SIM_DOMAIN_ID=(\d+)', SRC, re.M), \
        'SIM_DOMAIN_ID is gone — the sim would rejoin the robot\'s graph'


def test_the_sim_domain_is_not_the_robots():
    m = re.search(r'^SIM_DOMAIN_ID=(\d+)', SRC, re.M)
    assert m.group(1) != '0', \
        'domain 0 is the real robot (deploy/systemd/*.service)'


def test_the_gazebo_session_exports_that_domain():
    assert 'export ROS_DOMAIN_ID=' in _xstartup('gazebo'), \
        'the gazebo xstartup no longer isolates its domain'


def test_moveit_stays_on_the_real_domain():
    """MoveIt drives the arm that is actually attached — isolating it would
    make the tab a no-op instead of a hazard."""
    assert 'export ROS_DOMAIN_ID=' not in _xstartup('moveit'), \
        'moveit must talk to the real arm_driver'


def test_rviz_stays_on_the_real_domain():
    assert 'export ROS_DOMAIN_ID=' not in _xstartup('rviz'), \
        'the RViz tab visualises the real robot'


# ── 2. stop actually stops ──────────────────────────────────────────────────

def test_stop_does_not_match_on_an_environment_variable():
    """`pkill -f DISPLAY=:N` matched nothing, ever: pkill -f reads the command
    line, and DISPLAY is not on it."""
    assert not re.search(r'pkill.*-f.*DISPLAY=', _code(_block('stop'))), \
        'stop is back to matching DISPLAY= on the command line — it matches nothing'


def test_the_sessions_record_a_process_group():
    for app in ('gazebo', 'moveit'):
        body = _xstartup(app)
        assert 'set -m' in body, \
            f'{app} must run the launch in its own process group'
        assert '.pgid' in body, f'{app} does not record its PGID for stop'


def test_stop_kills_the_recorded_process_group():
    body = _block('stop')
    assert '.pgid' in body, 'stop no longer reads the PGID file'
    assert re.search(r'kill -TERM "-\$pgid"', body), \
        'stop must signal the whole process group (negative PID)'
    assert re.search(r'kill -9 "-\$pgid"', body), \
        'stop needs a SIGKILL fallback for what ignores SIGTERM'


def test_stop_gives_ros_a_chance_to_shut_down_first():
    body = _block('stop')
    term = body.index('kill -TERM')
    kill9 = body.index('kill -9 "-$pgid"')
    assert term < kill9, 'SIGKILL must not come before SIGTERM'
    assert 'sleep' in body[term:kill9], 'nothing waits between SIGTERM and SIGKILL'


def test_stop_kills_the_group_before_dropping_the_display():
    """Once Xvnc is gone the pid file is the only handle left, and the old order
    lost it."""
    body = _block('stop')
    assert body.index('kill -TERM') < body.index('tigervncserver -kill'), \
        'the process group must be signalled before the display is torn down'


def test_stop_removes_the_stale_pgid_file():
    assert 'rm -f "$pgidf"' in _block('stop'), \
        'a stale PGID could be reused by an unrelated process'


def test_stopping_rviz_is_still_refused():
    """:2 is the session `robot max` starts for the phone; a browser tab must
    not be able to kill it."""
    body = _block('stop')
    assert re.search(r'if \[ "\$APP" = rviz \]', body) and 'refusing' in body


def test_an_orphaned_gz_server_is_still_swept():
    """It holds /clock, and the next start then falls back to a namespaced one
    and dies silently."""
    assert re.search(r"pkill -9 -f 'gz sim'", _block('stop'))
