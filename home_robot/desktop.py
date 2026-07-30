"""Desktop-session access for the voice assistant: what time it is, what is
open on screen, and launching an app by voice.

Everything here runs against the user's GNOME session, which the robot's nodes
are NOT part of — they start from a tty or over ssh, with no DISPLAY and no
session bus. `session_env()` lifts those variables out of a process that IS in
the session (the gnome-settings-daemon helpers), which is the same trick used
to open RViz on the desk monitor.

‼️ Wayland limits what is knowable. GNOME 46 refuses
org.gnome.Shell.Introspect.GetWindows to anything but its own portal
("GetWindows is not allowed"), and there is no other supported way to enumerate
native Wayland windows — so `list_windows()` sees XWayland clients through
wmctrl plus running GUI processes matched to their .desktop entries. A native
Wayland window with no matching process name is invisible to it. Getting the
full list needs a GNOME shell extension (e.g. Window Calls) installed and the
session restarted; that is a user decision, not something to do behind their
back.
"""
import glob
import os
import subprocess
import time

# Voice-facing name -> desktop entry. Kept short and checked against what is
# actually installed, so the model is never offered an app that cannot start.
KNOWN_APPS = {
    'firefox': 'firefox_firefox.desktop',
    'files': 'org.gnome.Nautilus.desktop',
    'terminal': 'org.gnome.Terminal.desktop',
    'text_editor': 'org.gnome.TextEditor.desktop',
    'calculator': 'org.gnome.Calculator.desktop',
    'pdf_viewer': 'org.gnome.Evince.desktop',
    'video_player': 'org.gnome.Totem.desktop',
    'vscode': 'code_code.desktop',
}

_SEARCH_DIRS = ['/usr/share/applications',
                os.path.expanduser('~/.local/share/applications'),
                '/var/lib/snapd/desktop/applications']

# Process names worth reporting as "open" when a window cannot be enumerated.
# ‼️ Matched on the first 15 characters: that is all `ps -o comm=` reports, so
# "gnome-calculator" arrives as "gnome-calculato" and an exact compare silently
# never matches — which is why a visibly open calculator listed as nothing.
_GUI_PROCS = {'firefox': 'firefox', 'nautilus': 'files',
              'gnome-terminal-': 'terminal', 'gnome-text-editor': 'text_editor',
              'gnome-calculator': 'calculator', 'evince': 'pdf_viewer',
              'totem': 'video_player', 'code': 'vscode'}
_COMM = {name[:15]: app for name, app in _GUI_PROCS.items()}


def desktop_file(entry):
    """Full path of an installed .desktop entry, or None."""
    for d in _SEARCH_DIRS:
        p = os.path.join(d, entry)
        if os.path.exists(p):
            return p
    return None


def installed_apps():
    """The subset of KNOWN_APPS actually present on this machine."""
    return sorted(name for name, entry in KNOWN_APPS.items()
                  if desktop_file(entry))


def session_env():
    """Env of the user's graphical session, or None if no session is running."""
    for pattern in ('gsd-media-keys', 'gsd-color', 'gnome-session-binary'):
        try:
            pids = subprocess.run(['pgrep', '-u', str(os.getuid()), '-f', pattern],
                                  capture_output=True, text=True, timeout=5)
        except Exception:
            continue
        for pid in pids.stdout.split():
            try:
                with open(f'/proc/{pid}/environ', 'rb') as f:
                    raw = f.read().decode('utf-8', 'replace')
            except OSError:
                continue
            env = dict(kv.split('=', 1) for kv in raw.split('\0')
                       if '=' in kv)
            if 'DBUS_SESSION_BUS_ADDRESS' in env:
                out = {k: env[k] for k in
                       ('DBUS_SESSION_BUS_ADDRESS', 'XDG_RUNTIME_DIR',
                        'DISPLAY', 'XAUTHORITY', 'WAYLAND_DISPLAY')
                       if k in env}
                out.setdefault('DISPLAY', ':0')
                if 'XAUTHORITY' not in out:
                    xauth = glob.glob('/run/user/*/.mutter-Xwaylandauth.*')
                    if xauth:
                        out['XAUTHORITY'] = xauth[0]
                return out
    return None


def list_windows():
    """Best-effort list of what is open. See the Wayland caveat above."""
    env = session_env()
    if env is None:
        return {'status': 'error', 'reason': 'no graphical session running'}

    windows, partial = [], True
    full_env = {**os.environ, **env}
    try:
        r = subprocess.run(['wmctrl', '-lx'], capture_output=True, text=True,
                           timeout=8, env=full_env)
        for line in r.stdout.splitlines():
            bits = line.split(None, 4)
            if len(bits) == 5:
                windows.append({'app': bits[2].split('.')[-1], 'title': bits[4]})
    except Exception:
        pass

    seen = {w['app'].lower() for w in windows}
    try:
        r = subprocess.run(['ps', '-u', str(os.getuid()), '-o', 'comm='],
                           capture_output=True, text=True, timeout=5)
        for comm in {c.strip() for c in r.stdout.splitlines()}:
            app = _COMM.get(comm[:15])
            if app and app not in seen:
                windows.append({'app': app, 'title': ''})
                seen.add(app)
    except Exception:
        pass

    return {'status': 'ok', 'windows': windows, 'count': len(windows),
            'partial': partial}


def exec_command(path):
    """The Exec= line of a .desktop entry, minus the %f/%U field codes."""
    for line in open(path, encoding='utf-8'):
        if line.startswith('Exec='):
            return [w for w in line[5:].strip().split()
                    if not (len(w) == 2 and w[0] == '%')]
    return None


def open_app(name, settle=2.0):
    """Launch one of installed_apps() in the user's session.

    ‼️ Not `gio launch`: for a DBusActivatable app it hands off to the session
    bus, returns 0, and nothing starts — measured 2026-07-30 with the calculator
    (rc=0, no process). Reporting that as success would be the assistant
    claiming an action it never performed, so the Exec line is run directly and
    the process is checked to still be alive before this returns 'started'.
    """
    entry = KNOWN_APPS.get(name)
    if entry is None:
        return {'status': 'error', 'reason': f'unknown app: {name}',
                'available': installed_apps()}
    path = desktop_file(entry)
    if path is None:
        return {'status': 'error', 'reason': f'{name} is not installed'}
    cmd = exec_command(path)
    if not cmd:
        return {'status': 'error', 'reason': f'no Exec line in {entry}'}
    env = session_env()
    if env is None:
        return {'status': 'error', 'reason': 'no graphical session running'}
    try:
        proc = subprocess.Popen(cmd, env={**os.environ, **env},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
    except Exception as e:
        return {'status': 'error', 'reason': str(e)}

    time.sleep(settle)
    if proc.poll() is not None:      # died on the spot — say so, do not claim it
        try:
            proc.stdout.close()
        except Exception:
            pass
        return {'status': 'error', 'action': 'open_app', 'app': name,
                'reason': f'exited immediately (rc={proc.returncode})'}
    return {'status': 'started', 'action': 'open_app', 'app': name}
