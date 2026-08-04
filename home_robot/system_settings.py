"""Parsing the system tools the settings panel drives.

nmcli, bluetoothctl and pactl all print for humans, and the dashboard needs
structures. Keeping the parsing here means it can be tested against real
captured output instead of against a live radio, which is the only way to
exercise the cases that matter — a hidden SSID, an SSID containing a colon, a
Bluetooth device with no name.

Nothing here runs a command. The node does that; this only makes sense of what
came back, and builds the argument lists so no shell string is ever assembled
by hand.
"""

import re
import shlex

__all__ = ['parse_wifi_list', 'parse_devices', 'parse_bt_devices',
           'parse_volume', 'wifi_connect_args', 'bt_args', 'volume_args']


def _unescape(field):
    r"""nmcli -t escapes ':' and '\' inside fields as '\:' and '\\'."""
    out, esc = [], False
    for ch in field:
        if esc:
            out.append(ch)
            esc = False
        elif ch == '\\':
            esc = True
        else:
            out.append(ch)
    return ''.join(out)


def _split_terse(line):
    """Split an `nmcli -t` line on unescaped colons."""
    fields, cur, esc = [], [], False
    for ch in line:
        if esc:
            cur.append('\\')
            cur.append(ch)
            esc = False
        elif ch == '\\':
            esc = True
        elif ch == ':':
            fields.append(_unescape(''.join(cur)))
            cur = []
        else:
            cur.append(ch)
    if esc:
        cur.append('\\')
    fields.append(_unescape(''.join(cur)))
    return fields


def parse_wifi_list(output):
    """`nmcli -t -f IN-USE,SSID,SIGNAL,SECURITY device wifi list` -> networks.

    Sorted by signal, strongest first, one entry per SSID — the same network
    seen on two bands would otherwise appear twice and look like two networks.
    Hidden networks (empty SSID) are dropped: they cannot be joined by name and
    an unnamed row is only confusing.
    """
    seen = {}
    for line in (output or '').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = _split_terse(line)
        if len(parts) < 4:
            continue
        in_use, ssid, signal, security = parts[0], parts[1], parts[2], parts[3]
        if not ssid:
            continue
        try:
            sig = int(signal)
        except ValueError:
            sig = 0
        entry = {
            'ssid': ssid,
            'signal': sig,
            'secure': bool(security.strip()) and security.strip() != '--',
            'active': in_use.strip() in ('*', 'yes'),
        }
        prev = seen.get(ssid)
        if prev is None or sig > prev['signal'] or entry['active']:
            # Keep the strongest sighting, but never let it lose "active".
            entry['active'] = entry['active'] or (prev or {}).get('active', False)
            seen[ssid] = entry
    return sorted(seen.values(), key=lambda n: (-n['signal'], n['ssid']))


def parse_devices(output):
    """`nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device` -> interfaces."""
    out = []
    for line in (output or '').splitlines():
        parts = _split_terse(line.strip())
        if len(parts) < 3 or not parts[0]:
            continue
        out.append({'device': parts[0], 'type': parts[1],
                    'state': parts[2],
                    'connection': parts[3] if len(parts) > 3 else ''})
    return out


_BT_RE = re.compile(r'^Device\s+([0-9A-Fa-f:]{17})\s*(.*)$')


def parse_bt_devices(output, connected=()):
    """`bluetoothctl devices` -> [{mac, name, connected}].

    A device with no name shows its address, because an empty row cannot be
    told apart from another empty row.
    """
    conn = {m.upper() for m in connected}
    out = []
    for line in (output or '').splitlines():
        m = _BT_RE.match(line.strip())
        if not m:
            continue
        mac = m.group(1).upper()
        name = m.group(2).strip() or mac
        out.append({'mac': mac, 'name': name, 'connected': mac in conn})
    out.sort(key=lambda d: (not d['connected'], d['name'].lower()))
    return out


def parse_volume(output):
    """`pactl get-sink-volume @DEFAULT_SINK@` -> percent, or None.

    The line carries the same level twice (one per channel); the first is
    enough, and they are kept in step by pactl anyway.
    """
    m = re.search(r'(\d+)%', output or '')
    return int(m.group(1)) if m else None


# ── argument builders ─────────────────────────────────────────────────────────
# Lists, never strings. A password with a space or a quote in it would break a
# shell string, and an SSID is attacker-influenced text arriving over the
# network — it must never reach a shell.

def wifi_connect_args(ssid, password=None):
    if not ssid:
        return None
    args = ['nmcli', 'device', 'wifi', 'connect', ssid]
    if password:
        args += ['password', password]
    return args


def bt_args(action, mac=None):
    """bluetoothctl invocations for the actions the panel exposes."""
    simple = {'on': ['power', 'on'], 'off': ['power', 'off'],
              'scan': ['--timeout', '10', 'scan', 'on'],
              'devices': ['devices'], 'paired': ['devices', 'Paired']}
    if action in simple:
        return ['bluetoothctl'] + simple[action]
    if action in ('connect', 'disconnect', 'pair', 'trust', 'remove') and mac:
        return ['bluetoothctl', action, mac]
    return None


def volume_args(percent):
    """Clamped: a UI slider that can send 400% is a broken speaker away."""
    try:
        pct = int(percent)
    except (TypeError, ValueError):
        return None
    pct = max(0, min(150, pct))
    return ['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f'{pct}%']


def quote_for_display(args):
    """A copy-pasteable rendering of a command, for logs and the UI."""
    return ' '.join(shlex.quote(a) for a in (args or []))
