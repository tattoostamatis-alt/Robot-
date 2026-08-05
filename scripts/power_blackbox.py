#!/usr/bin/env python3
"""power-blackbox — record the APU's power draw so the next cut has evidence.

Installed to /usr/local/sbin/power-blackbox and run by power-blackbox.service.

WHY THIS EXISTS
---------------
The mini PC has been cut off 12 times. Every post-mortem so far has been an
argument from absence: the journal stops mid-sentence, no thermal entry, no
MCE, therefore "power". That tells us it was power but never tells us WHAT the
power was doing in the last second. With the machine on an Anker SOLIX C300 DC
(USB-C PD, no inverter) the live hypothesis is that a sudden STEP in draw trips
the port or forces a PD renegotiation. That hypothesis is testable only if
something is writing the draw down as it happens.

home_robot already has system_metrics_node.py, which is fsync-durable and
covers CPU/mem/thermals — but it is a ROS node, so it only runs when the robot
stack is up, and at least one crash happened with the machine idle. This is the
same idea as a plain systemd service that is always on, sampling the one number
that matters: PPT, the APU's own package power in watts.

DURABILITY
----------
An abrupt cut loses whatever is still in the page cache, so the log is flushed
and fsync'd every FSYNC_EVERY samples. At 2 Hz that bounds the loss at ~2 s —
enough to keep the step that killed it. fsync is deliberately NOT every sample:
this NVMe is already taking a beating from the cuts themselves.

  power-blackbox              record (what the service runs)
  power-blackbox --last [n]   show the last n seconds of the previous boot
  power-blackbox --analyse    biggest steps seen, and the tail before each gap
"""
import argparse
import csv
import glob
import os
import sys
import time

# Overridable so the thing can be exercised without root, and so a test never
# writes over the real evidence.
LOG = os.environ.get('POWER_BLACKBOX_LOG', '/var/log/power-blackbox.csv')
HZ = 2.0
FSYNC_EVERY = 4          # ~2 s of loss at 2 Hz
MAX_ROWS = 400_000       # ~55 h at 2 Hz; halved in place when reached
FIELDS = ['t', 'ppt_w', 'load1', 'mhz_sum', 'cpu_c']


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _find(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


class Sampler:
    """Opens every sysfs node once. Re-opening 16 cpufreq files at 2 Hz is
    itself a measurable load, which would be a silly thing for a power logger
    to add."""

    def __init__(self):
        self.ppt = _find('/sys/class/drm/card*/device/hwmon/hwmon*/power1_average')
        if not self.ppt:
            sys.exit('no PPT sensor (amdgpu power1_average) — nothing to record')
        self.freqs = sorted(glob.glob(
            '/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq'))
        self.temp = None
        for zone in glob.glob('/sys/class/hwmon/hwmon*'):
            try:
                with open(os.path.join(zone, 'name')) as f:
                    if f.read().strip() == 'k10temp':
                        self.temp = os.path.join(zone, 'temp1_input')
                        break
            except OSError:
                pass

    def sample(self):
        ppt = _read_int(self.ppt)
        mhz = sum(filter(None, (_read_int(p) for p in self.freqs))) // 1000
        try:
            load1 = float(open('/proc/loadavg').read().split()[0])
        except (OSError, ValueError):
            load1 = -1.0
        temp = _read_int(self.temp) if self.temp else None
        return {
            't': round(time.time(), 2),
            'ppt_w': round((ppt or 0) / 1e6, 2),
            'load1': load1,
            'mhz_sum': mhz,
            'cpu_c': round(temp / 1000.0, 1) if temp else '',
        }


def _trim(path):
    """Halve the file in place rather than rotating. A rotation leaves a window
    where the log the crash needs is the one being renamed."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return
    if len(lines) <= MAX_ROWS:
        return
    header, body = lines[0], lines[1 + len(lines) // 2:]
    tmp = path + '.trim'
    with open(tmp, 'w') as f:
        f.writelines([header] + body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def record():
    fresh = not os.path.exists(LOG) or os.path.getsize(LOG) == 0
    _trim(LOG)
    period = 1.0 / HZ
    sampler = Sampler()
    with open(LOG, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        # An explicit marker, not a guess. Inferring "the machine restarted"
        # from a gap in timestamps needs a threshold, and any threshold is
        # wrong: a fast reboot leaves a gap shorter than a stalled logger does.
        # A restart is something we KNOW here, so write it down.
        fh.write(f'#boot,{time.time():.2f},,,\n')
        n = 0
        while True:
            w.writerow(sampler.sample())
            n += 1
            if n % FSYNC_EVERY == 0:
                fh.flush()
                os.fsync(fh.fileno())
            time.sleep(period)


def _rows():
    """Yields (t, ppt_w, load1, mhz_sum) samples, and None at each boot
    marker."""
    with open(LOG) as f:
        for row in csv.DictReader(line for line in f if line.strip()):
            if (row.get('t') or '').startswith('#boot'):
                yield None
                continue
            try:
                yield (float(row['t']), float(row['ppt_w']),
                       float(row['load1']), int(row['mhz_sum'] or 0))
            except (ValueError, TypeError):
                continue


def _boots():
    """The log split into runs, one per boot, oldest first."""
    runs, cur = [], []
    for r in _rows():
        if r is None:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(r)
    if cur:
        runs.append(cur)
    return runs


def show_last(seconds):
    runs = _boots()
    if not runs:
        sys.exit('log is empty')
    # The run before the current one — i.e. the boot that got cut off. With
    # only one run recorded, show that one rather than refusing.
    prev = runs[-2] if len(runs) > 1 else runs[-1]
    if len(runs) == 1:
        print('(only one boot recorded — showing the current one)')
    end = prev[-1][0]
    tail = [r for r in prev if r[0] >= end - seconds]
    print(f'last {seconds:.0f}s before the log stopped '
          f'({time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))}):')
    print(f'{"time":<10} {"PPT W":>7} {"load":>6} {"GHz sum":>8} {"step W":>8}')
    for a, b in zip([tail[0]] + tail, tail):
        print(f'{time.strftime("%H:%M:%S", time.localtime(b[0])):<10} '
              f'{b[1]:>7.2f} {b[2]:>6.2f} {b[3] / 1000:>8.2f} {b[1] - a[1]:>+8.2f}')


def analyse():
    runs = _boots()
    rows = [r for run in runs for r in run]
    if len(rows) < 2:
        sys.exit('not enough samples')
    # Steps are only meaningful WITHIN a run: the jump across a boot boundary
    # is the machine having been off, not a load step.
    pairs = [(a, b) for run in runs for a, b in zip(run, run[1:])
             if b[0] - a[0] < 2.0]
    # Rank by magnitude but print the signed step: a 6 W collapse and a 6 W
    # surge are equally interesting and must not be shown as the same thing.
    steps = sorted(((abs(b[1] - a[1]), b[0], a[1], b[1]) for a, b in pairs),
                   reverse=True)
    print(f'{len(rows)} samples over {len(runs)} boot(s), '
          f'{(rows[-1][0] - rows[0][0]) / 3600:.1f} h of wall clock')
    print(f'PPT: mean {sum(r[1] for r in rows) / len(rows):.2f} W  '
          f'max {max(r[1] for r in rows):.2f} W')
    print('\nbiggest steps between consecutive samples:')
    for _, t, a, b in steps[:10]:
        print(f'  {time.strftime("%m-%d %H:%M:%S", time.localtime(t))}  '
              f'{a:6.2f} -> {b:6.2f} W   ({b - a:+.2f})')
    print(f'\n{max(0, len(runs) - 1)} interruption(s) — each is a reboot or a cut:')
    for prev, nxt in list(zip(runs, runs[1:]))[-10:]:
        dark = nxt[0][0] - prev[-1][0]
        print(f'  stopped {time.strftime("%m-%d %H:%M:%S", time.localtime(prev[-1][0]))}, '
              f'dark for {dark / 60:.1f} min, last draw {prev[-1][1]:.2f} W')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--last', nargs='?', type=float, const=30, default=None,
                    metavar='SEC', help='tail of the previous boot')
    ap.add_argument('--analyse', action='store_true', help='steps and gaps')
    args = ap.parse_args()
    if args.last is not None:
        show_last(args.last)
    elif args.analyse:
        analyse()
    else:
        record()
