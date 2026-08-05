"""Where in the house the odometry loses ground — the ROS-free core behind the
slip map in slip_monitor_node.py.

The Sensor fusion tab shows HOW MUCH AMCL is correcting, live. That answers
"is the odometry drifting" but not the question you actually want answered,
which is WHERE: a robot that loses 20 cm every single time it crosses one
particular rug has a fixable problem, and one that loses 2 cm everywhere has a
calibration problem. Those look identical on a live readout.

So: every time the map->odom transform jumps, the robot's position in the map
at that moment is recorded, accumulated into coarse grid cells, and kept
across restarts. After a few days the rug is a bright spot on the map.

Coarse cells (0.5 m by default) on purpose. The correction lands wherever AMCL
happened to run its update, not exactly on the offending floor, so pretending
to centimetre precision would spread one rug over a hundred distinct points and
show nothing.

Kept dependency-free so the accumulation and the file format can be tested
without a robot — see tests/test_slip_map.py.
"""

import json
import math
import os
import tempfile
import time

# Corrections smaller than this are AMCL's normal per-update jitter and would
# bury the real events under thousands of entries.
DEFAULT_MIN_SHIFT_M = 0.03
DEFAULT_MIN_TURN_RAD = math.radians(2.0)
DEFAULT_CELL_M = 0.5
# Cells that stop being corrected fade rather than staying lit for ever: the
# furniture moves, the rug gets taken up, and a map that only ever accumulates
# would keep pointing at a problem that was solved months ago.
DEFAULT_HALF_LIFE_DAYS = 14.0


class SlipMap:
    """Accumulated correction per grid cell, with a file behind it."""

    def __init__(self, path=None, map_name=None, cell=DEFAULT_CELL_M,
                 half_life_days=DEFAULT_HALF_LIFE_DAYS):
        self.path = path
        self.map_name = map_name
        self.cell = cell
        self.half_life_days = half_life_days
        # (ix, iy) -> {'m': total metres corrected, 'n': events, 't': last time}
        self.cells = {}

    # ── accumulation ───────────────────────────────────────────────────────
    def key(self, x, y):
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def add(self, x, y, shift_m, turn_rad, now=None):
        """Record one correction that happened while the robot was at (x, y)."""
        now = time.time() if now is None else now
        k = self.key(x, y)
        c = self.cells.setdefault(k, {'m': 0.0, 'rad': 0.0, 'n': 0, 't': now})
        c['m'] += abs(shift_m)
        c['rad'] += abs(turn_rad)
        c['n'] += 1
        c['t'] = now
        return k

    def decayed(self, now=None):
        """Cells with their totals aged, heaviest first.

        Returns dicts rather than the raw store so callers cannot accidentally
        write the decay back into the accumulator — the decay is a VIEW, and
        folding it in would make the totals depend on how often anyone looked.
        """
        now = time.time() if now is None else now
        half_life_s = self.half_life_days * 86400.0
        out = []
        for (ix, iy), c in self.cells.items():
            age = max(0.0, now - c['t'])
            weight = 0.5 ** (age / half_life_s) if half_life_s > 0 else 1.0
            out.append({
                'x': (ix + 0.5) * self.cell,
                'y': (iy + 0.5) * self.cell,
                'm': c['m'] * weight,
                'deg': math.degrees(c['rad']) * weight,
                'n': c['n'],
                'age_days': age / 86400.0,
            })
        out.sort(key=lambda d: -d['m'])
        return out

    def prune(self, now=None, floor=0.01):
        """Drop cells that have faded to nothing, so the file cannot grow for
        ever in a house that is being used normally."""
        keep = {}
        now = time.time() if now is None else now
        half_life_s = self.half_life_days * 86400.0
        for k, c in self.cells.items():
            age = max(0.0, now - c['t'])
            weight = 0.5 ** (age / half_life_s) if half_life_s > 0 else 1.0
            if c['m'] * weight >= floor:
                keep[k] = c
        dropped = len(self.cells) - len(keep)
        self.cells = keep
        return dropped

    # ── persistence ────────────────────────────────────────────────────────
    def to_dict(self):
        return {
            'map': self.map_name,
            'cell': self.cell,
            'cells': [{'ix': ix, 'iy': iy, **c} for (ix, iy), c in self.cells.items()],
        }

    def load(self):
        """Read the file back, unless it belongs to a different map.

        ‼️ A slip map from another map file is worse than none: the
        coordinates are meaningful only in the frame they were recorded in, so
        loading them would paint the previous house's rug onto this one's
        living room. Every remap starts clean.
        """
        if not self.path or not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return False
        if self.map_name and data.get('map') != self.map_name:
            self.cells = {}
            return False
        self.cell = data.get('cell', self.cell)
        self.cells = {(c['ix'], c['iy']): {'m': c.get('m', 0.0),
                                           'rad': c.get('rad', 0.0),
                                           'n': c.get('n', 0),
                                           't': c.get('t', 0.0)}
                      for c in data.get('cells', [])}
        return True

    def save(self):
        """Atomic: a half-written file read at the next boot would throw away
        weeks of accumulation for a power cut during a 200-byte write."""
        if not self.path:
            return False
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path),
                                   prefix='.slip_map.')
        try:
            with os.fdopen(fd, 'w') as fh:
                json.dump(self.to_dict(), fh)
            os.replace(tmp, self.path)
            return True
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False


def correction_delta(prev, cur):
    """(shift metres, turn radians) between two (x, y, yaw) map->odom samples."""
    if prev is None:
        return 0.0, 0.0
    dx, dy = cur[0] - prev[0], cur[1] - prev[1]
    dyaw = math.atan2(math.sin(cur[2] - prev[2]), math.cos(cur[2] - prev[2]))
    return math.hypot(dx, dy), abs(dyaw)


def is_significant(shift_m, turn_rad,
                   min_shift=DEFAULT_MIN_SHIFT_M, min_turn=DEFAULT_MIN_TURN_RAD):
    return shift_m >= min_shift or turn_rad >= min_turn
