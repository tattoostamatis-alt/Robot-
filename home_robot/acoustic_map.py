"""Where sounds come from — a spatial memory built out of bearings.

The microphone array gives a direction, not a position. One bearing from one
place is a ray, not a point. But the robot also knows where it is on a map, and
it hears the same doorbell over and over from different spots, so the rays pile
up and the source falls out of the intersection.

That is the whole idea, and it needs nothing new on the robot: `/doa/angle`,
`/sound_events` and AMCL already exist. What was missing was remembering them
together.

Two ways to use a bearing, and both matter:

  * **Immediately**, by walking the ray until it leaves the room the robot is
    standing in — that names a room from a SINGLE observation, which matters
    because a docked robot never moves and would otherwise never triangulate
    anything.
  * **Over time**, by accumulating rays in a grid. Cells crossed by many rays
    from different poses are where the sound actually comes from.

Pure, so the geometry can be tested without a robot. Angle conventions here are
the same trap as the compass: the XVF3800 reports 0-359 degrees measured
counter-clockwise from the robot's FRONT, while map yaw is counter-clockwise
from the map's +x axis. They add; they do not subtract.
"""

import math

__all__ = ['bearing_to_world', 'ray_points', 'SoundSourceMap']


def bearing_to_world(robot_yaw, doa_deg):
    """World-frame angle of a DoA bearing, in radians.

    ‼️ The XVF3800's 0 is the robot's nose and grows counter-clockwise, which
    is the SAME sense as ROS yaw. So the two add. (The compass subtracts,
    because compass bearings grow clockwise — different convention, same easy
    mistake.)
    """
    return math.atan2(math.sin(robot_yaw + math.radians(doa_deg)),
                      math.cos(robot_yaw + math.radians(doa_deg)))


def ray_points(x, y, world_angle, max_range=8.0, step=0.10):
    """Points along a ray, starting just off the robot itself.

    The first 30 cm are skipped: the robot's own body and the microphone mount
    are there, and attributing a sound to the cell the robot is standing in
    would make every hotspot land on the dock.
    """
    dx, dy = math.cos(world_angle), math.sin(world_angle)
    out = []
    d = 0.3
    while d <= max_range:
        out.append((x + dx * d, y + dy * d))
        d += step
    return out


class SoundSourceMap:
    """Accumulated bearings, per kind of sound.

    Stored as a sparse grid of counts. A cell's weight falls with distance from
    the robot, because a bearing error of a few degrees is centimetres nearby
    and metres far away — without the decay, distant cells dominate purely by
    being crossed by more rays.
    """

    def __init__(self, cell=0.25, max_range=8.0, decay_per_metre=0.12):
        self.cell = cell
        self.max_range = max_range
        self.decay_per_metre = decay_per_metre
        self._grid = {}          # (kind, ix, iy) -> weight
        self._counts = {}        # kind -> observations

    def _key(self, kind, x, y):
        return (kind, int(math.floor(x / self.cell)),
                int(math.floor(y / self.cell)))

    def cell_centre(self, ix, iy):
        return ((ix + 0.5) * self.cell, (iy + 0.5) * self.cell)

    def observe(self, kind, robot_x, robot_y, robot_yaw, doa_deg,
                max_range=None):
        """Record one heard-from-here bearing. Returns the cells it touched."""
        if not kind or doa_deg is None:
            return []
        rng = self.max_range if max_range is None else max_range
        angle = bearing_to_world(robot_yaw, doa_deg)
        touched = []
        for px, py in ray_points(robot_x, robot_y, angle, rng, self.cell):
            d = math.hypot(px - robot_x, py - robot_y)
            w = math.exp(-self.decay_per_metre * d)
            k = self._key(kind, px, py)
            self._grid[k] = self._grid.get(k, 0.0) + w
            touched.append(k)
        self._counts[kind] = self._counts.get(kind, 0) + 1
        return touched

    def observations(self, kind):
        return self._counts.get(kind, 0)

    def kinds(self):
        return sorted(self._counts)

    def hotspots(self, kind, top=3, min_weight=1.0):
        """The most likely places this sound comes from, strongest first.

        ‼️ Meaningless until the robot has heard the sound from more than one
        place. A single ray makes every cell along it equally "likely", and
        returning its far end as a location would be an invention. Callers get
        an empty list until there is real evidence.
        """
        if self._counts.get(kind, 0) < 2:
            return []
        cells = [(w, ix, iy) for (k, ix, iy), w in self._grid.items()
                 if k == kind and w >= min_weight]
        if not cells:
            return []
        cells.sort(reverse=True)
        out = []
        for w, ix, iy in cells:
            cx, cy = self.cell_centre(ix, iy)
            # Suppress neighbours of an already-reported peak, or the answer is
            # one blob reported three times.
            if any(math.dist((cx, cy), (px, py)) < 4 * self.cell
                   for px, py, _ in out):
                continue
            out.append((cx, cy, round(w, 2)))
            if len(out) >= top:
                break
        return out

    def prune(self, floor=0.05):
        """Drop cells that never accumulated anything worth keeping."""
        self._grid = {k: w for k, w in self._grid.items() if w >= floor}

    def to_dict(self):
        return {'cell': self.cell, 'max_range': self.max_range,
                'decay_per_metre': self.decay_per_metre,
                'counts': self._counts,
                # Keys become strings for JSON and back again on load.
                'grid': {f'{k}|{ix}|{iy}': round(w, 3)
                         for (k, ix, iy), w in self._grid.items()}}

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        m = cls(cell=data.get('cell', 0.25),
                max_range=data.get('max_range', 8.0),
                decay_per_metre=data.get('decay_per_metre', 0.12))
        m._counts = {k: int(v) for k, v in (data.get('counts') or {}).items()}
        for key, w in (data.get('grid') or {}).items():
            parts = key.rsplit('|', 2)
            if len(parts) == 3:
                try:
                    m._grid[(parts[0], int(parts[1]), int(parts[2]))] = float(w)
                except ValueError:
                    continue
        return m
