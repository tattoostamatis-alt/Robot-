"""Echolocation — what the room sounds like when you shout at it.

The robot has a speaker and a microphone array with acoustic echo cancellation,
which exists precisely so it can hear while it is talking. That combination is
unusual, and it makes an active acoustic measurement possible: emit a swept
chirp, record what comes back, and read the room out of the reverberation.

What survives the analysis, in decreasing order of how much I trust it:

  * **Reverberation time** — how long the energy takes to decay. A bare
    corridor rings; a room full of sofas does not. This is the robust one.
  * **Change** — comparing today's decay curve at the same spot to a stored
    one. A door that opened, a room that emptied. Comparisons are far more
    reliable than absolutes, because the speaker, the mic and the robot's own
    body all cancel out.
  * **Room size**, from the first strong reflection. The honest caveat is that
    it measures the distance to the nearest large flat surface, which is often
    a wall and sometimes a wardrobe.

Pure, so the signal processing is testable on synthetic audio.

‼️ None of this is calibrated in absolute terms. The numbers are for comparing
one measurement against another at the same spot, which is what the change
detection actually uses.
"""

import math

__all__ = ['chirp', 'energy_envelope', 'rt60_estimate', 'first_reflection',
           'EchoSignature', 'compare']

SAMPLE_RATE = 16000


def chirp(seconds=0.06, f0=800.0, f1=6000.0, rate=SAMPLE_RATE, amplitude=0.6):
    """A linear sweep, as float32 in [-1, 1].

    Short and mid-band on purpose: long enough to carry energy, short enough
    that the direct sound is over before the first reflection arrives from a
    couple of metres away, and inside the band where a small speaker and the
    array's beamformer both behave.
    """
    import numpy as np
    n = max(1, int(seconds * rate))
    t = np.arange(n) / float(rate)
    k = (f1 - f0) / max(seconds, 1e-6)
    sig = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    # Fade the ends, or the discontinuity rings louder than the room does.
    fade = max(1, n // 20)
    win = np.ones(n)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    return (amplitude * sig * win).astype(np.float32)


def energy_envelope(samples, rate=SAMPLE_RATE, window_ms=5.0):
    """RMS energy per short window, in dB relative to the loudest window."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return []
    step = max(1, int(rate * window_ms / 1000.0))
    frames = [x[i:i + step] for i in range(0, len(x) - step + 1, step)]
    if not frames:
        return []
    rms = np.array([math.sqrt(float(np.mean(f * f)) + 1e-12) for f in frames])
    peak = rms.max()
    # ‼️ Not `peak <= 0`. The epsilon above keeps the logarithm finite, so pure
    # silence comes out as a peak of 1e-6 and produced a perfectly valid-looking
    # envelope of zeros — an RT60 of "no decay" from a recording with nothing in
    # it. Anything this quiet is not a measurement.
    if peak < 1e-5:
        return []
    return (20.0 * np.log10(rms / peak)).tolist()


def rt60_estimate(envelope, window_ms=5.0, drop_db=20.0):
    """Seconds for the tail to fall `drop_db`, extrapolated to 60 dB.

    Measured from the PEAK onward. Returns None when the recording never falls
    that far — a room that has not decayed inside the window has not been
    measured, and extrapolating from a partial decay would invent a number.
    """
    if not envelope:
        return None
    peak_i = max(range(len(envelope)), key=lambda i: envelope[i])
    for i in range(peak_i + 1, len(envelope)):
        if envelope[i] <= envelope[peak_i] - drop_db:
            secs = (i - peak_i) * window_ms / 1000.0
            return secs * (60.0 / drop_db)
    return None


def first_reflection(envelope, window_ms=5.0, min_gap_ms=8.0, rise_db=3.0,
                     max_gap_ms=60.0):
    """Time in seconds from the direct sound to the first distinct echo.

    Looks for a local rise after the direct sound has begun decaying. Returns
    None when nothing stands out — an open space, or a reflection buried in the
    decay. Distance is then time * 343 / 2.

    ‼️ The search is BOUNDED. Without a limit the first live run reported a
    reflection 190 ms out and called it a wall 32.6 metres away — in a flat.
    That was a wobble in the decay tail, not a surface. 60 ms is about 10 m
    there-and-back, past any wall in a home, so anything later is tail noise
    and the honest answer is None.
    """
    if not envelope:
        return None
    peak_i = max(range(len(envelope)), key=lambda i: envelope[i])
    skip = max(1, int(min_gap_ms / window_ms))
    limit = min(len(envelope) - 1, peak_i + int(max_gap_ms / window_ms))
    for i in range(peak_i + skip, limit):
        if (envelope[i] > envelope[i - 1] + rise_db
                and envelope[i] >= envelope[i + 1]):
            return (i - peak_i) * window_ms / 1000.0
    return None


def reflection_distance(seconds):
    """Metres to the reflecting surface. Sound goes there and back."""
    if seconds is None:
        return None
    return seconds * 343.0 / 2.0


class EchoSignature:
    """One measurement, reduced to a few numbers that can be compared."""

    __slots__ = ('rt60', 'reflection_s', 'envelope', 'room')

    def __init__(self, rt60=None, reflection_s=None, envelope=None, room=None):
        self.rt60 = rt60
        self.reflection_s = reflection_s
        # Truncated: the useful part is the decay, and storing whole recordings
        # per room would grow without bound.
        self.envelope = list(envelope or [])[:200]
        self.room = room

    @property
    def distance(self):
        return reflection_distance(self.reflection_s)

    def to_dict(self):
        return {'rt60': self.rt60, 'reflection_s': self.reflection_s,
                'envelope': [round(v, 1) for v in self.envelope],
                'room': self.room}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(d.get('rt60'), d.get('reflection_s'),
                   d.get('envelope'), d.get('room'))


def compare(now, before, rt60_tol=0.12, curve_tol=3.5):
    """Has the room changed? Returns (changed, reason).

    ‼️ Comparison is the trustworthy half of this. The speaker, the microphone
    and the robot's own body all appear identically in both measurements and
    cancel; an absolute number carries all of them.
    """
    if before is None or now is None:
        return False, 'χωρίς σύγκριση'

    if now.rt60 is not None and before.rt60 is not None:
        d = abs(now.rt60 - before.rt60)
        if d > rt60_tol:
            longer = now.rt60 > before.rt60
            return True, ('ο χώρος αντηχεί περισσότερο' if longer
                          else 'ο χώρος αντηχεί λιγότερο')

    a, b = now.envelope, before.envelope
    n = min(len(a), len(b))
    if n >= 10:
        diff = sum(abs(a[i] - b[i]) for i in range(n)) / n
        if diff > curve_tol:
            return True, 'άλλαξε η καμπύλη απόσβεσης'

    return False, 'ίδιο με πριν'
