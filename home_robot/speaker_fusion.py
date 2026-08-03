"""Who is talking — fusing voice identity, direction and faces.

Three signals already exist on this robot and none of them was combined with
any other:

  * `current_speaker` (diarization_node) — a NAME, from resemblyzer d-vectors.
    Knows *who*, knows nothing about *where*.
  * `/doa/angle` (doa_node) — a BEARING from the XVF3800 DSP. Knows where a
    voice came from, has no idea whose it is.
  * `face_detections` (face_detection_node) — boxes in the image. Knows where
    people are on screen, not who is speaking.

Separately each answers a third of "who is talking to me". Together they
answer it: the DoA bearing is converted into an image column and matched
against the visible faces, and whichever face it lands on is the one the
diarised name belongs to.

Pure functions, no ROS: geometry and bookkeeping only, so it can be tested
without a microphone, a camera or a person.

‼️ Coordinate conventions, both easy to get backwards:
  * XVF3800 DoA is 0° at the device front and increases CLOCKWISE (same
    convention doa_node documents and negates for ROS twists).
  * Image columns increase to the RIGHT. So a bearing to the robot's right is
    a LARGER column — but only after wrapping the angle into +/-180.
"""
import math

# Horizontal field of view of the D435 colour stream, degrees. A face outside
# this cannot be matched to a bearing no matter how loud it is.
CAMERA_HFOV_DEG = 69.0

# How far a face's centre may sit from the projected bearing, as a fraction of
# the image width, and still be considered the source. 0.18 of 640 px is ~115
# px — roughly a head's width at conversational distance, which is the real
# precision limit of a 4-mic array in a reverberant room.
MATCH_TOLERANCE = 0.18


def wrap180(deg):
    """Fold an angle into (-180, 180]. 350° is 10° to the LEFT, not far right."""
    d = (float(deg) + 180.0) % 360.0 - 180.0
    return d + 360.0 if d <= -180.0 else d


def bearing_to_column(angle_deg, img_w, hfov=CAMERA_HFOV_DEG):
    """Image column a DoA bearing points at, or None if outside the frame.

    None is a real answer and must not be treated as 'centre': someone
    speaking from behind the robot is not the face it can see in front.
    """
    if img_w <= 0:
        return None
    a = wrap180(angle_deg)
    half = hfov / 2.0
    if abs(a) > half:
        return None
    # Linear across the sensor. Exact would be tan-based, but the tolerance
    # below is far wider than the error that approximation introduces.
    return (0.5 + a / hfov) * img_w


def face_center(face):
    return ((face.get('x1', 0) + face.get('x2', 0)) / 2.0,
            (face.get('y1', 0) + face.get('y2', 0)) / 2.0)


def match_face(angle_deg, faces, img_w, *, hfov=CAMERA_HFOV_DEG,
               tolerance=MATCH_TOLERANCE):
    """The face a voice bearing points at, or None.

    None whenever the answer would be a guess: no faces, bearing outside the
    camera's view, nothing close enough, or TWO faces equally close — in a
    two-person conversation, picking the wrong one is worse than admitting the
    robot cannot tell.
    """
    if not faces or img_w <= 0:
        return None
    col = bearing_to_column(angle_deg, img_w, hfov)
    if col is None:
        return None

    limit = tolerance * img_w
    scored = sorted(
        ((abs(face_center(f)[0] - col), f) for f in faces),
        key=lambda t: t[0])
    best_d, best = scored[0]
    if best_d > limit:
        return None
    if len(scored) > 1:
        second_d = scored[1][0]
        # Ambiguous: two faces within a whisker of each other relative to how
        # coarse the bearing is.
        if (second_d - best_d) < (limit * 0.5):
            return None
    return best


class SpeakerTracker:
    """Bookkeeping over the fused signals.

    Holds the latest of each with its own age, because they arrive at wildly
    different rates (DoA ~10 Hz, faces a few Hz, a diarised name only when
    somebody finishes speaking) and a stale one must not be presented as
    current.
    """

    def __init__(self, name_ttl=12.0, angle_ttl=4.0, face_ttl=2.0, clock=None):
        import time as _t
        self._clock = clock or _t.monotonic
        self.name_ttl = float(name_ttl)
        self.angle_ttl = float(angle_ttl)
        self.face_ttl = float(face_ttl)
        self._name = None
        self._name_at = 0.0
        self._angle = None
        self._angle_at = 0.0
        self._faces = []
        self._faces_at = 0.0
        self._img_w = 0

    # ── ingest ──
    def set_name(self, name, now=None):
        name = (name or '').strip()
        # "unknown" is the diarizer saying it could not match, not a person.
        if not name or name.lower() == 'unknown':
            return
        self._name = name
        self._name_at = self._clock() if now is None else now

    def set_angle(self, angle_deg, now=None):
        self._angle = float(angle_deg)
        self._angle_at = self._clock() if now is None else now

    def set_faces(self, faces, img_w, now=None):
        self._faces = list(faces or [])
        self._img_w = int(img_w or 0)
        self._faces_at = self._clock() if now is None else now

    # ── query ──
    def _fresh(self, stamp, ttl, now):
        return bool(stamp) and (now - stamp) <= ttl

    def snapshot(self, now=None):
        """What the robot can honestly say about who is speaking."""
        now = self._clock() if now is None else now
        name = self._name if self._fresh(self._name_at, self.name_ttl, now) else None
        angle = self._angle if self._fresh(self._angle_at, self.angle_ttl, now) else None
        faces_fresh = self._fresh(self._faces_at, self.face_ttl, now)
        faces = self._faces if faces_fresh else []

        face = None
        if angle is not None and faces:
            face = match_face(angle, faces, self._img_w)

        return {
            'name': name,
            'angle': None if angle is None else round(wrap180(angle), 1),
            'faces_visible': len(faces),
            'face': face,
            # True only when a name and a face were tied together by direction —
            # i.e. the robot knows who is speaking AND which person that is.
            'identified': bool(name and face),
        }

    def describe(self, snap=None):
        """One short Greek line for the LLM's situational context, or None.

        None rather than "δεν ξέρω": this goes into every prompt, and a line
        saying nothing costs prefill on every single utterance.
        """
        s = self.snapshot() if snap is None else snap
        if not s['name'] and s['angle'] is None:
            return None
        bits = []
        if s['name']:
            bits.append(f"Μιλάει ο/η {s['name']}")
        if s['angle'] is not None:
            side = ('μπροστά' if abs(s['angle']) <= 20
                    else 'δεξιά' if s['angle'] > 0 else 'αριστερά')
            bits.append(f'από {side} ({s["angle"]:.0f}°)')
        if s['faces_visible']:
            bits.append(f'βλέπω {s["faces_visible"]} πρόσωπο'
                        + ('' if s['faces_visible'] == 1 else 'α'))
        return ', '.join(bits) + '.'
