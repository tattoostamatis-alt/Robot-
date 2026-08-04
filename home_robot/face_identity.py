"""Knowing WHO is in the room, not just that a face is.

face_detection_node already finds faces with YuNet and publishes five landmarks
per face. This module turns those into an identity: SFace embeds an aligned
112x112 crop into 128 numbers, and two crops of the same person land close
together under cosine similarity.

Pairing SFace with YuNet is not incidental — SFace was trained on crops aligned
with exactly these five points, so the landmarks the detector already publishes
are what makes the embedding meaningful. Feeding it a raw bounding-box crop
instead costs a large chunk of the accuracy.

Pure: the geometry and the matching are testable without a camera. The ONNX
session lives in the node.

Two decisions that matter more than the model:

  * **Never guess.** Below the similarity threshold the answer is 'unknown'.
    A robot that confidently calls your daughter by your son's name is worse
    than one that says nothing, and face embeddings are quite happy to rank
    a stranger 0.3 against everyone.
  * **Enrolment is explicit.** diarization auto-enrols and the result is a
    gallery full of Speaker_2, Speaker_7… useful to nobody. A face gallery
    fills up the same way, except the entries also look like people. So a name
    has to be asked for.
"""

import math

__all__ = ['REFERENCE_LANDMARKS', 'cosine_similarity', 'FaceGallery',
           'PresenceTracker']

# The five-point template SFace/ArcFace crops are aligned to, for a 112x112
# output: left eye, right eye, nose, left mouth corner, right mouth corner.
REFERENCE_LANDMARKS = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)

# OpenCV Zoo's documented cosine threshold for SFace. Above this, same person.
_MATCH_THRESHOLD = 0.363


def cosine_similarity(a, b):
    """Cosine similarity of two embeddings, in [-1, 1]. 0 for a zero vector."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


class FaceGallery:
    """Enrolled people and their embeddings.

    Several embeddings per person on purpose: one photo of a face is one angle
    under one light, and the same person at a different distance can fall below
    threshold against it. Matching takes the BEST of a person's embeddings.
    """

    def __init__(self, threshold=_MATCH_THRESHOLD, max_per_person=12):
        self.threshold = threshold
        self.max_per_person = max_per_person
        self._people = {}          # name -> [embedding, ...]

    def __len__(self):
        return len(self._people)

    def names(self):
        return sorted(self._people)

    def enrol(self, name, embedding):
        """Add one sighting of a named person. Returns how many they now have.

        Keeps the most recent `max_per_person`: a gallery that grows without
        bound gets slower and, worse, keeps embeddings from a haircut ago
        forever.
        """
        name = (name or '').strip()
        if not name or not embedding:
            return 0
        vec = list(embedding)
        entries = self._people.setdefault(name, [])
        entries.append(vec)
        del entries[:-self.max_per_person]
        return len(entries)

    def forget(self, name):
        return self._people.pop(name, None) is not None

    def match(self, embedding):
        """(name, score) for the closest enrolled person, or (None, best_score).

        None means "not confident enough", NOT "no faces enrolled" — the caller
        must render it as unknown rather than picking the argmax anyway.
        """
        if not embedding or not self._people:
            return None, 0.0
        best_name, best_score = None, -1.0
        for name, vectors in self._people.items():
            for vec in vectors:
                score = cosine_similarity(embedding, vec)
                if score > best_score:
                    best_name, best_score = name, score
        if best_score < self.threshold:
            return None, best_score
        return best_name, best_score

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self):
        return {'threshold': self.threshold,
                'max_per_person': self.max_per_person,
                'people': {n: [list(v) for v in vs]
                           for n, vs in self._people.items()}}

    @classmethod
    def from_dict(cls, data):
        g = cls(threshold=data.get('threshold', _MATCH_THRESHOLD),
                max_per_person=data.get('max_per_person', 12))
        for name, vectors in (data.get('people') or {}).items():
            for vec in vectors:
                g.enrol(name, vec)
        return g


class PresenceTracker:
    """Turns per-frame matches into "who is in the room".

    A single frame is noisy: a profile view, a blink, somebody walking past.
    A person counts as present after `confirm` agreeing frames and stays
    present until `forget_after` seconds without a sighting, so looking away
    does not make them vanish and reappear.
    """

    def __init__(self, confirm=3, forget_after=25.0):
        self.confirm = confirm
        self.forget_after = forget_after
        self._hits = {}            # name -> consecutive sightings
        self._last_seen = {}       # name -> ts
        self._present = set()

    def update(self, names, now):
        """Feed the names recognised in one frame. Returns newly ARRIVED names.

        Only arrivals are returned, because that is the event worth acting on —
        greeting somebody, or writing a `saw` entry on the timeline. Continuing
        to be in the room is not news.
        """
        arrived = []
        seen = {n for n in names if n}
        for name in seen:
            self._last_seen[name] = now
            self._hits[name] = self._hits.get(name, 0) + 1
            if self._hits[name] >= self.confirm and name not in self._present:
                self._present.add(name)
                arrived.append(name)
        for name in list(self._hits):
            if name not in seen:
                # Decay rather than reset: one missed frame in a run of good
                # ones should not restart the count from zero.
                self._hits[name] = max(0, self._hits[name] - 1)
        self._expire(now)
        return arrived

    def _expire(self, now):
        for name in list(self._present):
            if now - self._last_seen.get(name, 0.0) > self.forget_after:
                self._present.discard(name)
                self._hits.pop(name, None)

    def present(self, now=None):
        if now is not None:
            self._expire(now)
        return sorted(self._present)

    def last_seen(self, name):
        return self._last_seen.get(name)

    def reset(self):
        self._hits.clear()
        self._last_seen.clear()
        self._present.clear()
