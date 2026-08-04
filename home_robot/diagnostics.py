"""Naming this robot's known failure modes instead of rediscovering them.

Every gremlin below has cost real debugging time at least once, and most of
them are SILENT: the node stays alive, the topic stays advertised, the preflight
still prints a tick, and only the behaviour is wrong. That is what makes them
expensive — the symptom never points at the cause.

This module holds the signatures and, crucially, the fix. It is pure, so a
signature can be tested against a synthetic health snapshot rather than by
breaking a real robot.

The design rule: a symptom must map to ONE named cause with ONE next action.
A diagnostic that says "something is wrong with the camera" is worth nothing —
the operator already knew. "This is the RealSense-died-at-bringup case, restart
the camera node" is worth the whole file.
"""

__all__ = ['Finding', 'CHECKS', 'diagnose', 'Health']

# Severity ordering for presentation.
CRITICAL, WARNING, INFO = 'critical', 'warning', 'info'
_RANK = {CRITICAL: 0, WARNING: 1, INFO: 2}


class Health:
    """A snapshot of what the robot can observe about itself.

    Deliberately plain data: the node fills it from ROS, the tests fill it by
    hand, and `diagnose` cannot tell the difference.
    """

    __slots__ = ('topic_hz', 'nodes', 'log_counts', 'cpu_pct', 'ram_pct',
                 'swap_pct', 'load1', 'cores', 'uptime_s')

    def __init__(self, topic_hz=None, nodes=None, log_counts=None,
                 cpu_pct=0.0, ram_pct=0.0, swap_pct=0.0, load1=0.0,
                 cores=16, uptime_s=0.0):
        self.topic_hz = topic_hz or {}         # '/topic' -> Hz (0.0 = silent)
        self.nodes = set(nodes or ())          # node names, no leading slash
        self.log_counts = log_counts or {}     # substring -> times seen lately
        self.cpu_pct = cpu_pct
        self.ram_pct = ram_pct
        self.swap_pct = swap_pct
        self.load1 = load1
        self.cores = cores
        self.uptime_s = uptime_s

    def hz(self, topic):
        return self.topic_hz.get(topic, 0.0)

    def has(self, node):
        return node in self.nodes

    def logged(self, needle):
        return self.log_counts.get(needle, 0)


class Finding:
    """One named problem, with the action that resolves it."""

    __slots__ = ('key', 'severity', 'title', 'detail', 'fix')

    def __init__(self, key, severity, title, detail, fix):
        self.key = key
        self.severity = severity
        self.title = title
        self.detail = detail
        self.fix = fix

    def to_dict(self):
        return {'key': self.key, 'severity': self.severity,
                'title': self.title, 'detail': self.detail, 'fix': self.fix}

    def __repr__(self):                                  # pragma: no cover
        return f'<Finding {self.severity} {self.key}>'


# ── the checks ────────────────────────────────────────────────────────────────
# Each is (key, predicate(health) -> bool, severity, title, detail, fix).
# Predicates must be cheap and must never raise on a partial snapshot.

def _camera_dead(h):
    # The camera node is up and advertising, but no frames come out. Measured
    # at ~2.5% of bringups; the preflight still reports a tick.
    return h.has('camera') and h.hz('/camera/camera/color/image_raw') < 1.0


def _depth_not_aligned(h):
    # Colour flows, aligned depth does not. Every 3D-aware consumer then
    # silently returns nothing: the detector runs and publishes [] forever.
    return (h.hz('/camera/camera/color/image_raw') > 1.0
            and h.hz('/camera/camera/aligned_depth_to_color/image_raw') < 1.0)


def _base_asleep(h):
    return h.has('roomba_driver') and h.hz('/odom') < 1.0


def _imu_silent(h):
    return h.has('imu_node') and h.hz('/imu/data') < 1.0


def _lidar_silent(h):
    return h.hz('/scan') < 1.0


def _not_localized(h):
    # AMCL only publishes while the robot moves, so a silent /amcl_pose is
    # normal when parked. A missing map->base_link transform is not.
    return h.has('amcl') and h.hz('/tf') > 0.0 and not h.has('__map_tf__')


def _mic_dead(h):
    return h.has('wake_word_node') and h.hz('/mic/audio') < 1.0


def _stt_stuck(h):
    # The wake word fires but nothing is ever transcribed: the classic "the
    # robot has gone deaf" — stt_node holding _busy forever.
    return h.logged('Already transcribing') >= 2


def _swap_thrashing(h):
    # Swap being USED is fine and normal here; swap being actively paged is
    # what hurts. The node measures si/so and only sets this when non-zero.
    return h.swap_pct > 0.0


def _overloaded(h):
    return h.cores > 0 and h.load1 > h.cores * 1.5


def _tts_flaky(h):
    return h.logged('TTS attempt') >= 3


def _serial_lost(h):
    return h.logged('Errno 110') >= 1 or h.logged('serial') >= 3


CHECKS = (
    ('camera_dead', _camera_dead, CRITICAL,
     'Η κάμερα δεν στέλνει καρέ',
     'Ο κόμβος της κάμερας τρέχει και το topic υπάρχει, αλλά δεν βγαίνει '
     'εικόνα. Συμβαίνει σε ~2.5% των εκκινήσεων και το preflight δείχνει ✔ — '
     'το ρομπότ είναι ΤΥΦΛΟ ενώ όλα μοιάζουν εντάξει.',
     'Ξαναξεκίνα μόνο τον κόμβο της κάμερας· δεν χρειάζεται όλο το stack.'),

    ('depth_not_aligned', _depth_not_aligned, CRITICAL,
     'Λείπει το ευθυγραμμισμένο βάθος',
     'Το χρώμα ρέει αλλά το aligned depth όχι. ΚΑΘΕ ανιχνευτής που θέλει 3D '
     'επιστρέφει σιωπηλά άδεια λίστα — τρέχει κανονικά και δεν βρίσκει ΠΟΤΕ '
     'τίποτα, χωρίς κανένα σφάλμα.',
     'Ξεκίνα με align_depth.enable:=true.'),

    ('base_asleep', _base_asleep, CRITICAL,
     'Η βάση κοιμάται — δεν υπάρχει odometry',
     'Το Roomba γυρνά 0 bytes σε κάθε ερώτημα. Χωρίς odometry ΚΑΘΕ διάγνωση '
     'πλοήγησης είναι ψεύτικη: το AMCL δεν διορθώνει, τα goals αποτυγχάνουν.',
     'Power cycle ~10s από το power station. Το BRC είναι νεκρό σε αυτό το '
     'σκαφάκι — ο driver ΔΕΝ μπορεί να το ξυπνήσει με κώδικα.'),

    ('lidar_silent', _lidar_silent, CRITICAL,
     'Το LiDAR δεν στέλνει',
     'Χωρίς /scan δεν υπάρχει ούτε εντοπισμός ούτε αποφυγή εμποδίων.',
     'systemctl restart ros-sllidar-c1.service'),

    ('mic_dead', _mic_dead, CRITICAL,
     'Το μικρόφωνο δεν στέλνει',
     'Ο κόμβος wake word κατέχει το ALSA stream και το αναδημοσιεύει. Αν '
     'σιωπήσει, χάνονται ΜΑΖΙ wake word, STT και αναγνώριση ήχων.',
     'Ξαναξεκίνα το wake_word_node· έλεγξε και το USB hub.'),

    ('imu_silent', _imu_silent, WARNING,
     'Το IMU σιωπά',
     'Νεκρό BNO085 είναι ΣΙΩΠΗΛΟ. Πριν κατηγορήσεις το lidar ή το AMCL, '
     'κοίτα εδώ.',
     'Το nRESET στο GPIO18 αυτοθεραπεύει· αλλιώς αποσύνδεσε/σύνδεσε το USB.'),

    ('stt_stuck', _stt_stuck, CRITICAL,
     'Το STT κόλλησε — το ρομπότ είναι ΚΟΥΦΟ',
     'Το wake word ξυπνά αλλά τίποτα δεν μεταγράφεται: ο κόμβος κρατά το '
     '_busy για πάντα και κάθε επόμενο ξύπνημα απορρίπτεται σιωπηλά.',
     'Ξαναξεκίνα το stt_node.'),

    ('serial_lost', _serial_lost, WARNING,
     'Πρόβλημα σειριακής',
     'Το CH340 κολλάει σε pyserial ioctls μετά από reset του USB hub.',
     'Έλεγξε το USB hub· ο κώδικας χρησιμοποιεί raw termios ακριβώς γι αυτό.'),

    ('tts_flaky', _tts_flaky, WARNING,
     'Η φωνή αποτυγχάνει και ξαναπροσπαθεί',
     'Το edge-tts θέλει δίκτυο και αποτυγχάνει συχνά. «Max:» χωρίς '
     '«Speaking:» σημαίνει ότι φταίει η ΦΩΝΗ, όχι η ακοή.',
     'Έλεγξε το δίκτυο. Οι επαναλήψεις συνήθως πετυχαίνουν.'),

    ('swap_thrashing', _swap_thrashing, WARNING,
     'Το swap δουλεύει ενεργά',
     'Σελίδες πηγαινοέρχονται στον δίσκο — αυτό ΟΝΤΩΣ κοστίζει, σε αντίθεση '
     'με το απλώς δεσμευμένο swap.',
     'Κλείσε ό,τι δεν βλέπει κανείς (RViz του VNC, foxglove) ή χρησιμοποίησε '
     'μικρότερο μοντέλο STT.'),

    ('overloaded', _overloaded, INFO,
     'Υψηλό φορτίο',
     'Το load ξεπερνά κατά πολύ τους πυρήνες.',
     'Δες ποιος τρώει CPU χωρίς κανέναν συνδεδεμένο.'),
)


def diagnose(health):
    """Every check that fires, most severe first.

    An empty list means nothing KNOWN is wrong — not that the robot is healthy.
    The distinction matters: this catalogue only recognises what has already
    gone wrong once.
    """
    out = []
    for key, predicate, severity, title, detail, fix in CHECKS:
        try:
            fired = bool(predicate(health))
        except Exception:                                 # noqa: BLE001
            # A partial snapshot must never take the diagnostics down — that
            # would lose the other checks too.
            fired = False
        if fired:
            out.append(Finding(key, severity, title, detail, fix))
    out.sort(key=lambda f: _RANK.get(f.severity, 9))
    return out
