"""Translations for the web dashboard's interface (Greek, English, German).

Greek is the source language: it is what is written in the HTML template and in
the JS, and it is what the browser shows when the language is 'el'. The other
two are looked up by the Greek string itself, so a string that has not been
translated still renders — as Greek — instead of showing a missing key.

‼️ This is the INTERFACE only. The robot keeps listening for "Έι ρομπότ" and
keeps answering in Greek: the wake word is a trained model (hey_robot.onnx)
built from Greek recordings, and switching that needs new recordings and a
retrain, not a dictionary. See [[project_robot_wakeword_rate]].

tests/test_dashboard_i18n.py extracts every Greek string out of the dashboard
and fails if one of them is missing from here, so this file cannot silently
fall behind the UI.
"""

# Code -> what to call it in its own language (never translated).
LANGUAGES = [('el', 'Ελληνικά'), ('en', 'English'), ('de', 'Deutsch')]

TRANSLATIONS = {
    # ── tabs ──────────────────────────────────────────────────────────────
    'Χάρτης':            ('Map', 'Karte'),
    'Κάμερα':            ('Camera', 'Kamera'),
    'Βραχίονας':         ('Arm', 'Arm'),
    'Σκούπα':            ('Vacuum', 'Sauger'),
    'Φωνή/LLM':          ('Voice/LLM', 'Sprache/LLM'),
    'Σύστημα':           ('System', 'System'),
    'Ρυθμίσεις':         ('Settings', 'Einstellungen'),

    # ── map pane ──────────────────────────────────────────────────────────
    'ΧΑΡΤΗΣ · κλικ για πλοήγηση': ('MAP · click to navigate',
                                   'KARTE · zum Navigieren klicken'),
    'Δωμάτια':           ('Rooms', 'Räume'),
    'Γωνία':             ('Angle', 'Winkel'),
    'Ταχύτητα':          ('Speed', 'Geschwindigkeit'),
    '🔍 Εντοπισμός':     ('🔍 Localize', '🔍 Lokalisieren'),
    '✕ Ακύρωση στόχου':  ('✕ Cancel goal', '✕ Ziel abbrechen'),

    # ── camera pane ───────────────────────────────────────────────────────
    'Τι βλέπει':         ('What it sees', 'Was es sieht'),
    'Κατάσταση περιβάλλοντος': ('Situation', 'Umgebungslage'),

    # ── arm pane ──────────────────────────────────────────────────────────
    'Αρθρώσεις':         ('Joints', 'Gelenke'),
    'Δαγκάνα':           ('Gripper', 'Greifer'),
    'άνοιγμα':           ('opening', 'Öffnung'),
    '✋ Άνοιγμα':         ('✋ Open', '✋ Öffnen'),
    '🤏 Κλείσιμο':       ('🤏 Close', '🤏 Schließen'),
    'Εντολές':           ('Commands', 'Befehle'),
    '🏠 Θέση ανάπαυσης': ('🏠 Rest pose', '🏠 Ruhestellung'),
    '💤 Χαλάρωση (T:210)': ('💤 Go limp (T:210)', '💤 Erschlaffen (T:210)'),
    '⚡ Επαναφορά ροπής': ('⚡ Restore torque', '⚡ Drehmoment zurück'),
    '🎯 Άνοιγμα MoveIt': ('🎯 Open MoveIt', '🎯 MoveIt öffnen'),
    '‼️ Στον ώμο το «πάνω» είναι η ΑΡΝΗΤΙΚΗ φορά. Τα όρια είναι τα μετρημένα '
    'στο χέρι (31/07), όχι του κατασκευαστή. Η χαλάρωση κόβει τη ροπή — '
    'κράτα τον βραχίονα πριν την πατήσεις.':
        ('‼️ On the shoulder, "up" is the NEGATIVE direction. The limits are '
         'the ones measured by hand (31/07), not the manufacturer\'s. Going '
         'limp cuts the torque — hold the arm before pressing it.',
         '‼️ An der Schulter ist "oben" die NEGATIVE Richtung. Die Grenzwerte '
         'wurden von Hand gemessen (31.07.), nicht vom Hersteller. Erschlaffen '
         'schaltet das Drehmoment ab — halte den Arm fest, bevor du drückst.'),
    'Κόβεται η ροπή — ο βραχίονας θα πέσει. Τον κρατάς;':
        ('Torque is about to be cut — the arm will drop. Are you holding it?',
         'Das Drehmoment wird abgeschaltet — der Arm fällt. Hältst du ihn?'),

    # ── vacuum base pane ──────────────────────────────────────────────────
    'Βάση Roomba 879':   ('Roomba 879 base', 'Roomba-879-Basis'),
    'Σύνδεση':           ('Link', 'Verbindung'),
    'Προφυλακτήρας':     ('Bumper', 'Stoßfänger'),
    'Γκρεμός':           ('Cliff', 'Absturzkante'),
    'Τροχοί':            ('Wheels', 'Räder'),
    'Κινητήρες':         ('Motors', 'Motoren'),
    'Έκβαση':            ('Outcome', 'Ergebnis'),
    'Ενέργειες':         ('Actions', 'Aktionen'),
    '🔌 Στη βάση':       ('🔌 Go to dock', '🔌 Zur Ladestation'),
    '✕ Άκυρο docking':   ('✕ Cancel docking', '✕ Andocken abbrechen'),
    '▶ ΞΕΜΠΛΟΚΑΡΙΣΜΑ':   ('▶ RELEASE', '▶ FREIGEBEN'),
    'ΝΑΙ':               ('YES', 'JA'),
    'όχι':               ('no', 'nein'),
    'ΚΟΙΜΑΤΑΙ':          ('ASLEEP', 'SCHLÄFT'),
    'ζητά':              ('wants', 'verlangt'),
    '‼️ Δεν εμφανίζεται μπαταρία: το σασί τρέφεται από powerbank και τα πεδία '
    'φόρτισης του OI δίνουν σκουπίδια. Αν το «Σύνδεση» ξεπεράσει τα ~3s, η '
    'βάση κοιμάται — τότε κάθε πρόβλημα πλοήγησης είναι ψεύτικο.':
        ('‼️ No battery is shown: the chassis runs off a powerbank and the OI '
         'charging fields return garbage. If "Link" goes past ~3 s the base is '
         'asleep — and then every navigation problem you see is a phantom.',
         '‼️ Kein Akku wird angezeigt: das Chassis läuft über eine Powerbank '
         'und die Ladefelder des OI liefern Unsinn. Steht "Verbindung" über '
         '~3 s, schläft die Basis — dann ist jedes Navigationsproblem unecht.'),

    # ── voice pane ────────────────────────────────────────────────────────
    'Στείλε':            ('Send', 'Senden'),
    'Γράψε στο ρομπότ…':  ('Type to the robot…', 'Schreib dem Roboter…'),
    'Να το πει δυνατά χωρίς να το σκεφτεί':
        ('Say it out loud without thinking about it',
         'Laut aussprechen, ohne nachzudenken'),
    '— ξύπνησε (':       ('— woke up (', '— aufgeweckt ('),

    # ── system pane ───────────────────────────────────────────────────────
    'Υπολογιστής':       ('Computer', 'Rechner'),
    'Μνήμη':             ('Memory', 'Speicher'),
    'Θερμοκρασία':       ('Temperature', 'Temperatur'),
    'Κόμβοι ROS (':      ('ROS nodes (', 'ROS-Knoten ('),

    # ── 3D pane ───────────────────────────────────────────────────────────
    '3D κάμερα (D435)':  ('3D camera (D435)', '3D-Kamera (D435)'),
    'Επαναφορά όψης':    ('Reset view', 'Ansicht zurücksetzen'),
    'Σύρε για περιστροφή · ροδέλα ή τσίμπημα για ζουμ':
        ('Drag to rotate · wheel or pinch to zoom',
         'Ziehen zum Drehen · Rad oder Kneifen zum Zoomen'),
    'σημεία':            ('points', 'Punkte'),
    'αναμονή για νέφος σημείων…': ('waiting for a point cloud…',
                                   'warte auf eine Punktwolke…'),

    # ── log pane ──────────────────────────────────────────────────────────
    'Προειδοποιήσεις & σφάλματα (/rosout)':
        ('Warnings & errors (/rosout)', 'Warnungen & Fehler (/rosout)'),
    'Καθάρισε':          ('Clear', 'Leeren'),

    # ── settings: language ────────────────────────────────────────────────
    'Γλώσσα':            ('Language', 'Sprache'),
    'Αλλάζει μόνο αυτή τη σελίδα. Το ρομπότ συνεχίζει να μιλά ελληνικά.':
        ('Changes this page only. The robot keeps speaking Greek.',
         'Ändert nur diese Seite. Der Roboter spricht weiterhin Griechisch.'),

    # ── settings: maps ────────────────────────────────────────────────────
    'Χάρτες':            ('Maps', 'Karten'),
    '🆕 Νέος χάρτης (SLAM)': ('🆕 New map (SLAM)', '🆕 Neue Karte (SLAM)'),
    '💾 Αποθήκευση':     ('💾 Save', '💾 Speichern'),
    'όνομα χάρτη':       ('map name', 'Kartenname'),
    'Ενεργοποίηση':      ('Activate', 'Aktivieren'),
    'επεκτάσιμος':       ('extendable', 'erweiterbar'),
    'χαρτογράφηση…':     ('mapping…', 'kartiere…'),
    'Θα σταματήσει η πλοήγηση και θα ξαναξεκινήσουν όλα με τον χάρτη':
        ('Navigation will stop and everything will restart with the map',
         'Die Navigation stoppt und alles startet neu mit der Karte'),
    'Διαρκεί περίπου 90 δευτερόλεπτα. Να συνεχίσω;':
        ('It takes about 90 seconds. Continue?',
         'Es dauert etwa 90 Sekunden. Fortfahren?'),
    'Επανεκκίνηση… η σελίδα θα ξανασυνδεθεί μόνη της.':
        ('Restarting… the page will reconnect on its own.',
         'Neustart… die Seite verbindet sich von selbst wieder.'),
    'Ξεκινά ΝΕΑ χαρτογράφηση (SLAM). Ο τρέχων χάρτης δεν χάνεται, αλλά η '
    'πλοήγηση σταματά μέχρι να αποθηκεύσεις τον καινούργιο. Να συνεχίσω;':
        ('A NEW mapping run (SLAM) will start. The current map is not lost, '
         'but navigation stops until you save the new one. Continue?',
         'Eine NEUE Kartierung (SLAM) beginnt. Die aktuelle Karte geht nicht '
         'verloren, aber die Navigation stoppt, bis du die neue speicherst. '
         'Fortfahren?'),
    'Ξεκινά χαρτογράφηση… οδήγησε το ρομπότ σε όλο τον χώρο και μετά αποθήκευσε.':
        ('Mapping started… drive the robot around the whole space, then save.',
         'Kartierung gestartet… fahre den Roboter durch den ganzen Raum und '
         'speichere dann.'),
    'Δώσε όνομα με λατινικά γράμματα, αριθμούς, - ή _':
        ('Use a name with Latin letters, digits, - or _',
         'Verwende einen Namen aus lateinischen Buchstaben, Ziffern, - oder _'),
    'Αποθήκευση…':       ('Saving…', 'Speichere…'),
    'Αποθηκεύτηκε':      ('Saved', 'Gespeichert'),
    'Απέτυχε':           ('Failed', 'Fehlgeschlagen'),

    # ── GUI session tabs (RViz / MoveIt / Gazebo) ─────────────────────────
    'Η ίδια συνεδρία :2 που ανοίγει το <code>robot max</code> — και αυτή που '
    'βλέπεις από το RealVNC στο κινητό.':
        ('The same :2 session <code>robot max</code> opens — and the one you '
         'see from RealVNC on the phone.',
         'Dieselbe :2-Sitzung, die <code>robot max</code> öffnet — und die, '
         'die du per RealVNC am Handy siehst.'),
    '‼️ ΠΡΟΣΟΜΟΙΩΣΗ. Δημοσιεύει δικά της /clock, /scan, /odom — μην την '
    'ανοίγεις ενώ οδηγείς το πραγματικό ρομπότ.':
        ('‼️ SIMULATION. It publishes its own /clock, /scan, /odom — do not '
         'open it while driving the real robot.',
         '‼️ SIMULATION. Sie veröffentlicht eigene /clock, /scan, /odom — '
         'nicht öffnen, während du den echten Roboter fährst.'),
    'Ξεκινά <code>arm_moveit.launch.py</code>: move_group + RViz με το Motion '
    'Planning panel. Τράβα τη δαγκάνα, Plan, Execute.':
        ('Starts <code>arm_moveit.launch.py</code>: move_group + RViz with the '
         'Motion Planning panel. Drag the gripper, Plan, Execute.',
         'Startet <code>arm_moveit.launch.py</code>: move_group + RViz mit dem '
         'Motion-Planning-Panel. Greifer ziehen, Plan, Execute.'),
    'Εκκίνηση':          ('Start', 'Starten'),
    '↻ Επανασύνδεση':    ('↻ Reconnect', '↻ Neu verbinden'),
    '■ Τερματισμός':     ('■ Stop', '■ Beenden'),
    'ενεργό':            ('running', 'läuft'),
    'ξεκινά…':           ('starting…', 'startet…'),
    'σταματημένο':       ('stopped', 'gestoppt'),
    'Λείπει το noVNC — <code>sudo apt install novnc</code>':
        ('noVNC is missing — <code>sudo apt install novnc</code>',
         'noVNC fehlt — <code>sudo apt install novnc</code>'),
    'Δεν απαντά ο server.': ('The server is not answering.',
                             'Der Server antwortet nicht.'),
    'Το Gazebo θέλει ~75 δευτερόλεπτα σε software rendering…':
        ('Gazebo needs ~75 seconds under software rendering…',
         'Gazebo braucht ~75 Sekunden bei Software-Rendering…'),
    'Ξεκινά η γραφική συνεδρία…': ('Starting the graphical session…',
                                   'Grafiksitzung wird gestartet…'),
    'Απέτυχε η εκκίνηση.': ('Failed to start.', 'Start fehlgeschlagen.'),
    'Σταμάτησε.':        ('Stopped.', 'Gestoppt.'),
    'Η συνεδρία δεν τρέχει.': ('The session is not running.',
                               'Die Sitzung läuft nicht.'),
}


def as_js_table() -> dict:
    """{greek: {'en': ..., 'de': ...}} — the shape the page's t() expects."""
    return {src: {'en': en, 'de': de} for src, (en, de) in TRANSLATIONS.items()}
