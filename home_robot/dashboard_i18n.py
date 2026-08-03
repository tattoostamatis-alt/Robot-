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
    '⌨️ Και από πληκτρολόγιο: βελάκια ή WASD (κράτα πατημένο), space = στοπ. '
    'Αφήνοντας το πλήκτρο σταματά· αν χαθεί το tab ή το δίκτυο, η βάση σταματά '
    'μόνη της σε 0.25s.':
        ('⌨️ Keyboard too: arrows or WASD (hold them down), space = stop. '
         'Releasing the key stops the robot; if the tab or the network is lost, '
         'the base stops by itself within 0.25 s.',
         '⌨️ Auch per Tastatur: Pfeiltasten oder WASD (gedrückt halten), Leertaste '
         '= Stopp. Loslassen stoppt; geht der Tab oder das Netz verloren, stoppt '
         'die Basis nach 0,25 s von selbst.'),

    # ── costmap pane ──────────────────────────────────────────────────────
    # 'Costmap' and 'inflation' carry no Greek, so the extractor never sees them
    # and an entry here would be dead weight (same as 'IMU' above).
    'COSTMAP · 3×3m γύρω από το ρομπότ': ('COSTMAP · 3×3m around the robot',
                                          'COSTMAP · 3×3 m um den Roboter'),
    'Υπόμνημα':          ('Legend', 'Legende'),
    'θανάσιμο':          ('lethal', 'tödlich'),
    'ακουμπά':           ('inscribed', 'berührt'),
    'από ανίχνευση':     ('from detection', 'aus Erkennung'),
    'άγνωστο':           ('unknown', 'unbekannt'),
    'Αυτό είναι ό,τι ΒΛΕΠΕΙ ο planner, όχι ο χάρτης. Το μπλε γύρω από τους '
    'τοίχους είναι το inflation — αν δύο μπλε ζώνες ενωθούν σε μια πόρτα, το '
    'ρομπότ δεν περνά, ακόμη κι αν το άνοιγμα φαίνεται καθαρό στον χάρτη. Τα ροζ '
    'σημεία είναι εμπόδια που έβαλε η ΑΝΙΧΝΕΥΣΗ (άνθρωποι παίρνουν μεγαλύτερο '
    'περιθώριο), όχι το lidar.':
        ('This is what the planner SEES, not the map. The blue around walls is '
         'inflation — where two blue zones meet in a doorway the robot will not '
         'go through, however clear the gap looks on the map. The pink points '
         'are obstacles placed by DETECTION (people get a wider margin), not by '
         'the lidar.',
         'Das ist, was der Planer SIEHT, nicht die Karte. Das Blau um Wände ist '
         'die Inflation — wo sich zwei blaue Zonen in einer Tür treffen, fährt '
         'der Roboter nicht durch, so frei die Lücke auf der Karte auch '
         'aussieht. Die rosa Punkte sind Hindernisse aus der ERKENNUNG '
         '(Menschen bekommen mehr Abstand), nicht vom Lidar.'),

    # ── vision / detection ────────────────────────────────────────────────
    'Ανίχνευση':         ('Detection', 'Erkennung'),
    '👁 Πλαίσια/σκελετός': ('👁 Boxes/skeleton', '👁 Rahmen/Skelett'),
    '👣 Ακολούθησέ με':  ('👣 Follow me', '👣 Folge mir'),
    '■ Σταμάτα να ακολουθείς': ('■ Stop following', '■ Nicht mehr folgen'),
    'Τα πράσινα πλαίσια είναι άνθρωποι, τα πορτοκαλί αντικείμενα· ο κίτρινος '
    'σκελετός είναι 17 σημεία COCO. ‼️ Χρειάζονται τα perception nodes — ξεκίνα '
    'με «robot max use_perception:=true», αλλιώς η εικόνα μένει καθαρή και ο '
    'μετρητής στο 0. Το «Ακολούθησέ με» σταματά μόνο του μετά από 30 '
    'δευτερόλεπτα.':
        ('Green boxes are people, orange ones objects; the yellow skeleton is 17 '
         'COCO keypoints. ‼️ Needs the perception nodes — start with "robot max '
         'use_perception:=true", otherwise the picture stays clean and the '
         'counter sits at 0. "Follow me" stops by itself after 30 seconds.',
         'Grüne Rahmen sind Menschen, orange sind Objekte; das gelbe Skelett '
         'sind 17 COCO-Punkte. ‼️ Braucht die Perception-Nodes — starte mit '
         '„robot max use_perception:=true", sonst bleibt das Bild sauber und '
         'der Zähler auf 0. „Folge mir" stoppt nach 30 Sekunden von selbst.'),

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

    # ── IMU pane (BNO085) ─────────────────────────────────────────────────
    # 'IMU' and 'Quaternion w/x/y/z' are deliberately absent: they carry no
    # Greek, so the extractor never sees them and an entry here would be dead.
    'Προσανατολισμός':   ('Orientation', 'Ausrichtung'),
    'Yaw (στροφή)':      ('Yaw (turn)', 'Gierwinkel (Drehung)'),
    'Pitch (μύτη)':      ('Pitch (nose)', 'Nickwinkel'),
    'Roll (κλίση)':      ('Roll (tilt)', 'Rollwinkel'),
    'Ρυθμός στροφής':    ('Turn rate', 'Drehrate'),
    'Συχνότητα':         ('Rate', 'Frequenz'),
    'Ακατέργαστες τιμές BNO085': ('Raw BNO085 values', 'BNO085-Rohwerte'),
    'Γυροσκόπιο X/Y/Z':  ('Gyroscope X/Y/Z', 'Gyroskop X/Y/Z'),
    'Επιτάχυνση X/Y/Z':  ('Acceleration X/Y/Z', 'Beschleunigung X/Y/Z'),
    'Θέση στο ρομπότ':   ('Mounting', 'Einbau am Roboter'),
    'χωρίς σήμα':        ('no signal', 'kein Signal'),
    'ΝΕΚΡΟ':             ('DEAD', 'TOT'),
    'ΓΥΡΟΣΚΟΠΙΟ ΝΕΚΡΟ':  ('GYRO DEAD', 'GYROSKOP TOT'),
    'εντάξει':           ('ok', 'ok'),
    'σταθερά 0 ενώ στρίβει': ('stuck at 0 while turning',
                              'konstant 0 während der Drehung'),
    'ακίνητο':           ('at rest', 'in Ruhe'),
    'δεν στέλνεται (ανενεργό report)':
        ('not streamed (report disabled)', 'wird nicht gesendet (Report aus)'),
    '‼️ ΣΧΕΤΙΚΗ πυξίδα, όχι Βορράς. Το firmware στέλνει GAME_ROTATION_VECTOR — '
    'σύντηξη γυροσκοπίου και επιταχυνσιομέτρου χωρίς το μαγνητόμετρο, επίτηδες: '
    'μέσα στο σπίτι οι κινητήρες DC της Roomba και τα μέταλλα διέλυαν την απόλυτη '
    'γωνία, ο EKF γύριζε και το AMCL δεν κρατούσε σύγκλιση. Το 0° είναι τυχαία '
    'κατεύθυνση σε κάθε boot. Για πλοήγηση δεν χρειάζεται αληθινός Βορράς — μόνο '
    'σταθερή σχετική γωνία, και το AMCL διορθώνει τη μικρή απόκλιση με scan matching.':
        ('‼️ RELATIVE compass, not North. The firmware sends GAME_ROTATION_VECTOR '
         '— gyro and accelerometer fusion without the magnetometer, deliberately: '
         'indoors the Roomba\'s DC motors and nearby metal wrecked the absolute '
         'heading, the EKF rotated and AMCL could not stay converged. 0° is an '
         'arbitrary direction on every boot. Navigation does not need true north '
         '— only a stable relative heading, and AMCL corrects the slow drift by '
         'scan matching.',
         '‼️ RELATIVER Kompass, nicht Norden. Die Firmware sendet '
         'GAME_ROTATION_VECTOR — Fusion aus Gyroskop und Beschleunigungsmesser '
         'ohne Magnetometer, mit Absicht: drinnen zerstörten die DC-Motoren des '
         'Roomba und Metall die absolute Richtung, der EKF drehte sich und AMCL '
         'blieb nicht konvergent. 0° ist bei jedem Start eine zufällige Richtung. '
         'Die Navigation braucht keinen echten Norden — nur eine stabile relative '
         'Richtung, und AMCL korrigiert die Drift per Scan-Matching.'),
    'Το BNO085 μπορεί να δώσει και μαγνητόμετρο, γραμμική επιτάχυνση, βαρύτητα, '
    'βήματα και ταξινόμηση κίνησης — κανένα δεν είναι ενεργό. Το firmware '
    'ενεργοποιεί μόνο δύο reports (γωνία και γυροσκόπιο), γιατί όταν ζητούνται '
    'πολλά μαζί το I2C ρίχνει σιωπηλά μερικά — έτσι είχε «πεθάνει» το γυροσκόπιο. '
    'Η επιτάχυνση στέλνεται ως σταθερό 0.':
        ('The BNO085 can also report magnetometer, linear acceleration, gravity, '
         'step count and activity classification — none of them are enabled. The '
         'firmware turns on only two reports (orientation and gyro), because '
         'requesting several at once makes the I2C bus drop some silently — that '
         'is how the gyro once "died". Acceleration is streamed as a constant 0.',
         'Das BNO085 kann auch Magnetometer, lineare Beschleunigung, Schwerkraft, '
         'Schritte und Bewegungsklassifikation liefern — nichts davon ist aktiv. '
         'Die Firmware aktiviert nur zwei Reports (Lage und Gyroskop), denn bei '
         'mehreren gleichzeitig verwirft der I2C-Bus stillschweigend einige — so '
         '„starb“ einmal das Gyroskop. Die Beschleunigung wird als konstante 0 '
         'gesendet.'),
    'BNO085 σε ESP32 (CH340) στο /dev/imu, τοποθετημένο ανάποδα· nRESET στο '
    'GPIO18 ώστε να επανέρχεται μόνο του όταν κολλήσει το πρωτόκολλο. Τροφοδοτεί '
    'τον EKF μαζί με το odometry.':
        ('BNO085 on an ESP32 (CH340) at /dev/imu, mounted UPSIDE-DOWN; nRESET on '
         'GPIO18 so it recovers by itself when the protocol wedges. It feeds the '
         'EKF together with the odometry.',
         'BNO085 an einem ESP32 (CH340) unter /dev/imu, KOPFÜBER montiert; nRESET '
         'an GPIO18, damit es sich selbst erholt, wenn das Protokoll hängt. Es '
         'speist zusammen mit der Odometrie den EKF.'),

    # ── RTAB-Map / 3D map pane ────────────────────────────────────────────
    '3D Χάρτης':         ('3D Map', '3D-Karte'),
    'Χαρτογράφηση':      ('Mapping', 'Kartierung'),
    'Καρέ-κλειδιά':      ('Keyframes', 'Schlüsselbilder'),
    'Κλεισίματα βρόχου': ('Loop closures', 'Schleifenschlüsse'),
    '⏸ Παύση':           ('⏸ Pause', '⏸ Pause'),
    '▶ Συνέχεια':        ('▶ Resume', '▶ Fortsetzen'),
    '🆕 Νέος χάρτης':    ('🆕 New map', '🆕 Neue Karte'),
    'χαρτογραφεί':       ('mapping', 'kartiert'),
    'ανενεργό':          ('idle', 'inaktiv'),
    'Χτίζει τρισδιάστατο χάρτη του σπιτιού από την D435. Οδήγησε αργά και κοίτα '
    'τους τοίχους· ο χάρτης μεγαλώνει μόνο όσο κινείσαι. ΔΕΝ πειράζει την '
    'πλοήγηση: δεν δημοσιεύει TF και ζει σε δικό του namespace.':
        ('Builds a 3D map of the house from the D435. Drive slowly and face the '
         'walls — the map only grows while you move. It does NOT disturb '
         'navigation: it publishes no TF and lives in its own namespace.',
         'Baut eine 3D-Karte der Wohnung aus der D435. Fahre langsam und richte '
         'die Kamera auf die Wände — die Karte wächst nur in Bewegung. Sie '
         'stört die Navigation NICHT: kein TF, eigener Namespace.'),
    'Ο χάρτης αποθηκεύεται μόνος του στο ~/.home_robot/rtabmap/house.db. Για '
    'εξαγωγή σε .ply/.obj χρησιμόποιησε File → Export στο ίδιο το RTAB-Map '
    'παραπάνω. ‼️ Το «Νέος χάρτης» ξεκινά καθαρή συνεδρία — ό,τι έχεις '
    'χαρτογραφήσει ως τώρα μένει στη βάση αλλά βγαίνει από τον τρέχοντα χάρτη.':
        ('The map saves itself to ~/.home_robot/rtabmap/house.db. To export a '
         '.ply/.obj use File → Export in RTAB-Map itself above. ‼️ "New map" '
         'starts a clean session — what you have mapped so far stays in the '
         'database but leaves the current map.',
         'Die Karte speichert sich selbst unter ~/.home_robot/rtabmap/house.db. '
         'Zum Export als .ply/.obj nutze File → Export in RTAB-Map oben. '
         '‼️ „Neue Karte" startet eine saubere Sitzung — das bisher Kartierte '
         'bleibt in der Datenbank, verlässt aber die aktuelle Karte.'),

    # ── NeRF pane ─────────────────────────────────────────────────────────
    'Καταγραφή':         ('Capture', 'Aufnahme'),
    'Καρέ':              ('Frames', 'Bilder'),
    'Φάκελος':           ('Folder', 'Ordner'),
    '⏺ Ξεκίνα καταγραφή': ('⏺ Start capture', '⏺ Aufnahme starten'),
    '■ Σταμάτα':         ('■ Stop', '■ Stopp'),
    'καταγράφει':        ('recording', 'nimmt auf'),
    'Εκπαίδευση':        ('Training', 'Training'),
    'Η εκπαίδευση τρέχει ΞΕΧΩΡΙΣΤΑ, από τερματικό:':
        ('Training runs SEPARATELY, from a terminal:',
         'Das Training läuft SEPARAT, im Terminal:'),
    'Καταγράφει εικόνες μαζί με τις ΜΕΤΡΗΜΕΝΕΣ πόζες της κάμερας από το TF — '
    "γι' αυτό δεν χρειάζεται COLMAP, που είναι το αργό και εύθραυστο μισό κάθε "
    'NeRF. Οδήγησε αργά γύρω από τον χώρο κοιτώντας τον από πολλές γωνίες. '
    'Κρατά καρέ μόνο όταν η κάμερα έχει όντως μετακινηθεί (12cm ή 8°), αλλιώς '
    '30 πανομοιότυπες λήψεις τον δευτερόλεπτο δεν διδάσκουν τίποτα.':
        ('Records images together with the MEASURED camera poses from TF — which '
         'is why it needs no COLMAP, the slow and fragile half of every NeRF. '
         'Drive slowly around the space, looking at it from many angles. Frames '
         'are only kept once the camera has actually moved (12 cm or 8°); '
         'otherwise 30 identical shots a second teach it nothing.',
         'Nimmt Bilder zusammen mit den GEMESSENEN Kameraposen aus TF auf — '
         'deshalb braucht es kein COLMAP, die langsame und fragile Hälfte jedes '
         'NeRF. Fahre langsam umher und blicke aus vielen Winkeln. Bilder werden '
         'nur gespeichert, wenn sich die Kamera wirklich bewegt hat (12 cm oder '
         '8°); sonst lehren 30 identische Aufnahmen pro Sekunde nichts.'),
    '‼️ ΔΕΝ μπορεί να μοιραστεί το iGPU με το perception. Με το object_detector '
    'και το pose_node ενεργά, η εκπαίδευση ΡΙΧΝΕΙ την ουρά του ROCm (memory '
    'aperture violation) — δεν είναι έλλειψη μνήμης, μετρήθηκαν 5.9GB ελεύθερα. '
    'Σταμάτα πρώτα το perception· το script αρνείται να ξεκινήσει και μόνο του. '
    'Μετρημένη ταχύτητα: ~310ms/βήμα, άρα ένα δωμάτιο θέλει 45 λεπτά έως 2 ώρες.':
        ('‼️ It CANNOT share the iGPU with perception. With object_detector and '
         'pose_node running, training brings down the ROCm queue (memory '
         'aperture violation) — not a memory shortage, 5.9 GB was free. Stop '
         'perception first; the script also refuses to start on its own. '
         'Measured speed: ~310 ms/step, so one room takes 45 minutes to 2 hours.',
         '‼️ Es kann die iGPU NICHT mit der Perception teilen. Mit laufendem '
         'object_detector und pose_node bricht das Training die ROCm-Queue ab '
         '(memory aperture violation) — kein Speichermangel, 5,9 GB waren frei. '
         'Stoppe zuerst die Perception; das Skript verweigert den Start auch '
         'selbst. Gemessen: ~310 ms/Schritt, ein Raum dauert 45 Min bis 2 Std.'),

    # ── vacuum base pane ──────────────────────────────────────────────────
    'Βάση Roomba 879':   ('Roomba 879 base', 'Roomba-879-Basis'),
    'Κατάσταση':         ('State', 'Zustand'),
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
    'ΞΥΠΝΙΑ':            ('AWAKE', 'WACH'),
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
    # 'CPU' and 'RAM' are deliberately absent: identical in all three
    # languages, and t() passes anything it does not know through unchanged.
    'Θερμοκρασίες':      ('Temperatures', 'Temperaturen'),
    'Δίσκοι':            ('Disks', 'Datenträger'),
    'Φόρτος':            ('Load', 'Last'),
    'νήματα':            ('threads', 'Threads'),
    'GB ελεύθερα':       ('GB free', 'GB frei'),
    'Κόμβοι ROS (':      ('ROS nodes (', 'ROS-Knoten ('),

    # ── room colours (map pane) ───────────────────────────────────────────
    'Χρώματα':           ('Colours', 'Farben'),

    # ── who is speaking ───────────────────────────────────────────────────
    'Ποιος μιλάει':      ('Who is speaking', 'Wer spricht'),
    'Θέλει use_diarization (ποιος), DoA (από πού), use_face_detection '
    '(ποιον βλέπω). Ό,τι λείπει παραλείπεται.':
        ('Needs use_diarization (who), DoA (from where), use_face_detection '
         '(who is visible). Whatever is missing is left out.',
         'Braucht use_diarization (wer), DoA (woher), use_face_detection '
         '(wer sichtbar ist). Fehlendes wird weggelassen.'),
    'μπροστά':           ('ahead', 'vorne'),
    'δεξιά':             ('right', 'rechts'),
    'αριστερά':          ('left', 'links'),
    'πρόσωπα':           ('faces', 'Gesichter'),
    'άγνωστος':          ('unknown', 'unbekannt'),
    'ταυτοποιημένος':    ('identified', 'identifiziert'),
    'κανείς δεν μιλάει': ('nobody speaking', 'niemand spricht'),

    # ── fall alert ────────────────────────────────────────────────────────
    'Κάποιος μπορεί να έπεσε': ('Someone may have fallen',
                                'Jemand ist möglicherweise gestürzt'),
    'Το είδα':           ('Dismiss', 'Gesehen'),
    'κλίση':             ('tilt', 'Neigung'),

    # ── check mission ("πήγαινε να δεις") ─────────────────────────────────
    '🔎 Πήγαινε να δεις': ('🔎 Go and look', '🔎 Nachsehen'),
    'Έλεγξε':            ('Check', 'Prüfen'),
    'π.χ. είναι κλειστό το παράθυρο;':
        ('e.g. is the window closed?', 'z.B. ist das Fenster zu?'),
    'Ξεκίνησε…':         ('Started…', 'Gestartet…'),
    'Ακυρώθηκε.':        ('Cancelled.', 'Abgebrochen.'),
    'σε αναμονή':        ('idle', 'bereit'),
    'πηγαίνει…':         ('driving…', 'fährt…'),
    'κοιτάζει…':         ('looking…', 'schaut…'),
    'γυρίζει…':          ('returning…', 'kommt zurück…'),
    'ολοκληρώθηκε':      ('done', 'fertig'),
    'απέτυχε':           ('failed', 'fehlgeschlagen'),
    'ακυρώθηκε':         ('cancelled', 'abgebrochen'),

    # ── speed sliders (map pane) ──────────────────────────────────────────
    '🚀 Ταχύτητα':       ('🚀 Speed', '🚀 Geschwindigkeit'),
    '🔄 Στροφή':         ('🔄 Turn rate', '🔄 Drehrate'),
    '🐢 Αργά':           ('🐢 Slow', '🐢 Langsam'),
    'Προεπιλογή':        ('Default', 'Standard'),
    '🐇 Γρήγορα':        ('🐇 Fast', '🐇 Schnell'),

    # ── LLM backend switch ────────────────────────────────────────────────
    'Ποιος απαντά':      ('Who answers', 'Wer antwortet'),
    'Αλλαγή…':           ('Switching…', 'Wechsel…'),
    'Το Gemini απαντά σε ~0.5s και δεν πιάνει μνήμη. Το Qwen τρέχει τοπικά '
    'στο NPU — δεν χρειάζεται ίντερνετ, αλλά αργεί ~6s και κρατά 4.7 GB RAM. '
    'Η αλλαγή σβήνει τις τελευταίες ατάκες της κουβέντας.':
        ('Gemini answers in ~0.5s and uses no memory. Qwen runs locally on the '
         'NPU — no internet needed, but it takes ~6s and holds 4.7 GB of RAM. '
         'Switching clears the last few lines of the conversation.',
         'Gemini antwortet in ~0.5s und braucht keinen Speicher. Qwen läuft '
         'lokal auf der NPU — ohne Internet, aber ~6s langsam und belegt '
         '4,7 GB RAM. Der Wechsel löscht die letzten Gesprächszeilen.'),

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
