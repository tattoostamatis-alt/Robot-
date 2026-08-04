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
    'Σκούπα':            ('Vacuum', 'Sauger'),
    'Σύστημα':           ('System', 'System'),
    'Ρύθμιση':           ('Settings', 'Einstellung'),

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

    # ── 3D arm view ───────────────────────────────────────────────────────
    'Σύρε για περιστροφή · ροδέλα για ζουμ · κινείται με τις αρθρώσεις από κάτω':
        ('Drag to rotate · wheel to zoom · follows the joints below',
         'Ziehen zum Drehen · Rad zum Zoomen · folgt den Gelenken unten'),
    'φόρτωση…':          ('loading…', 'lädt…'),
    'τρίγωνα':           ('triangles', 'Dreiecke'),
    'δεν φορτώθηκε':     ('failed to load', 'nicht geladen'),

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

    '🧺 Μάζεψε':         ('🧺 Tidy up', '🧺 Aufräumen'),
    'Μαζεύει αντικείμενα με τον βραχίονα':
        ('Picks objects up with the arm', 'Hebt Gegenstände mit dem Arm auf'),

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

    # ── gestures pane ─────────────────────────────────────────────────────
    'Δείξε με το χέρι':  ('Point with your hand', 'Mit der Hand zeigen'),
    'Στόχος X':          ('Target X', 'Ziel X'),
    'Στόχος Y':          ('Target Y', 'Ziel Y'),
    'Χέρι':              ('Arm', 'Arm'),
    'Ευθύτητα':          ('Straightness', 'Streckung'),
    '👉 Πήγαινε εκεί':   ('👉 Go there', '👉 Geh dorthin'),
    'δεξί':              ('right', 'rechts'),
    'αριστερό':          ('left', 'links'),
    'κλείδωσε':          ('locked', 'fixiert'),
    'δείχνεις…':         ('pointing…', 'zeigt…'),
    'αδρανές':           ('idle', 'inaktiv'),
    'Στάλθηκε.':         ('Sent.', 'Gesendet.'),
    'Πώς δουλεύει':      ('How it works', 'So funktioniert es'),
    'Τεντώνεις το χέρι και δείχνεις στο ΠΑΤΩΜΑ. Το ρομπότ παίρνει τον ώμο και '
    'τον καρπό σου από τον σκελετό (pose_node), τα ανεβάζει σε 3D με το βάθος '
    'της D435, και προεκτείνει τη γραμμή ώσπου να συναντήσει το δάπεδο. '
    'Ο κύκλος γεμίζει καθώς μαζεύονται καρέ που συμφωνούν· γίνεται πράσινος '
    'όταν κλειδώσει.':
        ('Extend your arm and point at the FLOOR. The robot takes your shoulder '
         'and wrist from the skeleton (pose_node), lifts them into 3D using the '
         "D435's depth, and extends the line until it meets the floor. The ring "
         'fills as agreeing frames accumulate; it turns green when it locks.',
         'Strecke den Arm aus und zeige auf den BODEN. Der Roboter nimmt Schulter '
         'und Handgelenk aus dem Skelett (pose_node), hebt sie mit der Tiefe der '
         'D435 ins 3D und verlängert die Linie bis zum Boden. Der Ring füllt '
         'sich, während übereinstimmende Bilder gesammelt werden; er wird grün, '
         'sobald er fixiert.'),
    '‼️ ΔΕΝ οδηγεί μόνο του. Το να δείξεις κάτι γίνεται πολύ εύκολα κατά λάθος '
    '— απλώνοντας το χέρι για μια κούπα γράφεις την ίδια γεωμετρία — οπότε '
    'χρειάζεται ρητή επιβεβαίωση: αυτό το κουμπί ή «πήγαινε εκεί» με τη φωνή. '
    'Θέλει':
        ("‼️ It does NOT drive on its own. Pointing is far too easy to do by "
         'accident — reaching for a cup traces the same geometry — so it needs '
         'explicit confirmation: this button, or "πήγαινε εκεί" by voice. '
         'Requires',
         '‼️ Er fährt NICHT von selbst. Zeigen passiert viel zu leicht '
         'versehentlich — nach einer Tasse greifen ergibt dieselbe Geometrie — '
         'daher braucht es eine ausdrückliche Bestätigung: diese Taste oder '
         '„πήγαινε εκεί“ per Sprache. Benötigt'),
    'και ευθυγραμμισμένο βάθος.': ('and aligned depth.',
                                   'und ausgerichtete Tiefe.'),

    # ── observations pane ─────────────────────────────────────────────────
    'Πρόσεξα':           ('Noticed', 'Bemerkt'),
    'Τι πρόσεξε':        ('What it noticed', 'Was aufgefallen ist'),
    'Καμία παρατήρηση ακόμη.': ('No observations yet.',
                                'Noch keine Beobachtungen.'),
    'γνωστά αντικείμενα': ('known objects', 'bekannte Objekte'),
    'Πώς μαθαίνει':      ('How it learns', 'So lernt er'),
    'Το ρομπότ χτίζει μόνο του μια βάση αναφοράς: πού «ζει» κανονικά κάθε '
    'αντικείμενο. Ένα αντικείμενο μετράει ως μόνιμο μόνο αφού το δει στο ίδιο '
    'σημείο σε ΞΕΧΩΡΙΣΤΕΣ επισκέψεις, με απόσταση μεταξύ τους — αλλιώς μια '
    'παρατεταμένη ματιά σε ένα δωμάτιο θα γινόταν «κανονικότητα» και όλα μετά '
    'θα έμοιαζαν μετακινημένα.':
        ('The robot builds its own baseline of where each object normally '
         'lives. An object counts as settled only after being seen in the same '
         'spot on SEPARATE visits, well apart in time — otherwise one long look '
         'at a room would become "normal" and everything after it would look '
         'moved.',
         'Der Roboter baut selbst eine Referenz auf, wo jedes Objekt '
         'normalerweise liegt. Ein Objekt gilt erst als fest, wenn es an '
         'derselben Stelle bei GETRENNTEN Besuchen gesehen wurde — sonst würde '
         'ein einziger langer Blick in einen Raum zur „Normalität“ und alles '
         'Spätere sähe verschoben aus.'),
    'Σε καινούριο χάρτη θα σιωπά για μέρες. Αυτό είναι το σωστό: δεν ξέρει '
    'ακόμη τι σημαίνει «κανονικά». Μιλάει ΜΟΝΟ όταν υπάρχει άνθρωπος μπροστά '
    'του, το πολύ 4 φορές την ώρα, ποτέ 23:00–08:00, και ποτέ την ίδια '
    'παρατήρηση δύο φορές σε 2 ώρες.':
        ('On a fresh map it will stay silent for days. That is correct: it does '
         'not yet know what "normal" means. It speaks ONLY with a person in '
         'view, at most 4 times an hour, never between 23:00 and 08:00, and '
         'never the same remark twice within 2 hours.',
         'Auf einer neuen Karte schweigt er tagelang. Das ist richtig: er weiß '
         'noch nicht, was „normal“ heißt. Er spricht NUR, wenn eine Person zu '
         'sehen ist, höchstens 4-mal pro Stunde, nie zwischen 23:00 und 08:00, '
         'und nie dieselbe Bemerkung zweimal in 2 Stunden.'),

    # ── timeline pane ─────────────────────────────────────────────────────
    'Χρονολόγιο':        ('Timeline', 'Zeitleiste'),
    'Χρονικό':           ('Timeline', 'Verlauf'),
    'Χέρι':              ('Arm', 'Arm'),
    'Σπίτι 3D':          ('House 3D', 'Haus 3D'),
    'Φωνή':              ('Voice', 'Sprache'),

    # ── compass ───────────────────────────────────────────────────────────
    'Πυξίδα':            ('Compass', 'Kompass'),
    'Κατεύθυνση':        ('Direction', 'Richtung'),
    'Μοίρες':            ('Degrees', 'Grad'),
    'Βορράς στον χάρτη': ('North on the map', 'Norden auf der Karte'),
    '🧭 Κοιτάω Βορρά':   ('🧭 I am facing north', '🧭 Ich schaue nach Norden'),
    'Βορρά':             ('north', 'Norden'),
    'Ανατολή':           ('east', 'Osten'),
    'Νότο':              ('south', 'Süden'),
    'Δύση':              ('west', 'Westen'),
    '✕ Καθάρισε':        ('✕ Clear', '✕ Löschen'),
    'αβαθμονόμητη':      ('not calibrated', 'nicht kalibriert'),
    'χωρίς εντοπισμό':   ('no localization', 'keine Lokalisierung'),
    'βαθμονομημένη':     ('calibrated', 'kalibriert'),
    'Βαθμονομήθηκε.':    ('Calibrated.', 'Kalibriert.'),
    'Δεν υπάρχει θέση — κάνε πρώτα εντοπισμό.':
        ('No pose — localize first.', 'Keine Position — zuerst lokalisieren.'),
    # Compass rose letters.
    'Β': ('N', 'N'),   'ΒΑ': ('NE', 'NO'),  'Α': ('E', 'O'),   'ΝΑ': ('SE', 'SO'),
    'Ν': ('S', 'S'),   'ΝΔ': ('SW', 'SW'),  'Δ': ('W', 'W'),   'ΒΔ': ('NW', 'NW'),
    'Η πυξίδα ΔΕΝ βγαίνει από μαγνητόμετρο — βγαίνει από τον ΧΑΡΤΗ. Ο χάρτης '
    'δεν γυρίζει ποτέ και το AMCL διορθώνει τη γωνία πάνω σε αληθινούς '
    'τοίχους, οπότε μέσα στο σπίτι είναι πολύ σταθερότερο από κάθε '
    'μαγνητόμετρο δίπλα σε μοτέρ σκούπας.':
        ('The compass does NOT come from a magnetometer — it comes from the '
         'MAP. The map never rotates and AMCL corrects the angle against real '
         'walls, which indoors is far steadier than any magnetometer next to a '
         'vacuum motor.',
         'Der Kompass kommt NICHT von einem Magnetometer — er kommt von der '
         'KARTE. Die Karte dreht sich nie und AMCL korrigiert den Winkel an '
         'echten Wänden, was drinnen weit stabiler ist als jedes Magnetometer '
         'neben einem Saugermotor.'),
    'Λείπει μόνο ΕΝΑ νούμερο: προς τα πού είναι ο Βορράς μέσα στον χάρτη. '
    'Γύρνα το ρομπότ να κοιτάει Βορρά και πάτα το κουμπί — μία φορά για κάθε '
    'χάρτη. Αν ξέρεις ότι κοιτάει άλλη κατεύθυνση, διάλεξέ την από τη λίστα. '
    '‼️ Θέλει εντοπισμό: χωρίς AMCL δεν υπάρχει γωνία να μετρηθεί.':
        ('Only ONE number is missing: which way north points inside the map. '
         'Turn the robot to face north and press the button — once per map. If '
         'you know it faces some other direction, pick that from the list. '
         '‼️ Needs localization: without AMCL there is no angle to measure.',
         'Es fehlt nur EINE Zahl: wohin Norden in der Karte zeigt. Drehe den '
         'Roboter nach Norden und drücke die Taste — einmal pro Karte. Wenn du '
         'weißt, dass er anders steht, wähle die Richtung aus der Liste. '
         '‼️ Benötigt Lokalisierung: ohne AMCL gibt es keinen Winkel.'),
    'Ρώτα τη μνήμη':     ('Ask the memory', 'Frag das Gedächtnis'),
    'Ρώτα':              ('Ask', 'Fragen'),
    'π.χ. τι έγινε σήμερα το πρωί;': ('e.g. what happened this morning?',
                                      'z.B. was ist heute Morgen passiert?'),
    'γεγονότα':          ('events', 'Ereignisse'),
    'Άδειο.':            ('Empty.', 'Leer.'),
    # The chips display these but SEND the Greek — parse_time_window is Greek.
    'σήμερα':            ('today', 'heute'),
    'σήμερα το πρωί':    ('this morning', 'heute Morgen'),
    'χθες':              ('yesterday', 'gestern'),
    'πριν από 2 ώρες':   ('2 hours ago', 'vor 2 Stunden'),

    # ── open-vocabulary search pane ───────────────────────────────────────
    'Ψάξε':              ('Search', 'Suchen'),
    'Ψάξε ό,τι θέλεις':  ('Search for anything', 'Suche irgendetwas'),
    '🔎 Ψάξε':           ('🔎 Search', '🔎 Suchen'),
    'π.χ. κλειδιά, γυαλιά, φορτιστής': ('e.g. keys, glasses, charger',
                                        'z.B. Schlüssel, Brille, Ladegerät'),
    'Ελευθερώνει το iGPU': ('Releases the iGPU', 'Gibt die iGPU frei'),
    'Τι βλέπει':         ('What it sees', 'Was er sieht'),
    'Δεν ψάχνει τίποτα.': ('Not searching for anything.', 'Sucht nach nichts.'),
    'Δεν το βλέπω.':     ('I do not see it.', 'Ich sehe es nicht.'),
    'ευρήματα':          ('hits', 'Treffer'),
    'Ψάχνω: ':           ('Searching for: ', 'Suche nach: '),
    'φορτώνει…':         ('loading…', 'lädt…'),
    'ψάχνει':            ('searching', 'sucht'),
    'άγνωστο':           ('unknown', 'unbekannt'),
    # Chips: displayed translated, but the Greek is what gets sent.
    'κλειδιά':           ('keys', 'Schlüssel'),
    'γυαλιά':            ('glasses', 'Brille'),
    'φορτιστής':         ('charger', 'Ladegerät'),
    'τηλεκοντρόλ':       ('remote', 'Fernbedienung'),
    'πορτοφόλι':         ('wallet', 'Geldbörse'),
    'Το κανονικό YOLO ξέρει 80 σταθερές κατηγορίες — κούπες, καρέκλες, '
    'βιβλία. «Κλειδιά», «φορτιστής», «πορτοφόλι» ΔΕΝ υπάρχουν σε αυτές. '
    'Το YOLO-World παίρνει τη λίστα ως κείμενο, οπότε ψάχνει ό,τι του πεις.':
        ('Ordinary YOLO knows 80 fixed classes — cups, chairs, books. "Keys", '
         '"charger" and "wallet" are simply not among them. YOLO-World takes '
         'the class list as text, so it looks for whatever you name.',
         'Normales YOLO kennt 80 feste Klassen — Tassen, Stühle, Bücher. '
         '„Schlüssel“, „Ladegerät“ und „Geldbörse“ sind schlicht nicht dabei. '
         'YOLO-World nimmt die Klassenliste als Text und sucht daher, was immer '
         'du nennst.'),
    '‼️ Ξεκινά ΜΟΝΟ όταν ζητήσεις κάτι και σβήνει μόνο του μετά από 90 '
    'δευτερόλεπτα. Ένας δεύτερος ανιχνευτής που τρέχει συνέχεια στο ίδιο '
    'iGPU είναι ακριβώς το πρόβλημα φόρτου που έχει ξαναχτυπήσει εδώ. '
    'Θέλει':
        ('‼️ It starts ONLY when you ask for something and idles again after 90 '
         'seconds. A second detector running continuously on the same iGPU is '
         'exactly the load problem that has bitten this project before. '
         'Requires',
         '‼️ Er startet NUR auf Anfrage und schaltet nach 90 Sekunden wieder ab. '
         'Ein zweiter Detektor, der dauerhaft auf derselben iGPU läuft, ist '
         'genau das Lastproblem, das dieses Projekt schon getroffen hat. '
         'Benötigt'),

    # ── gesture bindings (in the Gestures tab) ────────────────────────────
    'Χειρονομίες':       ('Gestures', 'Gesten'),
    'Τι κάνει η κάθε χειρονομία': ('What each gesture does',
                                   'Was jede Geste bewirkt'),
    'στάση σώματος':     ('body pose', 'Körperhaltung'),
    'δάχτυλα':           ('fingers', 'Finger'),
    'κινεί':             ('moves', 'bewegt'),
    'Να επιτρέπονται χειρονομίες που ΚΙΝΟΥΝ το ρομπότ':
        ('Allow gestures that MOVE the robot',
         'Gesten zulassen, die den Roboter BEWEGEN'),
    'κίνηση ΕΝΕΡΓΗ':     ('motion ON', 'Bewegung AN'),
    'μόνο ασφαλείς':     ('safe only', 'nur sichere'),
    'Αποθηκεύτηκε.':     ('Saved.', 'Gespeichert.'),
    '‼️ Οι χειρονομίες που ΣΤΑΜΑΤΟΥΝ δουλεύουν πάντα, ακόμη κι όταν ο '
    'διακόπτης είναι κλειστός — το αντίθετο θα ήταν η χειρότερη δυνατή '
    'συμπεριφορά. Όσες ΞΕΚΙΝΟΥΝ κίνηση θέλουν διπλάσιο κράτημα, γιατί ένα '
    'λάθος «έλα εδώ» στέλνει μηχάνημα πάνω σε άνθρωπο.':
        ('‼️ Gestures that STOP always work, even with the switch off — the '
         'opposite would be the worst possible behaviour. Ones that START '
         'motion need twice the hold, because a mistaken "come here" sends a '
         'machine at a person.',
         '‼️ Gesten, die STOPPEN, funktionieren immer, auch bei ausgeschaltetem '
         'Schalter — das Gegenteil wäre das schlimmstmögliche Verhalten. Gesten, '
         'die Bewegung STARTEN, brauchen doppelt so langes Halten, denn ein '
         'irrtümliches „komm her" schickt eine Maschine auf einen Menschen zu.'),
    'Οι στάσεις σώματος διαβάζονται από απόσταση· τα δάχτυλα θέλουν '
    'κοντινή απόσταση και':
        ('Body poses read from across the room; fingers need to be close, and',
         'Körperhaltungen sind aus der Ferne lesbar; Finger brauchen Nähe und'),

    # ── people tab ────────────────────────────────────────────────────────
    'Άτομα':             ('People', 'Personen'),
    'Βλέπω':             ('Seeing', 'Sehe'),
    'Ακούω':             ('Hearing', 'Höre'),
    'Ύψος':              ('Height', 'Größe'),
    'Γιατί':             ('Why', 'Warum'),
    'Πρόσθεσε άτομο':    ('Add a person', 'Person hinzufügen'),
    '➕ Πρόσθεσε':       ('➕ Add', '➕ Hinzufügen'),
    'Γνωστά άτομα':      ('Known people', 'Bekannte Personen'),
    'άτομα':             ('people', 'Personen'),
    'Κανένα άτομο ακόμη. Γράψε ένα όνομα παραπάνω.':
        ('Nobody yet. Type a name above.',
         'Noch niemand. Gib oben einen Namen ein.'),
    'πλήρες':            ('complete', 'vollständig'),
    'ελλιπές':           ('incomplete', 'unvollständig'),
    'Πρόσωπο':           ('Face', 'Gesicht'),
    'Μάθε πρόσωπο':      ('Learn face', 'Gesicht lernen'),
    'Μάθε φωνή':         ('Learn voice', 'Stimme lernen'),
    'Κοίτα την κάμερα: ': ('Look at the camera: ', 'Schau in die Kamera: '),
    'Μίλα τώρα: ':       ('Speak now: ', 'Sprich jetzt: '),
    'Διαγραφή; ':        ('Delete? ', 'Löschen? '),
    'Προστέθηκε: ':      ('Added: ', 'Hinzugefügt: '),
    'Γράψε πρώτα ένα όνομα.': ('Type a name first.',
                               'Gib zuerst einen Namen ein.'),
    'Τρία σήματα, το καθένα τυφλό αλλού. Το ΠΡΟΣΩΠΟ δουλεύει σιωπηλά και '
    'από απόσταση, αλλά όχι στο σκοτάδι ή από πίσω. Η ΦΩΝΗ δουλεύει στο '
    'σκοτάδι και πίσω από γωνίες, αλλά μόνο όσο κάποιος μιλάει. Το ΥΨΟΣ '
    'υπάρχει πάντα.':
        ('Three signals, each blind somewhere else. The FACE works silently '
         'and at a distance, but not in the dark or from behind. The VOICE '
         'works in the dark and around corners, but only while somebody is '
         'talking. HEIGHT is always there.',
         'Drei Signale, jedes anderswo blind. Das GESICHT wirkt lautlos und '
         'auf Distanz, aber nicht im Dunkeln oder von hinten. Die STIMME wirkt '
         'im Dunkeln und um Ecken, aber nur solange jemand spricht. Die GRÖSSE '
         'ist immer da.'),
    '‼️ Το ύψος ΔΕΝ αναγνωρίζει από μόνο του — δύο ενήλικες διαφέρουν '
    'συχνά λίγα εκατοστά. Χρησιμεύει για να ΑΠΟΚΛΕΙΕΙ: «όποιος κι αν '
    'είναι, δεν είναι το παιδί». Μετριέται από το πάτωμα του χάρτη, οπότε '
    'θέλει εντοπισμό — χωρίς αυτόν δεν δείχνει ύψος αντί για λάθος ύψος.':
        ('‼️ Height does NOT identify on its own — two adults are often within '
         'a few centimetres. It is there to RULE OUT: "whoever that is, it is '
         'not the child". It is measured from the map floor, so it needs '
         'localization — without it there is no height rather than a wrong one.',
         '‼️ Die Größe identifiziert NICHT allein — zwei Erwachsene liegen oft '
         'wenige Zentimeter auseinander. Sie dient dem AUSSCHLIESSEN: „wer das '
         'auch ist, es ist nicht das Kind". Gemessen wird ab dem Kartenboden, '
         'also braucht es Lokalisierung — ohne sie gibt es keine Größe statt '
         'einer falschen.'),

    # ── who is here (face identity) ───────────────────────────────────────
    'Ποιος είναι εδώ':   ('Who is here', 'Wer ist hier'),
    'όνομα':             ('name', 'Name'),
    'άγνωστος':          ('unknown', 'unbekannt'),
    'Γράψε πρώτα ένα όνομα.': ('Type a name first.',
                               'Gib zuerst einen Namen ein.'),

    # ── self-diagnosis ────────────────────────────────────────────────────
    'Αυτοδιάγνωση':      ('Self-diagnosis', 'Selbstdiagnose'),
    'τίποτα γνωστό':     ('nothing known', 'nichts Bekanntes'),
    'σοβαρά':            ('critical', 'kritisch'),
    'προειδοποιήσεις':   ('warnings', 'Warnungen'),
    'Κανένα γνωστό πρόβλημα.': ('No known problem.', 'Kein bekanntes Problem.'),
    'Ελέγχει τους ΓΝΩΣΤΟΥΣ τρόπους που χαλάει αυτό το ρομπότ — αυτούς που '
    'έχουν ήδη κοστίσει χρόνο. Σχεδόν όλοι είναι ΣΙΩΠΗΛΟΙ: ο κόμβος ζει, '
    'το topic υπάρχει, και μόνο η συμπεριφορά είναι λάθος. Κενή λίστα '
    'σημαίνει «τίποτα γνωστό», όχι «όλα καλά».':
        ('It checks the KNOWN ways this robot breaks — the ones that have '
         'already cost time. Almost all of them are SILENT: the node is alive, '
         'the topic exists, and only the behaviour is wrong. An empty list '
         'means "nothing known", not "all good".',
         'Es prüft die BEKANNTEN Fehlerarten dieses Roboters — die, die schon '
         'Zeit gekostet haben. Fast alle sind STUMM: der Node lebt, das Topic '
         'existiert, nur das Verhalten ist falsch. Eine leere Liste heißt '
         '„nichts Bekanntes", nicht „alles gut".'),

    # ── live microphone ───────────────────────────────────────────────────
    '🔊 Άκου το μικρόφωνο': ('🔊 Listen to the microphone',
                             '🔊 Mikrofon abhören'),
    'Ακούς ζωντανά.':    ('Listening live.', 'Live-Wiedergabe.'),
    'Χωρίς υποστήριξη ήχου.': ('No audio support.', 'Kein Audio-Support.'),
    'ΖΩΝΤΑΝΟΣ ήχος από το δωμάτιο όπου βρίσκεται το ρομπότ — ακούς ό,τι '
    'ακούει. Ίδιο κανάλι με το «Έι ρομπότ», καθαρισμένο από τον XVF3800. '
    'Ξεκινά μόνο όταν το πατήσεις και κόβεται μόλις κλείσεις το tab.':
        ('LIVE audio from the room the robot is in — you hear what it hears. '
         'The same channel as the wake word, cleaned up by the XVF3800. It '
         'starts only when you press it and stops the moment you close the tab.',
         'LIVE-Ton aus dem Raum, in dem der Roboter steht — du hörst, was er '
         'hört. Derselbe Kanal wie das Wakeword, vom XVF3800 aufbereitet. '
         'Startet nur auf Tastendruck und endet, sobald du den Tab schließt.'),

    # ── touch (arm pane) ──────────────────────────────────────────────────
    'Αφή':               ('Touch', 'Tastsinn'),
    'Πίεση':             ('Pressure', 'Druck'),
    'Σκληρότητα':        ('Hardness', 'Härte'),
    'Βάρος':             ('Weight', 'Gewicht'),
    'ΕΠΑΦΗ':             ('CONTACT', 'KONTAKT'),
    'περιμένει':         ('waiting', 'wartet'),
    'ελεύθερος':         ('free', 'frei'),
    'χωρίς αναφορά':     ('no reference', 'keine Referenz'),
    'Ο βραχίονας στέλνει ΦΟΡΤΙΟ ανά άρθρωση δίπλα σε κάθε γωνία, και μέχρι '
    'τώρα δεν το διάβαζε κανείς. Η επαφή είναι ΣΚΑΛΟΠΑΤΙ πάνω από το '
    'φορτίο που κουβαλά κινούμενος στον αέρα· η σκληρότητα είναι φορτίο '
    'ανά χιλιοστό διαδρομής· το βάρος είναι η διαφορά στον ώμο με κλειστή '
    'και ανοιχτή δαγκάνα.':
        ('The arm reports LOAD per joint next to every angle, and until now '
         'nothing read it. Contact is a STEP above the load it carries moving '
         'through air; hardness is load per millimetre of travel; weight is '
         'the difference at the shoulder between a closed and an open gripper.',
         'Der Arm meldet neben jedem Winkel auch die LAST pro Gelenk, und '
         'bisher hat sie niemand gelesen. Kontakt ist eine STUFE über der Last '
         'in der Luft; Härte ist Last pro Millimeter Weg; Gewicht ist die '
         'Differenz an der Schulter zwischen geschlossenem und offenem '
         'Greifer.'),
    '‼️ Οι τιμές είναι ΑΚΑΤΕΡΓΑΣΤΕΣ μονάδες σερβοκινητήρα, όχι γραμμάρια. '
    'Συγκρίνονται μεταξύ τους και με τίποτα άλλο — γι\' αυτό λέει «βαρύ» '
    'και όχι «180 γραμμάρια». Η μέτρηση είναι ΠΑΘΗΤΙΚΗ: δεν δίνει ποτέ '
    'εντολή στον βραχίονα.':
        ('‼️ The values are RAW servo units, not grams. They compare to each '
         'other and to nothing else — which is why it says "heavy" and not '
         '"180 grams". The measurement is PASSIVE: it never commands the arm.',
         '‼️ Die Werte sind ROHE Servo-Einheiten, keine Gramm. Sie sind '
         'untereinander vergleichbar und sonst mit nichts — deshalb heißt es '
         '„schwer" und nicht „180 Gramm". Die Messung ist PASSIV: sie steuert '
         'den Arm nie.'),

    # ── system settings (network / bluetooth / audio / power) ─────────────
    'Δίκτυο':            ('Network', 'Netzwerk'),
    'Διευθύνσεις':       ('Addresses', 'Adressen'),
    '🔄 Σάρωση':         ('🔄 Scan', '🔄 Suchen'),
    'κωδικός δικτύου':   ('network password', 'Netzwerk-Passwort'),
    'συνδεδεμένο':       ('connected', 'verbunden'),
    'εκτός':             ('offline', 'offline'),
    'Πάτα σάρωση.':      ('Press scan.', 'Suchen drücken.'),
    'Συνδέομαι…':        ('Connecting…', 'Verbinde…'),
    'Σάρωση…':           ('Scanning…', 'Suche…'),
    'Αναζήτηση…':        ('Searching…', 'Suche…'),
    '⏻ Ενεργό':          ('⏻ On', '⏻ Ein'),
    '🔍 Αναζήτηση (10s)': ('🔍 Search (10s)', '🔍 Suchen (10s)'),
    'ενεργό':            ('on', 'ein'),
    'ανενεργό':          ('off', 'aus'),
    'Αποσύνδεση':        ('Disconnect', 'Trennen'),
    'Καμία συσκευή.':    ('No devices.', 'Keine Geräte.'),
    'Ήχος & σύστημα':    ('Audio & system', 'Audio & System'),
    'Ένταση':            ('Volume', 'Lautstärke'),
    '↻ Επανεκκίνηση':    ('↻ Reboot', '↻ Neustart'),
    '⏻ Τερματισμός':     ('⏻ Shut down', '⏻ Herunterfahren'),
    'Επανεκκίνηση του υπολογιστή;': ('Reboot the computer?',
                                     'Rechner neu starten?'),
    'Τερματισμός; Θα χρειαστεί να το ανάψεις με το χέρι.':
        ('Shut down? You will have to switch it back on by hand.',
         'Herunterfahren? Du musst ihn von Hand wieder einschalten.'),
    '‼️ Ο κωδικός ταξιδεύει μέσα από αυτή τη σελίδα. Το token του πίνακα '
    'είναι αδύναμο και ακούει σε ΟΛΟ το τοπικό δίκτυο — σύνδεσε νέο WiFi '
    'από εδώ μόνο αν εμπιστεύεσαι όποιον είναι στο ίδιο δίκτυο.':
        ('‼️ The password travels through this page. The dashboard token is '
         'weak and it listens on the WHOLE local network — only join a new '
         'WiFi from here if you trust everyone on that network.',
         '‼️ Das Passwort läuft über diese Seite. Das Dashboard-Token ist '
         'schwach und lauscht im GESAMTEN lokalen Netz — verbinde ein neues '
         'WLAN nur von hier, wenn du allen im Netz vertraust.'),

    # ── dashboard key ─────────────────────────────────────────────────────
    'Ασφάλεια':          ('Security', 'Sicherheit'),
    '🔑 Νέο κλειδί':     ('🔑 New key', '🔑 Neuer Schlüssel'),
    'νέο κλειδί':        ('new key', 'neuer Schlüssel'),
    'Κράτησέ το τώρα — μετά την επανεκκίνηση χρειάζεται:':
        ('Save it now — it is needed after the restart:',
         'Jetzt speichern — nach dem Neustart wird er gebraucht:'),
    'Νέο κλειδί; Ο παλιός σύνδεσμος θα πάψει να δουλεύει.':
        ('New key? The old link will stop working.',
         'Neuer Schlüssel? Der alte Link funktioniert dann nicht mehr.'),
    'Ο πίνακας ακούει σε ΟΛΟ το τοπικό δίκτυο και δίνει κάμερα, μικρόφωνο, '
    'χειριστήριο και χάρτη του σπιτιού. Το κλειδί είναι το μόνο που τον '
    'προστατεύει. Το νέο ισχύει μετά από επανεκκίνηση — η τρέχουσα καρτέλα '
    'συνεχίζει να δουλεύει ώσπου τότε. Άνοιξε τον νέο σύνδεσμο μία φορά σε '
    'κάθε συσκευή.':
        ('The dashboard listens on the WHOLE local network and hands over the '
         'camera, the microphone, the drive controls and a map of the home. '
         'The key is the only thing protecting it. A new one takes effect '
         'after a restart — this tab keeps working until then. Open the new '
         'link once on every device.',
         'Das Dashboard lauscht im GESAMTEN lokalen Netz und gibt Kamera, '
         'Mikrofon, Fahrsteuerung und eine Karte der Wohnung frei. Der '
         'Schlüssel ist das Einzige, was es schützt. Ein neuer gilt nach einem '
         'Neustart — dieser Tab funktioniert bis dahin weiter. Öffne den neuen '
         'Link einmal auf jedem Gerät.'),

    # ── echolocation ──────────────────────────────────────────────────────
    'Ηχοεντοπισμός':     ('Echolocation', 'Echoortung'),
    'Αντήχηση':          ('Reverberation', 'Nachhall'),
    'Πρώτη ανάκλαση':    ('First reflection', 'Erste Reflexion'),
    'Ετυμηγορία':        ('Verdict', 'Ergebnis'),
    '📡 Μέτρησε τον χώρο': ('📡 Measure the room', '📡 Raum messen'),
    'Τσιρίζει…':         ('Chirping…', 'Zirpt…'),
    'μετράει…':          ('measuring…', 'misst…'),
    'δωμάτια':           ('rooms', 'Räume'),
    'Το ρομπότ βγάζει ένα σύντομο τσίρπισμα και ακούει την απάντηση του '
    'δωματίου. Το XVF3800 έχει ακύρωση ηχούς ακριβώς για να ακούει ΕΝΩ '
    'μιλάει — γι\' αυτό γίνεται. Ένας γυμνός διάδρομος αντηχεί· ένα σαλόνι '
    'με καναπέδες όχι.':
        ('The robot emits a short chirp and listens to the room answer. The '
         'XVF3800 has echo cancellation precisely so it can hear WHILE it '
         'speaks — which is what makes this possible. A bare corridor rings; a '
         'living room full of sofas does not.',
         'Der Roboter gibt einen kurzen Chirp aus und hört der Antwort des '
         'Raums zu. Der XVF3800 hat Echounterdrückung genau dafür, WÄHREND des '
         'Sprechens zu hören — deshalb geht das. Ein kahler Flur hallt; ein '
         'Wohnzimmer voller Sofas nicht.'),
    '‼️ Αυτό που εμπιστεύεσαι είναι η ΑΛΛΑΓΗ, όχι οι απόλυτοι αριθμοί. Το '
    'ηχείο, το μικρόφωνο και το ίδιο το σώμα του ρομπότ μπαίνουν το ίδιο '
    'σε δύο μετρήσεις από το ίδιο σημείο και αλληλοαναιρούνται. Δεν τρέχει '
    'ΠΟΤΕ μόνο του: το τσίρπισμα ακούγεται, και ρομπότ που τσιρίζει στις '
    '3 τα ξημερώματα το ξεσυνδέεις. Θέλει':
        ('‼️ What you trust is the CHANGE, not the absolute numbers. The '
         'speaker, the microphone and the robot\'s own body enter two '
         'measurements from the same spot identically and cancel out. It never '
         'runs by itself: the chirp is audible, and a robot that chirps at 3am '
         'gets unplugged. Requires',
         '‼️ Vertrauen kannst du der VERÄNDERUNG, nicht den absoluten Zahlen. '
         'Lautsprecher, Mikrofon und der Roboterkörper gehen in zwei Messungen '
         'vom selben Ort identisch ein und heben sich auf. Es läuft nie von '
         'selbst: der Chirp ist hörbar, und ein Roboter, der um 3 Uhr zirpt, '
         'wird ausgesteckt. Benötigt'),

    # ── acoustic map ──────────────────────────────────────────────────────
    'Από πού ακούγονται': ('Where they come from', 'Woher sie kommen'),
    'εντοπισμένα':       ('located', 'geortet'),
    'θέλει δεύτερο σημείο': ('needs a second spot', 'braucht einen 2. Standort'),
    'Το μικρόφωνο δίνει ΚΑΤΕΥΘΥΝΣΗ, όχι θέση. Μία γωνία από ένα σημείο '
    'είναι ακτίνα, όχι σημείο — γι\' αυτό η θέση εμφανίζεται μόνο αφού '
    'ακουστεί ο ίδιος ήχος από ΔΥΟ διαφορετικά σημεία. Το δωμάτιο όμως '
    'βγαίνει από την πρώτη κιόλας φορά, ακολουθώντας την ακτίνα πάνω '
    'στον χάρτη.':
        ('The microphone gives a DIRECTION, not a position. One bearing from '
         'one spot is a ray, not a point — so a location only appears once the '
         'same sound has been heard from TWO different places. The room, '
         'though, comes out the very first time, by following the ray across '
         'the map.',
         'Das Mikrofon liefert eine RICHTUNG, keine Position. Eine Peilung von '
         'einem Ort ist ein Strahl, kein Punkt — eine Position erscheint erst, '
         'wenn derselbe Ton von ZWEI Stellen gehört wurde. Der Raum ergibt '
         'sich dagegen schon beim ersten Mal, indem der Strahl über die Karte '
         'verfolgt wird.'),

    # ── sound events pane ─────────────────────────────────────────────────
    'Ήχοι':              ('Sounds', 'Geräusche'),
    'Τι ακούω':          ('What I hear', 'Was ich höre'),
    'Κατεύθυνση':        ('Direction', 'Richtung'),
    'Ομιλία':            ('Speech', 'Sprache'),
    'Παράθυρα':          ('Windows', 'Fenster'),
    'Ιστορικό ήχων':     ('Sound history', 'Geräuschverlauf'),
    'Τίποτα ακόμη.':     ('Nothing yet.', 'Noch nichts.'),
    'ακούει':            ('listening', 'hört zu'),
    'σε παύση':          ('paused', 'pausiert'),
    'ησυχία':            ('quiet', 'still'),
    'Το YAMNet αναγνωρίζει 521 ήχους· εδώ κρατάμε τους δώδεκα που αφορούν '
    'ένα σπίτι: κουδούνι, σπασμένο γυαλί, συναγερμός, μωρό που κλαίει, '
    'νερό που τρέχει, κάτι που έπεσε. Η ΟΜΙΛΙΑ δεν αναγγέλλεται ποτέ — '
    'είναι ο πιο συχνός ήχος σε ένα σπίτι και τον χειρίζεται ήδη η φωνή.':
        ('YAMNet recognises 521 sounds; we keep the dozen that matter in a '
         'house: doorbell, breaking glass, alarm, a baby crying, running water, '
         'something falling. SPEECH is never announced — it is the most common '
         'sound in a home and the voice stack already handles it.',
         'YAMNet erkennt 521 Geräusche; wir behalten das Dutzend, das in einem '
         'Haus zählt: Türklingel, Glasbruch, Alarm, weinendes Baby, laufendes '
         'Wasser, etwas fällt. SPRACHE wird nie gemeldet — sie ist das '
         'häufigste Geräusch daheim und die Sprachkette behandelt sie bereits.'),
    '‼️ ΔΕΝ ανοίγει το μικρόφωνο. Διαβάζει το':
        ('‼️ It does NOT open the microphone. It reads',
         '‼️ Er öffnet das Mikrofon NICHT. Er liest'),
    'που δημοσιεύει ο κόμβος wake word — δεύτερο ALSA stream στην ίδια '
    'συσκευή θα τσακωνόταν με το «Έι ρομπότ». Θέλει':
        ('published by the wake word node — a second ALSA stream on the same '
         'device would fight the wake word for it. Requires',
         'das vom Wakeword-Knoten veröffentlicht wird — ein zweiter ALSA-Stream '
         'auf demselben Gerät würde mit dem Wakeword konkurrieren. Benötigt'),
}


def as_js_table() -> dict:
    """{greek: {'en': ..., 'de': ...}} — the shape the page's t() expects."""
    return {src: {'en': en, 'de': de} for src, (en, de) in TRANSLATIONS.items()}
