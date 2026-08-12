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
    '💾 Αποθήκευση ονομάτων/χρωμάτων':
        ('💾 Save names/colours', '💾 Namen/Farben speichern'),
    'Αποτυχία.':         ('Failed.', 'Fehlgeschlagen.'),
    '🖱️ Κλικ στον χάρτη επιλέγει δωμάτιο (αντί να στέλνει το ρομπότ εκεί)':
        ('🖱️ Click on the map picks a room (instead of sending the robot there)',
         '🖱️ Klick auf die Karte wählt einen Raum (statt den Roboter dorthin zu schicken)'),
    'Δεν βρέθηκε δωμάτιο εκεί.':
        ('No room found there.', 'Dort wurde kein Raum gefunden.'),
    '➕ Κλικ στον χάρτη ΠΡΟΣΘΕΤΕΙ δωμάτιο εδώ, με το όνομα/χρώμα από κάτω':
        ('➕ Click on the map ADDS a room here, with the name/colour below',
         '➕ Klick auf die Karte FÜGT hier einen Raum hinzu, mit Name/Farbe unten'),
    'π.χ. κρεβατοκάμαρα': ('e.g. bedroom', 'z.B. Schlafzimmer'),
    'Δώσε πρώτα όνομα δωματίου.':
        ('Give the room a name first.', 'Gib dem Raum zuerst einen Namen.'),
    'Τοποθέτηση…':        ('Placing…', 'Platziere…'),

    # ── keepout zones (map pane) ────────────────────────────────────────────
    '🚫 2 κλικ στον χάρτη ορίζουν απαγορευμένη ζώνη (απέναντι γωνίες)':
        ('🚫 2 clicks on the map define a keepout zone (opposite corners)',
         '🚫 2 Klicks auf die Karte definieren eine Sperrzone (gegenüberliegende Ecken)'),
    'π.χ. χαλάκι σκύλου': ('e.g. dog mat', 'z.B. Hundematte'),
    '🚫 Ενεργοποίηση ζωνών': ('🚫 Enable zones', '🚫 Zonen aktivieren'),
    '✅ Απενεργοποίηση':  ('✅ Disable', '✅ Deaktivieren'),
    'Χρειάζεται επανεκκίνηση (~90s) για να πιάσει η αλλαγή — το Nav2 διαβάζει '
    'τις ζώνες μόνο στην εκκίνηση.':
        ('Needs a restart (~90s) to take effect — Nav2 only reads the zones '
         'at startup.',
         'Braucht einen Neustart (~90s), damit die Änderung wirkt — Nav2 '
         'liest die Zonen nur beim Start.'),
    'Προσθήκη…':          ('Adding…', 'Hinzufügen…'),
    'Διαγραφή…':          ('Deleting…', 'Lösche…'),
    'Θα επανεκκινήσει όλη τη στοίβα (~90 δευτερόλεπτα) με τις απαγορευμένες '
    'ζώνες ενεργές. Να συνεχίσω;':
        ('This will restart the whole stack (~90 seconds) with keepout zones '
         'enabled. Continue?',
         'Das startet den gesamten Stack neu (~90 Sekunden) mit aktivierten '
         'Sperrzonen. Fortfahren?'),
    'Ενεργοποίηση…':      ('Enabling…', 'Aktiviere…'),
    'Θα επανεκκινήσει όλη τη στοίβα (~90 δευτερόλεπτα) χωρίς τις απαγορευμένες '
    'ζώνες. Να συνεχίσω;':
        ('This will restart the whole stack (~90 seconds) without keepout '
         'zones. Continue?',
         'Das startet den gesamten Stack neu (~90 Sekunden) ohne Sperrzonen. '
         'Fortfahren?'),
    'Απενεργοποίηση…':    ('Disabling…', 'Deaktiviere…'),

    # ── 3D arm view ───────────────────────────────────────────────────────
    'Σύρε για περιστροφή · ροδέλα για ζουμ · κινείται με τις αρθρώσεις από κάτω':
        ('Drag to rotate · wheel to zoom · follows the joints below',
         'Ziehen zum Drehen · Rad zum Zoomen · folgt den Gelenken unten'),
    'φόρτωση…':          ('loading…', 'lädt…'),
    'τρίγωνα':           ('triangles', 'Dreiecke'),
    'δεν φορτώθηκε':     ('failed to load', 'nicht geladen'),

    # ── 3D floorplan (walls extruded from the 2D map) ──────────────────────
    'Τοίχοι 3D':         ('Walls 3D', '3D-Wände'),
    'τοίχοι':            ('walls', 'Wände'),
    'Σύρε για περιστροφή · ροδέλα για ζουμ · από τον καθαρό (τετραγωνισμένο) 2D χάρτη':
        ('Drag to rotate · wheel to zoom · from the clean (squared) 2D map',
         'Ziehen zum Drehen · Rad zum Zoomen · aus der bereinigten (quadrierten) 2D-Karte'),

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

    # ── how much Gemini is left today ─────────────────────────────────────
    # Sentence fragments, because the numbers are formatted in the browser's
    # locale and have to sit between them.
    'Ερωτήσεις σήμερα':  ('Questions today', 'Fragen heute'),
    ' από ':             (' of ', ' von '),
    'τέλος για σήμερα':  ('none left today', 'heute aufgebraucht'),
    ' ώρες':             (' hours', ' Stunden'),
    ' λεπτά':            (' minutes', ' Minuten'),
    'περίπου ':          ('about ', 'etwa '),
    'Έχουν φύγει ':      ('Used ', 'Verbraucht: '),
    ' ακόμη. Μηδενίζει σε ': (' left. Resets in ', ' übrig. Zurücksetzung in '),
    '. Μία εντολή που κινεί το ρομπότ πιάνει δύο.':
        ('. A command that moves the robot costs two.',
         '. Ein Befehl, der den Roboter bewegt, kostet zwei.'),
    'Το ημερήσιο όριο εξαντλήθηκε. Ξεκλειδώνει σε ':
        ('The daily limit is used up. It unlocks in ',
         'Das Tageslimit ist aufgebraucht. Es öffnet wieder in '),
    ' — ως τότε γύρνα στο Qwen (NPU).':
        (' — until then switch to Qwen (NPU).',
         ' — bis dahin auf Qwen (NPU) wechseln.'),

    # ── 3D pane ───────────────────────────────────────────────────────────
    '3D κάμερα (D435)':  ('3D camera (D435)', '3D-Kamera (D435)'),
    'Επαναφορά όψης':    ('Reset view', 'Ansicht zurücksetzen'),
    'μικρότερο':         ('smaller', 'kleiner'),
    'μεγαλύτερο':        ('bigger', 'größer'),
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

    # ── settings: turn toward the speaker ─────────────────────────────────
    'Αυτόνομη κίνηση':   ('Autonomous motion', 'Autonome Bewegung'),
    'Να στρίβει προς όποιον μιλάει (μετά το «Έι ρομπότ»)':
        ('Turn toward whoever is speaking (after “Hey robot”)',
         'Zu der Person drehen, die spricht (nach „Hey Roboter“)'),
    'ΕΝΕΡΓΗ':            ('ON', 'AN'),
    'κλειστή':           ('off', 'aus'),
    'Στάλθηκε…':         ('Sent…', 'Gesendet…'),
    '‼️ Κλειστό από προεπιλογή. Ανοιχτό, το ρομπότ γυρίζει τη βάση του μόλις '
    'ακούσει το wake word — ΠΡΙΝ πει κανείς εντολή και χωρίς να ρωτήσει. '
    'Επειδή το «Έι ρομπότ» πιάνεται και μέσα από κουβέντες που δεν του '
    'απευθύνονται, αυτό φαινόταν σαν να «φεύγει» μόνο του.':
        ('‼️ Off by default. When on, the robot turns its base the moment it '
         'hears the wake word — BEFORE anyone gives a command and without '
         'asking. Since “Hey robot” is also picked up from conversations that '
         'are not addressed to it, this looked like the robot wandering off on '
         'its own.',
         '‼️ Standardmäßig aus. Eingeschaltet dreht der Roboter seine Basis, '
         'sobald er das Weckwort hört — BEVOR jemand einen Befehl gibt und ohne '
         'zu fragen. Da „Hey Roboter“ auch aus Gesprächen aufgeschnappt wird, '
         'die ihm gar nicht gelten, wirkte das, als würde er von selbst '
         'davonfahren.'),
    'Ο φωτεινός δακτύλιος δείχνει ΠΑΝΤΑ την κατεύθυνση της φωνής, ακόμη κι '
    'όταν ο διακόπτης είναι κλειστός. Η ρύθμιση θυμάται μετά από':
        ('The LED ring ALWAYS shows the direction of the voice, even with the '
         'switch off. The setting is remembered after',
         'Der LED-Ring zeigt IMMER die Richtung der Stimme, auch bei '
         'ausgeschaltetem Schalter. Die Einstellung bleibt erhalten nach'),

    # ── settings: language ────────────────────────────────────────────────
    'Γλώσσα':            ('Language', 'Sprache'),
    'Αλλάζει μόνο αυτή τη σελίδα. Το ρομπότ συνεχίζει να μιλά ελληνικά.':
        ('Changes this page only. The robot keeps speaking Greek.',
         'Ändert nur diese Seite. Der Roboter spricht weiterhin Griechisch.'),

    # ── settings: maps ────────────────────────────────────────────────────
    'Χάρτες':            ('Maps', 'Karten'),
    'Φόρτωση…':          ('Loading…', 'Lade…'),
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
    'Ξεκινά ΝΕΑ χαρτογράφηση (SLAM). Ο τρέχων χάρτης δεν χάνεται, αλλά ΟΛΑ '
    'ξαναξεκινούν (~90 δευτερόλεπτα) και η πλοήγηση σταματά μέχρι να '
    'αποθηκεύσεις τον καινούργιο. Να συνεχίσω;':
        ('A NEW mapping run (SLAM) will start. The current map is not lost, '
         'but EVERYTHING restarts (~90 seconds) and navigation stops until '
         'you save the new one. Continue?',
         'Eine NEUE Kartierung (SLAM) beginnt. Die aktuelle Karte geht nicht '
         'verloren, aber ALLES startet neu (~90 Sekunden) und die Navigation '
         'stoppt, bis du die neue speicherst. Fortfahren?'),
    'Επανεκκίνηση σε χαρτογράφηση… η σελίδα θα ξανασυνδεθεί μόνη της. '
    'Μετά οδήγησε το ρομπότ σε όλο τον χώρο και αποθήκευσε.':
        ('Restarting into mapping… the page will reconnect on its own. '
         'Then drive the robot around the whole space and save.',
         'Neustart in die Kartierung… die Seite verbindet sich von selbst '
         'wieder. Fahre dann den Roboter durch den ganzen Raum und '
         'speichere.'),
    'Δώσε όνομα με λατινικά γράμματα, αριθμούς, - ή _':
        ('Use a name with Latin letters, digits, - or _',
         'Verwende einen Namen aus lateinischen Buchstaben, Ziffern, - oder _'),
    'Αποθήκευση…':       ('Saving…', 'Speichere…'),
    'Αποθηκεύτηκε':      ('Saved', 'Gespeichert'),
    'Απέτυχε':           ('Failed', 'Fehlgeschlagen'),
    '🧹 Καθαρή εκδοχή — ισιώνει τα σκαλοπάτια των τοίχων.':
        ('🧹 Clean version — straightens the wall staircasing.',
         '🧹 Saubere Version — begradigt die Treppenstufen der Wände.'),
    'Μόνο εμφάνιση':     ('Display only', 'Nur Anzeige'),
    'μέχρι να διαλέξεις: μπορεί να μετατοπίσει τοίχους λίγα εκατοστά ή να '
    'αφαιρέσει ένα πραγματικό εμπόδιο που έμοιαζε με θόρυβο σάρωσης — σύγκρινε '
    'οπτικά πριν διαλέξεις.':
        ('until you choose: it can shift walls by a few centimetres or remove '
         'a real obstacle that looked like scan noise — compare visually '
         'before choosing.',
         'bis du dich entscheidest: sie kann Wände um ein paar Zentimeter '
         'verschieben oder ein echtes Hindernis entfernen, das wie '
         'Scan-Rauschen aussah — vergleiche visuell, bevor du dich '
         'entscheidest.'),
    'Καθαρή εκδοχή':     ('Clean version', 'Saubere Version'),
    'Πρωτότυπο':         ('Original', 'Original'),
    'Ισιωμένο':          ('Straightened', 'Begradigt'),
    'Κράτησε το πρωτότυπο': ('Keep the original', 'Original behalten'),
    'Χρήση ισιωμένης εκδοχής':
        ('Use the straightened version', 'Begradigte Version verwenden'),
    'Θα αντικαταστήσει τον χάρτη':
        ('This will replace the map', 'Das ersetzt die Karte'),
    'με την ισιωμένη εκδοχή (κρατά αντίγραφο ασφαλείας). Αν αυτός ο '
    'χάρτης είναι ήδη ενεργός, θέλει "Ενεργοποίηση" για να φανεί η '
    'αλλαγή. Να συνεχίσω;':
        ('with the straightened version (a backup copy is kept). If this map '
         'is already active, it needs "Activate" for the change to show. '
         'Continue?',
         'durch die begradigte Version (eine Sicherungskopie wird behalten). '
         'Wenn diese Karte bereits aktiv ist, braucht es "Aktivieren", damit '
         'die Änderung sichtbar wird. Fortfahren?'),
    'Εφαρμογή ισιωμένης εκδοχής…':
        ('Applying the straightened version…', 'Wende begradigte Version an…'),
    'Έγινε':             ('Done', 'Erledigt'),
    'αντίγραφο ασφαλείας': ('backup', 'Sicherungskopie'),

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
    '⛶ Πλήρης οθόνη':    ('⛶ Fullscreen', '⛶ Vollbild'),
    'Έξοδος από πλήρη οθόνη':
        ('Exit fullscreen', 'Vollbild verlassen'),
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

    # ── object memory pane ────────────────────────────────────────────────
    'Αντικείμενα':       ('Objects', 'Objekte'),
    'Πού είναι τα πράγματα': ('Where things are', 'Wo die Dinge sind'),
    'Τίποτα γνωστό ακόμη.': ('Nothing known yet.', 'Noch nichts bekannt.'),
    'αντικείμενα':       ('objects', 'Objekte'),
    'άγνωστο δωμάτιο':   ('unknown room', 'unbekannter Raum'),
    'Το ρομπότ θυμάται πού είδε τελευταία κάθε αντικείμενο, από την κάμερα — '
    'η ίδια μνήμη που χρησιμοποιεί το «φέρε μου το Χ». Ένα αντικείμενο '
    'μπαίνει εδώ μόνο αφού επιβεβαιωθεί σε αρκετές ξεχωριστές παρατηρήσεις, '
    'όχι από μία ματιά.':
        ('The robot remembers where it last saw each object, from the camera '
         '— the same memory "bring me the X" uses. An object only appears '
         'here once confirmed across several separate observations, not from '
         'a single glance.',
         'Der Roboter merkt sich, wo er jedes Objekt zuletzt gesehen hat, '
         'über die Kamera — dasselbe Gedächtnis, das „bring mir das X" nutzt. '
         'Ein Objekt erscheint hier erst, nachdem es in mehreren getrennten '
         'Beobachtungen bestätigt wurde, nicht nach einem einzigen Blick.'),

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

    # ── safety tab ────────────────────────────────────────────────────────
    # 'Ασφάλεια' is the tab; the dashboard token's card next door is
    # 'Κλειδί πρόσβασης', because two panels both called "Security" is how a
    # user ends up looking for the stopping distance under the access key.
    'Ασφάλεια':          ('Safety', 'Sicherheit'),
    'Απόσταση από εμπόδια': ('Distance from obstacles', 'Abstand zu Hindernissen'),
    'Πόσο κοντά πλησιάζει πριν σταματήσει':
        ('How close it gets before stopping',
         'Wie nah es herankommt, bevor es stoppt'),
    'Το ρομπότ κόβει την ευθεία κίνηση όταν δει κάτι πιο κοντά από αυτό. '
    'Συνεχίζει να στρίβει και να κάνει όπισθεν, ώστε να μπορεί να ξεφύγει.':
        ('The robot cuts forward motion when it sees something closer than '
         'this. It keeps turning and reversing, so it can still get away.',
         'Der Roboter stoppt die Vorwärtsfahrt, wenn er etwas Näheres sieht. '
         'Drehen und Rückwärtsfahren bleiben möglich, damit er wegkommt.'),
    'Πόσο φαρδιά κοιτάει μπροστά του':
        ('How wide "ahead" is', 'Wie breit „voraus“ ist'),
    'Ποιο κομμάτι της εικόνας μετράει ως «μπροστά». Στο 1.00 τον σταματά '
    'ο,τιδήποτε φαίνεται στην άκρη του κάδρου — σε διάδρομο αυτό σημαίνει ότι '
    'δεν ξεκινά ποτέ.':
        ('Which part of the image counts as "ahead". At 1.00 anything at the '
         'edge of the frame stops it — in a corridor that means it never '
         'starts at all.',
         'Welcher Teil des Bildes als „voraus“ zählt. Bei 1.00 stoppt ihn '
         'alles am Bildrand — in einem Flur heißt das: er fährt nie los.'),
    'Σε πόση σιωπή της κάμερας φρενάρει':
        ('How long a silent camera is tolerated',
         'Wie lange eine stumme Kamera toleriert wird'),
    'Αν η ανίχνευση σταματήσει (κόλλησε η κάμερα, τερμάτισε ο detector), η '
    'ευθεία κίνηση μπλοκάρεται μετά από τόση ώρα αντί να θεωρηθεί ο δρόμος '
    'καθαρός.':
        ('If detection stops (the camera hung, the detector exited), forward '
         'motion is blocked after this long instead of the path being assumed '
         'clear.',
         'Wenn die Erkennung ausfällt (Kamera hängt, Detector beendet), wird '
         'die Vorwärtsfahrt nach dieser Zeit gesperrt, statt den Weg als frei '
         'anzunehmen.'),
    'Απόσταση από τοίχους': ('Distance from walls', 'Abstand zu Wänden'),
    'Πόσο μακριά από τοίχους σχεδιάζει τη διαδρομή':
        ('How far from walls it plans its path',
         'Wie weit von Wänden es seinen Weg plant'),
    'Ζώνη γύρω από κάθε εμπόδιο που ο planner αποφεύγει. Μεγαλύτερη = περνά '
    'πιο κεντραρισμένο.':
        ('A zone around every obstacle that the planner avoids. Larger = it '
         'passes more centred.',
         'Eine Zone um jedes Hindernis, die der Planer meidet. Größer = er '
         'fährt mittiger.'),
    '‼️ Οι πόρτες εδώ είναι ~0.78 m. Πάνω από 0.30 m οι ζώνες των δύο '
    'παραστάδων ενώνονται και το ρομπότ αρνείται να περάσει από άνοιγμα που '
    'στον χάρτη φαίνεται καθαρό.':
        ('‼️ The doorways here are ~0.78 m. Above 0.30 m the zones of the two '
         'jambs meet and the robot refuses an opening that looks clear on the '
         'map.',
         '‼️ Die Türen hier sind ~0,78 m. Über 0,30 m treffen sich die Zonen '
         'beider Pfosten und der Roboter verweigert eine Öffnung, die auf der '
         'Karte frei aussieht.'),
    'Πόσο απότομα χαλαρώνει αυτή η ζώνη':
        ('How sharply that zone falls off',
         'Wie stark diese Zone abfällt'),
    'Μεγαλύτερο = η αποφυγή σβήνει πιο γρήγορα με την απόσταση, άρα δέχεται '
    'να περάσει πιο κοντά στον τοίχο.':
        ('Higher = the avoidance fades faster with distance, so it accepts '
         'passing closer to the wall.',
         'Höher = das Ausweichen klingt mit der Entfernung schneller ab, er '
         'fährt also näher an der Wand vorbei.'),
    'Αισθητήρες επαφής': ('Contact sensors', 'Kontaktsensoren'),
    '‼️ Η βάση τρέχει σε FULL mode: το ίδιο το Roomba ΔΕΝ σταματά μόνο του σε '
    "προφυλακτήρα ή σε σκαλί. Αυτοί οι τρεις διακόπτες είναι το μόνο που το "
    "κάνει. Κλείσ' τους μόνο για χαλασμένο αισθητήρα.":
        ('‼️ The base runs in FULL mode: the Roomba itself does NOT stop for '
         'its bumper or a step. These three switches are the only thing that '
         'does. Turn one off only for a broken sensor.',
         '‼️ Die Basis läuft im FULL-Modus: der Roomba selbst stoppt NICHT bei '
         'Stoßfänger oder Stufe. Diese drei Schalter sind das Einzige, was es '
         'tut. Nur bei defektem Sensor ausschalten.'),
    'Στοπ στον προφυλακτήρα': ('Stop on bumper', 'Stopp bei Stoßfänger'),
    'Στοπ στον γκρεμό (σκαλί)': ('Stop at a drop (step)', 'Stopp an Absturzkante'),
    'Στοπ όταν πέφτει τροχός στο κενό':
        ('Stop when a wheel drops', 'Stopp bei durchhängendem Rad'),
    'Πόσο μένει μπλοκαρισμένο μετά από χτύπημα':
        ('How long it stays blocked after a bump',
         'Wie lange nach einem Stoß blockiert bleibt'),
    'Κρατά την ευθεία κλειστή τόση ώρα αφού ελευθερωθεί ο προφυλακτήρας, ώστε '
    'το ρομπότ να μην ξαναμπεί στο ίδιο έπιπλο.':
        ('Keeps forward blocked this long after the bumper releases, so the '
         'robot does not drive back into the same furniture.',
         'Hält die Vorwärtsfahrt so lange gesperrt, nachdem der Stoßfänger '
         'frei ist, damit er nicht gleich wieder ins selbe Möbel fährt.'),
    'Πόσο μένει μπλοκαρισμένο μετά από γκρεμό':
        ('How long it stays blocked after a drop',
         'Wie lange nach einer Kante blockiert bleibt'),
    'Σκληρά όρια': ('Hard limits', 'Harte Grenzen'),
    'σταθερά': ('fixed', 'fest'),
    'ΔΕΝ αλλάζει από εδώ. Ο collision_monitor το διαβάζει μία φορά στην '
    'εκκίνηση, οπότε ένας διακόπτης εδώ θα έδειχνε νούμερο που το ρομπότ δεν '
    'χρησιμοποιεί. Αλλάζει στο config/nav2_params.yaml και θέλει πλήρη '
    'επανεκκίνηση.':
        ('Does NOT change from here. collision_monitor reads it once at '
         'startup, so a control here would show a number the robot is not '
         'using. It changes in config/nav2_params.yaml and needs a full '
         'restart.',
         'Ändert sich hier NICHT. collision_monitor liest ihn einmal beim '
         'Start, ein Regler hier würde also einen Wert zeigen, den der '
         'Roboter nicht benutzt. Änderung in config/nav2_params.yaml, mit '
         'vollständigem Neustart.'),
    '↺ Επαναφορά προεπιλογών': ('↺ Restore defaults', '↺ Standardwerte'),
    'Επαναφορά όλων των ρυθμίσεων ασφαλείας στις προεπιλογές;':
        ('Restore all safety settings to their defaults?',
         'Alle Sicherheitseinstellungen auf Standard zurücksetzen?'),
    'Επαναφέρθηκαν.': ('Restored.', 'Zurückgesetzt.'),
    'ΕΝΕΡΓΟ':            ('ON', 'AN'),
    'ΚΛΕΙΣΤΟ':           ('OFF', 'AUS'),
    'δεν τρέχει':        ('not running', 'läuft nicht'),

    # ── collision_monitor hard-stop skirt (map tab → Ασφάλεια) ─────────────
    '🛑 Απόσταση πλήρους στάσης':
        ('🛑 Full-stop distance', '🛑 Vollstopp-Abstand'),
    'Το τελευταίο όριο: μόλις το lidar δει κάτι μέσα σε αυτή την απόσταση '
    'γύρω από το σώμα, ο collision_monitor μηδενίζει αμέσως την ταχύτητα — '
    'ανεξάρτητα από το τι σχεδιάζει ο planner. Δεν είναι η «Απόσταση από '
    'τοίχους» πιο πάνω: εκείνο επηρεάζει πού περνά η διαδρομή, αυτό είναι το '
    'φρένο ανάγκης. Μικρότερη απόσταση αφήνει το ρομπότ να πλησιάσει '
    'περισσότερο πριν σταματήσει απότομα.':
        ('The last-resort limit: the instant the lidar sees anything inside '
         'this distance around the body, collision_monitor zeroes the speed '
         'immediately — regardless of what the planner is doing. This is NOT '
         'the "Distance from walls" above: that decides where the path goes, '
         'this is the emergency brake. A smaller distance lets the robot get '
         'closer before it stops abruptly.',
         'Die letzte Grenze: sobald der Lidar etwas innerhalb dieses Abstands '
         'um den Körper sieht, setzt collision_monitor die Geschwindigkeit '
         'sofort auf null — unabhängig davon, was der Planer vorhat. Das ist '
         'NICHT der "Abstand von Wänden" oben: der entscheidet, wo die Route '
         'verläuft, das hier ist die Notbremse. Ein kleinerer Abstand lässt '
         'den Roboter näher herankommen, bevor er abrupt stoppt.'),
    'Εφαρμογή (επανεκκίνηση ~90s)':
        ('Apply (restart ~90s)', 'Anwenden (Neustart ~90s)'),
    'Αυτό είναι το σκληρό όριο πλήρους στάσης, όχι η απόσταση σχεδίασης '
    'διαδρομής. Χρειάζεται πλήρη επανεκκίνηση (~90 δευτερόλεπτα) και '
    'μικρότερη απόσταση αφήνει το ρομπότ να πλησιάσει περισσότερο πριν '
    'σταματήσει απότομα. Να συνεχίσω;':
        ('This is the hard full-stop limit, not the path-planning distance. '
         'It needs a full restart (~90 seconds), and a smaller distance lets '
         'the robot get closer before it stops abruptly. Continue?',
         'Das ist die harte Vollstopp-Grenze, nicht der Abstand für die '
         'Routenplanung. Es braucht einen vollständigen Neustart (~90 '
         'Sekunden), und ein kleinerer Abstand lässt den Roboter näher '
         'herankommen, bevor er abrupt stoppt. Fortfahren?'),

    # ── dashboard key ─────────────────────────────────────────────────────
    'Κλειδί πρόσβασης':  ('Access key', 'Zugangsschlüssel'),
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

    # ── Camera "why it looks empty" banner ────────────────────────────────
    # The three reasons come from the SERVER (_judge_frame), so they never
    # appear in the markup the i18n test scrapes — they have to be listed by
    # hand or they would only ever render in Greek.
    'Η κάμερα ΔΟΥΛΕΥΕΙ': ('The camera IS working', 'Die Kamera FUNKTIONIERT'),
    'σκοτάδι — κοιτάει σε σκοτεινό χώρο ή είναι καλυμμένη':
        ('darkness — it is looking into an unlit space, or something is over it',
         'Dunkelheit — sie schaut in einen unbeleuchteten Raum oder ist verdeckt'),
    'κατάλευκο — κοιτάει κατευθείαν σε φως ή σε τοίχο από πολύ κοντά':
        ('pure white — it is looking straight into a light, or at a wall from '
         'very close up',
         'reinweiß — sie schaut direkt in ein Licht oder aus nächster Nähe auf '
         'eine Wand'),
    'επίπεδη επιφάνεια χωρίς λεπτομέρεια — μάλλον τοίχος μπροστά της':
        ('a flat surface with no detail — most likely a wall in front of it',
         'eine flache Fläche ohne Details — höchstwahrscheinlich eine Wand '
         'davor'),

    # ── Slip map (map pane) ───────────────────────────────────────────────
    '🩹 Πού γλιστράει': ('🩹 Where it slips', '🩹 Wo es rutscht'),
    'Δείχνει τα σημεία του σπιτιού όπου το AMCL χρειάστηκε να διορθώσει '
    'περισσότερο τη θέση — δηλαδή εκεί που η οδομετρία χάνει έδαφος. '
    'Μαζεύεται με τον καιρό και επιβιώνει σε restart. Ένα φωτεινό σημείο σε '
    'ένα χαλί είναι πρόβλημα που φτιάχνεται· παντού λίγο, είναι βαθμονόμηση.':
        ('Shows the places in the house where AMCL had to correct the position '
         'most — where the odometry loses ground. It builds up over time and '
         'survives restarts. One bright spot on a rug is a fixable problem; a '
         'little everywhere is calibration.',
         'Zeigt die Stellen im Haus, an denen AMCL die Position am meisten '
         'korrigieren musste — wo die Odometrie Boden verliert. Es sammelt '
         'sich über die Zeit und überlebt Neustarts. Ein heller Fleck auf '
         'einem Teppich ist ein behebbares Problem; überall ein wenig ist '
         'Kalibrierung.'),
    'καθαρά': ('clear', 'sauber'),
    'Χειρότερο σημείο': ('Worst spot', 'Schlimmste Stelle'),
    'συνολικής διόρθωσης σε': ('of correction in total, across',
                               'Korrektur insgesamt, über'),
    'περάσματα': ('passes', 'Durchgänge'),

    # ── CPU frequency band for the power station (System pane) ────────────
    'Ομαλό ρεύμα': ('Steady power draw', 'Gleichmäßige Leistung'),
    '🔌 Πρίζα (κανονικό)': ('🔌 Wall power (normal)', '🔌 Steckdose (normal)'),
    '🍃 Ήπιο': ('🍃 Gentle', '🍃 Sanft'),
    '📏 Πολύ σταθερό': ('📏 Very steady', '📏 Sehr gleichmäßig'),
    'κανονικό': ('normal', 'normal'),
    'ήπιο': ('gentle', 'sanft'),
    'πολύ σταθερό': ('very steady', 'sehr gleichmäßig'),
    'Για όταν το mini PC τρέχει από την powerstation (Anker SOLIX C300 DC). '
    'Αυτή δεν έχει inverter — τροφοδοτεί το PC μέσω USB-C Power Delivery. Μια '
    'πηγή PD δεν ενοχλείται από το πόσο ρεύμα τραβάς, αλλά από το πόσο απότομα '
    'αλλάζει: ένα σκαλοπάτι φορτίου ρίχνει την προστασία της θύρας ή προκαλεί '
    'επαναδιαπραγμάτευση PD, και η επαναδιαπραγμάτευση κόβει τη γραμμή αρκετά '
    'ώστε το PC να σβήσει ακαριαία.':
        ('For when the mini PC runs off the power station (Anker SOLIX C300 '
         'DC). That one has no inverter — it feeds the PC over USB-C Power '
         'Delivery. A PD source does not mind how much current you draw, it '
         'minds how abruptly that changes: a step in load trips the port\'s '
         'protection or forces a PD renegotiation, and a renegotiation drops '
         'the rail for long enough to kill the PC outright.',
         'Für den Betrieb des Mini-PCs an der Powerstation (Anker SOLIX C300 '
         'DC). Diese hat keinen Wechselrichter — sie versorgt den PC über '
         'USB-C Power Delivery. Eine PD-Quelle stört nicht, wie viel Strom du '
         'ziehst, sondern wie abrupt sich das ändert: eine Laststufe löst den '
         'Schutz des Ports aus oder erzwingt eine PD-Neuverhandlung, und diese '
         'unterbricht die Versorgung lange genug, um den PC sofort '
         'abzuschalten.'),
    'Τα προφίλ πειράζουν τέσσερα πράγματα, όλα για να μειώσουν το άλμα: '
    'στενεύουν τη ζώνη συχνότητας της CPU, κλείνουν το boost, βάζουν τον '
    'επεξεργαστή σε λειτουργία εξοικονόμησης, και — μόνο στο «πολύ σταθερό» — '
    'κόβουν το ψηλότερο σκαλί της iGPU. Γι\' αυτό το «πολύ σταθερό» κοστίζει '
    'σε ταχύτητα γραφικών· το «ήπιο» τα αφήνει ήσυχα.':
        ('The profiles change four things, all of them to shrink the step: '
         'they narrow the CPU frequency band, turn boost off, put the '
         'processor in a power-saving mode, and — only in "very steady" — cut '
         'the iGPU\'s top step. That is why "very steady" costs graphics '
         'speed, while "gentle" leaves graphics alone.',
         'Die Profile ändern vier Dinge, alle um den Sprung zu verkleinern: '
         'sie verengen das Frequenzband der CPU, schalten Boost ab, setzen den '
         'Prozessor in einen Sparmodus und — nur bei „sehr gleichmäßig" — '
         'kappen die oberste Stufe der iGPU. Deshalb kostet „sehr '
         'gleichmäßig" Grafikleistung, „sanft" lässt die Grafik in Ruhe.'),
    'Μετρημένο εδώ σε πραγματικά Watt (αισθητήρας PPT του επεξεργαστή), με '
    'ριπές φορτίου σε όλους τους πυρήνες:':
        ('Measured here in real watts (the processor\'s own PPT sensor), under '
         'bursts of all-core load:',
         'Hier in echten Watt gemessen (PPT-Sensor des Prozessors), unter '
         'Lastspitzen auf allen Kernen:'),
    '21.5 W μέσος όρος και διακύμανση 30.0 W, από 7 ως 37·':
        ('21.5 W average and a 30.0 W swing, from 7 to 37;',
         '21,5 W Mittel und 30,0 W Schwankung, von 7 bis 37;'),
    '11.9 και 12.1·': ('11.9 and 12.1;', '11,9 und 12,1;'),
    '10.5 και 11.1. Η διακύμανση πέφτει κατά 63% και η μέση κατανάλωση στο '
    'μισό — εδώ δεν πληρώνεις τίποτα για τη σταθερότητα.':
        ('10.5 and 11.1. The swing drops by 63% and the average draw halves — '
         'here steadiness costs you nothing.',
         '10,5 und 11,1. Die Schwankung sinkt um 63% und der '
         'Durchschnittsverbrauch halbiert sich — Gleichmäßigkeit kostet hier '
         'nichts.'),
    '✅ Επιβιώνει σε restart: η επιλογή αποθηκεύεται και ξαναμπαίνει μόνη της '
    'σε κάθε boot. Στην πρίζα γύρνα το σε «κανονικό» — δεν χρειάζεται.':
        ('✅ It survives a restart: the choice is saved and reapplies itself at '
         'every boot. On wall power switch it back to "normal" — it is not '
         'needed there.',
         '✅ Übersteht einen Neustart: die Auswahl wird gespeichert und bei '
         'jedem Boot erneut angewendet. Am Netz wieder auf „normal" stellen — '
         'dort wird es nicht gebraucht.'),

    # ── USB power cycle (System pane) ─────────────────────────────────────
    'Ξεκόλλα αισθητήρα': ('Unstick a sensor', 'Sensor entklemmen'),
    'Κόβει το ρεύμα στη θύρα USB της συσκευής για δύο δευτερόλεπτα και το '
    'ξαναδίνει — ό,τι ακριβώς κάνει το να την βγάλεις και να την ξαναβάλεις, '
    'χωρίς να σηκωθεί κανείς. Για αισθητήρα που κόλλησε και δεν ξεκολλάει με '
    'restart του κόμβου.':
        ('Cuts power to the device\'s USB port for two seconds and gives it '
         'back — exactly what unplugging and replugging it does, without '
         'anyone getting up. For a sensor that has wedged and does not come '
         'back when its node restarts.',
         'Unterbricht zwei Sekunden lang den Strom am USB-Port des Geräts und '
         'gibt ihn zurück — genau das, was Aus- und Einstecken bewirkt, ohne '
         'dass jemand aufstehen muss. Für einen Sensor, der hängt und beim '
         'Neustart seines Knotens nicht zurückkommt.'),
    '‼️ Ο ΟΔΗΓΟΣ ΔΕΝ ΕΠΑΝΕΚΚΙΝΕΙ ΜΟΝΟΣ ΤΟΥ. Η συσκευή θα ξαναεμφανιστεί στο '
    'σύστημα, αλλά ο κόμβος που την είχε ανοιχτή κρατά πεθαμένο handle — '
    'θέλει και δικό του restart. Η κάμερα είναι η εξαίρεση: ο camera_watchdog '
    'την ξαναπιάνει μόνος του.':
        ('‼️ THE DRIVER DOES NOT RESTART ITSELF. The device will come back to '
         'the system, but the node that had it open still holds a dead handle '
         '— it needs restarting too. The camera is the exception: '
         'camera_watchdog picks it up on its own.',
         '‼️ DER TREIBER STARTET SICH NICHT SELBST NEU. Das Gerät kommt im '
         'System zurück, aber der Knoten, der es offen hatte, hält weiter ein '
         'totes Handle — er braucht ebenfalls einen Neustart. Die Kamera ist '
         'die Ausnahme: camera_watchdog fängt sie selbst wieder ein.'),
    'Η «σκούπα» κόβει το ρεύμα στη σειριακή της βάσης. Αν το ρομπότ κινείται, '
    'οι τροχοί συνεχίζουν με την τελευταία εντολή ώσπου να πιάσει το watchdog '
    'των 0.25 δευτερολέπτων — μην το πατήσεις εν κινήσει.':
        ('"Vacuum" cuts power to the base\'s serial link. If the robot is '
         'moving, the wheels keep running on the last command until the '
         '0.25 s watchdog catches it — do not press this while it drives.',
         '„Sauger" unterbricht den Strom der seriellen Verbindung zur Basis. '
         'Fährt der Roboter, laufen die Räder mit dem letzten Befehl weiter, '
         'bis der 0,25-s-Watchdog greift — nicht während der Fahrt drücken.'),
    'Θα κοπεί η σειριακή της βάσης. Αν το ρομπότ κινείται, οι τροχοί '
    'συνεχίζουν ώσπου να πιάσει το watchdog. Να συνεχίσω;':
        ('The base\'s serial link will be cut. If the robot is moving, the '
         'wheels keep running until the watchdog catches it. Continue?',
         'Die serielle Verbindung zur Basis wird unterbrochen. Fährt der '
         'Roboter, laufen die Räder weiter, bis der Watchdog greift. '
         'Fortfahren?'),
    'σε εξέλιξη': ('in progress', 'läuft'),
    'κόβω το ρεύμα…': ('cutting power…', 'unterbreche den Strom…'),
    'έγινε': ('done', 'erledigt'),
    'ξαναήρθε. Ο κόμβος που την είχε ανοιχτή θέλει restart.':
        ('is back. The node that had it open needs restarting.',
         'ist zurück. Der Knoten, der es offen hatte, braucht einen Neustart.'),

    # ── Sensor fusion pane ────────────────────────────────────────────────
    'Ποιος λέει τι':     ('Who says what', 'Wer sagt was'),
    'EKF (αποτέλεσμα)':  ('EKF (the result)', 'EKF (Ergebnis)'),
    '60 δευτερόλεπτα':   ('60 seconds', '60 Sekunden'),
    'Τροχοί − EKF':      ('Wheels − EKF', 'Räder − EKF'),
    '⟲ Μηδένισε τη σύγκριση': ('⟲ Re-zero the comparison',
                               '⟲ Vergleich nullen'),
    "Οι τρεις γωνίες ΔΕΝ έχουν κοινό μηδέν — το BNO085 ξεκινά από όπου "
    "κοιτούσε στο boot, οι τροχοί από όπου άναψε η βάση. Το κουμπί τις "
    "ευθυγραμμίζει εδώ και τώρα, οπότε ό,τι ανοίγει μετά είναι πραγματική "
    "απόκλιση. Οδήγησε ένα γύρο και γύρνα στο ίδιο σημείο: η γραμμή που δεν "
    "επιστρέφει στο μηδέν είναι ο αισθητήρας που λέει ψέματα. Οι τροχοί "
    "ανοίγουν πάντα σε χαλί και σε στροφές — γι' αυτό ο EKF παίρνει γωνία "
    "μόνο από το IMU (config/ekf.yaml).":
        ('The three headings share NO common zero — the BNO085 starts from '
         'wherever it was facing at boot, the wheels from wherever the base '
         'powered up. The button aligns them here and now, so anything that '
         'opens up afterwards is real drift. Drive a loop and come back to the '
         'same spot: the line that does not return to zero is the sensor that '
         'is lying. The wheels always drift on carpet and in turns — which is '
         'why the EKF takes heading from the IMU alone (config/ekf.yaml).',
         'Die drei Winkel haben KEINEN gemeinsamen Nullpunkt — der BNO085 '
         'startet dort, wohin er beim Booten schaute, die Räder dort, wo die '
         'Basis eingeschaltet wurde. Der Knopf richtet sie hier und jetzt aus, '
         'alles danach ist echte Abweichung. Fahre eine Runde und komm zum '
         'selben Punkt zurück: die Linie, die nicht auf null zurückkehrt, ist '
         'der Sensor, der lügt. Die Räder driften auf Teppich und in Kurven '
         'immer — deshalb nimmt der EKF den Winkel nur vom IMU '
         '(config/ekf.yaml).'),

    'Πόσο διορθώνει το AMCL': ('How much AMCL is correcting',
                               'Wie viel AMCL korrigiert'),
    'Μετατόπιση από το μηδένισμα': ('Shift since the re-zero',
                                    'Verschiebung seit dem Nullen'),
    'Γωνία από το μηδένισμα': ('Angle since the re-zero',
                               'Winkel seit dem Nullen'),
    'Μέγιστο': ('Peak', 'Maximum'),
    'Πόσο χρειάστηκε να ΞΑΝΑΒΑΛΕΙ το AMCL το ρομπότ πάνω στους τοίχους από '
    'τη στιγμή που άνοιξες την καρτέλα — δηλαδή το λάθος που μάζεψε η '
    'σύντηξη μόνη της, όσο κοιτούσες. Στον χάρτη δεν φαίνεται ποτέ: εκεί το '
    'ρομπότ δείχνει πάντα σωστά τοποθετημένο, επειδή ακριβώς αυτή η διόρθωση '
    'το κρατά εκεί. Λίγα εκατοστά ανά διαδρομή είναι φυσιολογικά· δεκάδες '
    'σημαίνουν ότι η οδομετρία γλιστράει και το AMCL μπαλώνει.':
        ('How far AMCL had to put the robot back onto the walls since you '
         'opened this tab — the error the fusion accumulated on its own while '
         'you watched. It never shows on the map: there the robot always looks '
         'correctly placed, precisely because this correction keeps it there. '
         'A few centimetres per trip is normal; tens of them mean the odometry '
         'is slipping and AMCL is papering over it.',
         'Wie weit AMCL den Roboter seit dem Öffnen dieses Tabs wieder auf die '
         'Wände setzen musste — der Fehler, den die Fusion allein angesammelt '
         'hat, während du zusahst. Auf der Karte sieht man das nie: dort steht '
         'der Roboter immer richtig, genau weil diese Korrektur ihn dort hält. '
         'Wenige Zentimeter pro Fahrt sind normal; Dutzende heißen, die '
         'Odometrie rutscht und AMCL überdeckt es.'),
    '‼️ ΔΙΑΦΟΡΑ, όχι απόλυτη τιμή. Το ωμό':
        ('‼️ A DIFFERENCE, not an absolute value. Raw',
         '‼️ Eine DIFFERENZ, kein Absolutwert. Das rohe'),
    'περιέχει και το πού έτυχε να είναι η αρχή του':
        ('also carries wherever the origin of', 'enthält auch, wo der Ursprung von'),
    'στο boot: μετρήθηκε 366 cm και 123° σε ρομπότ που εντόπιζε τέλεια — '
    'απλώς είχε οδηγήσει από τότε που άναψε. Το κουμπί «Μηδένισε» ξαναπιάνει '
    'το σημείο αναφοράς.':
        ('happened to be at boot: 366 cm and 123° were measured on a robot '
         'that was localizing perfectly — it had simply driven since it was '
         'switched on. The re-zero button retakes the reference.',
         'beim Booten zufällig lag: 366 cm und 123° wurden an einem Roboter '
         'gemessen, der perfekt lokalisierte — er war seit dem Einschalten '
         'einfach gefahren. Der Null-Knopf setzt den Bezugspunkt neu.'),

    'Ταχύτητες': ('Speeds', 'Geschwindigkeiten'),
    'Μπροστά — τροχοί': ('Forward — wheels', 'Vorwärts — Räder'),
    'Μπροστά — EKF': ('Forward — EKF', 'Vorwärts — EKF'),
    'Στροφή — τροχοί': ('Turn — wheels', 'Drehung — Räder'),
    'Στροφή — γυροσκόπιο': ('Turn — gyro', 'Drehung — Gyroskop'),
    'Στροφή — EKF': ('Turn — EKF', 'Drehung — EKF'),
    'Οι τροχοί λένε πόσο ΖΗΤΗΘΗΚΕ να γυρίσει, το γυροσκόπιο πόσο γύρισε '
    'ΠΡΑΓΜΑΤΙΚΑ. Όταν οι δύο αριθμοί διαφέρουν σταθερά, η βάση γλιστράει ή '
    'έχει κολλήσει σε κάτι. Κάτω από ~0.31 rad/s η 879 δεν στρίβει καθόλου: '
    'θα δεις εντολή στροφής στους τροχούς και μηδέν στο γυροσκόπιο, και αυτό '
    'είναι το γνωστό κατώφλι, όχι βλάβη.':
        ('The wheels say how much turn was ASKED for, the gyro how much '
         'actually happened. When the two differ consistently, the base is '
         'slipping or caught on something. Below ~0.31 rad/s the 879 does not '
         'turn at all: you will see a turn at the wheels and zero at the gyro, '
         'and that is the known floor, not a fault.',
         'Die Räder sagen, wie viel Drehung ANGEFORDERT wurde, das Gyroskop, '
         'wie viel wirklich geschah. Weichen beide dauerhaft ab, rutscht die '
         'Basis oder hängt fest. Unter ~0,31 rad/s dreht der 879 gar nicht: '
         'Räder zeigen Drehung, Gyroskop null — das ist die bekannte Schwelle, '
         'kein Fehler.'),

    'Υγεία πηγών': ('Source health', 'Zustand der Quellen'),
    '‼️ Νεκρή πηγή είναι ΣΙΩΠΗΛΗ, όχι λάθος: ο EKF συνεχίζει να δημοσιεύει '
    'την τελευταία καλή γωνία και όλα δείχνουν υγιή. Μόνο η συχνότητα και η '
    "ηλικία δείγματος το δείχνουν — γι' αυτό μετριούνται εδώ χωριστά από τις "
    'τιμές.':
        ('‼️ A dead source is SILENT, not wrong: the EKF keeps publishing its '
         'last good heading and everything looks healthy. Only the rate and '
         'the sample age show it — which is why they are measured here, apart '
         'from the values.',
         '‼️ Eine tote Quelle ist STUMM, nicht falsch: der EKF veröffentlicht '
         'weiter seinen letzten guten Winkel und alles wirkt gesund. Nur Rate '
         'und Alter der Probe zeigen es — deshalb werden sie hier getrennt von '
         'den Werten gemessen.'),
    'Φυσιολογικές τιμές, μετρημένες σε φρέσκο `robot max`: τροχοί ~20 Hz, '
    'IMU ~110 Hz, EKF ~30 Hz. ‼️ Το IMU μετρήθηκε κάποτε στα 6 Hz και πέρασε '
    'για φυσιολογικό — ήταν υποβαθμισμένη ροή που κανείς δεν είχε προσέξει. '
    'Αν το δεις κάτω από 40, δεν είναι «αργό», είναι χαλασμένο.':
        ('Normal values, measured on a fresh `robot max`: wheels ~20 Hz, IMU '
         '~110 Hz, EKF ~30 Hz. ‼️ The IMU once measured 6 Hz and was taken for '
         'normal — it was a degraded stream nobody had noticed. Below 40 it is '
         'not "slow", it is broken.',
         'Normalwerte, gemessen bei frischem `robot max`: Räder ~20 Hz, IMU '
         '~110 Hz, EKF ~30 Hz. ‼️ Das IMU wurde einmal mit 6 Hz gemessen und '
         'galt als normal — es war ein degradierter Stream, den niemand '
         'bemerkt hatte. Unter 40 ist es nicht „langsam", sondern kaputt.'),

    'Αβεβαιότητα': ('Uncertainty', 'Unsicherheit'),
    'EKF γωνία ±': ('EKF heading ±', 'EKF Winkel ±'),
    'AMCL θέση ± / γωνία ±': ('AMCL position ± / heading ±',
                              'AMCL Position ± / Winkel ±'),
    'Πόσο σίγουρος δηλώνει ο καθένας ότι είναι, σε εκατοστά και μοίρες '
    '(τυπική απόκλιση, όχι το ωμό covariance). Το AMCL φουσκώνει όταν χάνει '
    'τον εντοπισμό και ξαναμαζεύει μόλις κλειδώσει σε τοίχους — μια τιμή που '
    'μεγαλώνει και δεν ξαναμαζεύει είναι το «οι κόκκινες γραμμές έφυγαν» '
    'πριν το δεις στον χάρτη.':
        ('How sure each one claims to be, in centimetres and degrees (standard '
         'deviation, not raw covariance). AMCL swells when it loses '
         'localization and shrinks again once it locks onto walls — a value '
         'that grows and never comes back is "the red lines went wrong" before '
         'you can see it on the map.',
         'Wie sicher sich jeder gibt, in Zentimetern und Grad '
         '(Standardabweichung, nicht rohe Kovarianz). AMCL schwillt an, wenn '
         'es die Lokalisierung verliert, und schrumpft wieder, sobald es an '
         'Wänden einrastet — ein Wert, der wächst und nicht zurückkommt, ist '
         '„die roten Linien stimmen nicht", bevor du es auf der Karte siehst.'),
    '‼️ Η ΘΕΣΗ του EKF δεν εμφανίζεται επίτηδες. Ο EKF δουλεύει στο':
        ('‼️ The EKF POSITION is deliberately not shown. The EKF works in',
         '‼️ Die EKF-POSITION wird bewusst nicht gezeigt. Der EKF arbeitet in'),
    ', όπου καμία απόλυτη μέτρηση δεν τον διορθώνει, οπότε η αβεβαιότητα '
    'θέσης του μεγαλώνει για πάντα — μετρήθηκε στα ±2607 km σε ρομπότ που '
    'εντόπιζε μια χαρά. Δεν είναι βλάβη, είναι ο ορισμός του frame. Η γωνία '
    'του παραμένει φραγμένη (το IMU τη διορθώνει), και για τη θέση ο μόνος '
    'αριθμός με νόημα είναι του AMCL.':
        (', where no absolute measurement corrects it, so its position '
         'uncertainty grows for ever — measured at ±2607 km on a robot that '
         'was localizing perfectly well. Not a fault, the definition of the '
         'frame. Its heading stays bounded (the IMU corrects it), and for '
         'position the only meaningful number is AMCL\'s.',
         ', wo keine absolute Messung ihn korrigiert, also wächst seine '
         'Positionsunsicherheit ewig — gemessen bei ±2607 km an einem Roboter, '
         'der einwandfrei lokalisierte. Kein Fehler, sondern die Definition '
         'des Frames. Sein Winkel bleibt begrenzt (das IMU korrigiert ihn), '
         'und für die Position zählt nur die Zahl von AMCL.'),

    'LiDAR εναντίον κάμερας': ('LiDAR versus camera', 'LiDAR gegen Kamera'),
    'Σύγκρινε με το βάθος της D435 (ανάβει το νέφος σημείων)':
        ('Compare against the D435 depth (switches the point cloud on)',
         'Mit der D435-Tiefe vergleichen (schaltet die Punktwolke ein)'),
    'Συμφωνία': ('Agreement', 'Übereinstimmung'),
    'Βλέπει μόνο η κάμερα': ('Camera only', 'Nur die Kamera sieht'),
    'Πλησιέστερο κρυφό εμπόδιο': ('Nearest hidden obstacle',
                                  'Nächstes verstecktes Hindernis'),
    'Κάτοψη γύρω από το ρομπότ, 4 μέτρα ακτίνα, μύτη προς τα πάνω.':
        ('Top-down around the robot, 4 metre radius, nose pointing up.',
         'Draufsicht um den Roboter, 4 Meter Radius, Nase nach oben.'),
    'Λευκό': ('White', 'Weiß'),
    '= το lidar,': ('= the lidar,', '= das LiDAR,'),
    'γαλάζιο': ('blue', 'blau'),
    '= η κάμερα,': ('= the camera,', '= die Kamera,'),
    'κόκκινο': ('red', 'rot'),
    '= εκεί που η κάμερα βλέπει εμπόδιο ΠΙΟ ΚΟΝΤΑ από το lidar.':
        ('= where the camera sees an obstacle NEARER than the lidar does.',
         '= wo die Kamera ein Hindernis NÄHER sieht als das LiDAR.'),
    'Τα κόκκινα είναι ο λόγος που υπάρχει αυτή η κάρτα: το C1 κόβει μία '
    'οριζόντια φέτα στα 60.6 cm, οπότε ένα τραπέζι, ένα σκαλί, ένα σκυμμένο '
    'κεφάλι ή μια γάτα ζουν ακριβώς στο κενό του. Η κάμερα κοιτάει από τα '
    '53.6 cm και τα πιάνει, αλλά μόνο μπροστά — τα ~87° του κώνου της. Έξω '
    'από αυτόν υπάρχει μόνο λευκό, και αυτό είναι σωστό, όχι διαφωνία.':
        ('The red is why this card exists: the C1 cuts one horizontal slice at '
         '60.6 cm, so a table top, a step, a bent-over head or a cat lives '
         'exactly in its blind gap. The camera looks from 53.6 cm and catches '
         'them, but only ahead — its ~87° cone. Outside that there is only '
         'white, and that is correct, not a disagreement.',
         'Das Rot ist der Grund für diese Karte: der C1 schneidet eine '
         'waagerechte Scheibe bei 60,6 cm, also leben Tischplatte, Stufe, ein '
         'gebeugter Kopf oder eine Katze genau in seiner Lücke. Die Kamera '
         'schaut ab 53,6 cm und erfasst sie, aber nur nach vorn — ihr ~87°- '
         'Kegel. Außerhalb gibt es nur Weiß, und das ist richtig, kein '
         'Widerspruch.'),
    'Πολύ κοντά ο κώνος στενεύει μόνος του: το βάθος βγαίνει από δύο φακούς '
    'και σε απόσταση μισού μέτρου τα άκρα του καρέ δεν τα βλέπουν και οι '
    'δύο, οπότε εκεί δεν υπάρχει μέτρηση. Μετρήθηκε 36° μπροστά σε εμπόδιο '
    'στα 40 cm. Δεν είναι βλάβη — κάνε ένα βήμα πίσω και ο κώνος ανοίγει.':
        ('Up close the cone narrows on its own: depth comes from two lenses, '
         'and at half a metre the edges of the frame are not seen by both, so '
         'there is no measurement there. 36° was measured facing an obstacle '
         'at 40 cm. Not a fault — step back and the cone opens up.',
         'Aus der Nähe verengt sich der Kegel von selbst: Tiefe entsteht aus '
         'zwei Linsen, und bei einem halben Meter sehen nicht beide die Ränder '
         'des Bildes, dort gibt es also keine Messung. 36° wurden vor einem '
         'Hindernis in 40 cm gemessen. Kein Fehler — tritt zurück und der '
         'Kegel öffnet sich.'),
    'Το πάτωμα κόβεται επίτηδες κάτω από':
        ('The floor is deliberately cut below', 'Der Boden wird bewusst unter'),
    'cm και το ταβάνι πάνω από': ('cm and the ceiling above',
                                  'cm abgeschnitten und die Decke über'),
    'cm: χωρίς αυτό η κάμερα «βλέπει» το χαλί μισό μέτρο μπροστά και τα '
    'πάντα γίνονται κόκκινα.':
        ('cm: without it the camera "sees" the carpet half a metre ahead and '
         'everything turns red.',
         'cm: sonst „sieht" die Kamera den Teppich einen halben Meter voraus '
         'und alles wird rot.'),

    # ── Sensor fusion badges (JS) ─────────────────────────────────────────
    'καμία ένδειξη': ('no reading', 'keine Anzeige'),
    'ΣΙΩΠΗ': ('SILENT', 'STUMM'),
    'αργό': ('slow', 'langsam'),
    'λείπει πηγή': ('a source is missing', 'Quelle fehlt'),
    'μεγάλη απόκλιση': ('large divergence', 'große Abweichung'),
    'αποκλίνουν': ('diverging', 'weichen ab'),
    'συμφωνούν': ('agreeing', 'stimmen überein'),
    'μεγάλη διόρθωση': ('large correction', 'große Korrektur'),
    'μαζεύει': ('pulling back', 'zieht zurück'),
    'μόλις κινηθεί το ρομπότ': ('once the robot moves',
                                'sobald der Roboter fährt'),
    'χωρίς βάθος': ('no depth', 'keine Tiefe'),
    'κρυφά': ('hidden', 'versteckt'),
    'σε': ('across', 'über'),
    'κατευθύνσεις': ('directions', 'Richtungen'),
    'κανένα': ('none', 'keines'),
}


def as_js_table() -> dict:
    """{greek: {'en': ..., 'de': ...}} — the shape the page's t() expects."""
    return {src: {'en': en, 'de': de} for src, (en, de) in TRANSLATIONS.items()}
