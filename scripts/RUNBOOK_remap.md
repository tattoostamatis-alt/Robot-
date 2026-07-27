# Runbook — Ξαναχαρτογράφηση (remap) του σπιτιού

Γράφτηκε 2026-07-27 για τον χάρτη `malou`. Ισχύει για κάθε remap.

**Γιατί το κάνουμε αυτή τη φορά:** η περιοχή της βάσης φόρτισης έχει καταγραφεί
με περιττά εμπόδια. Μετρημένο στον τρέχοντα `malou`: η θέση `dock` έχει μόνο
**0.14 m** ελεύθερο χώρο ενώ το ρομπότ χρειάζεται 0.20 m, και **ΟΛΗ** η γραμμή
προσέγγισης — από 0.20 ως 1.90 m μακριά από τη βάση — μένει κάτω από 0.22 m.
Ο planner δεν μπορεί να πλησιάσει τη βάση, γι' αυτό το ρομπότ σταματούσε πάντα
«κοντά αλλά όχι μέσα». Η θέση του AprilTag βγάζει 0.05 m ενώ στην πραγματικότητα
το ρομπότ στέκεται εκεί άνετα — απόδειξη ότι ο χάρτης εκεί είναι λάθος, όχι ο
χώρος. Τα υπόλοιπα 6 δωμάτια είναι μια χαρά (42/42 διαδρομές βρέθηκαν).

---

## ⚠️ ΠΡΙΝ ΞΕΚΙΝΗΣΕΙΣ — τι θα χαλάσει

Ένα remap ακυρώνει **ό,τι δένει σε συντεταγμένες χάρτη**. Δεν είναι προαιρετικά,
είναι μέρος της δουλειάς — άφησε χρόνο και για το ΜΕΡΟΣ Γ:

- `config/locations.yaml` — και τα 7 σημεία
- το room_mask
- η βαθμονόμηση AprilTag (`~/.ros/saloni_tag_map_pose.yaml`)
- `config/keepout_zones.yaml` (ήδη stale· ανενεργό με `use_keepout:=false`,
  οπότε δεν επείγει)

**Κράτα αντίγραφο πριν:**

```bash
cd ~/robot_ws/src/home_robot
cp maps/malou.pgm maps/malou.pgm.bak
cp maps/malou.yaml maps/malou.yaml.bak
cp config/locations.yaml config/locations.yaml.bak
cp ~/.ros/saloni_tag_map_pose.yaml ~/.ros/saloni_tag_map_pose.yaml.bak
```

---

## ΜΕΡΟΣ Α — Χαρτογράφηση

### A0. Προετοιμασία χώρου (καθορίζει την ποιότητα)

- [ ] **Μάζεψε ό,τι δεν είναι μόνιμο**, ειδικά γύρω από τη βάση: καλώδια στο
      πάτωμα, το power station, τσάντες, καρέκλες εκτός θέσης. **Ό,τι δει το
      lidar γίνεται μόνιμος τοίχος.** Αυτό είναι ο λόγος του remap — μην το
      επαναλάβεις.
- [ ] Άνοιξε όλες τις πόρτες που θέλεις διαβατές.
- [ ] Κανείς να μην κυκλοφορεί στον χώρο όσο χαρτογραφείς.

### A1. Έλεγχοι υγείας — ΠΡΙΝ, όχι μετά

```bash
./scripts/preflight.sh          # ορφανά processes
```

- [ ] **IMU ζωντανό:** `ros2 topic hz /imu/data` → πρέπει να δίνει ρυθμό.
      ‼️ Νεκρό IMU είναι **σιωπηλό** και παραμορφώνει τον χάρτη χωρίς κανένα
      μήνυμα λάθους. Αν δεν δίνει τίποτα, φτιάξ' το πρώτα.
- [ ] Lidar ζωντανό: `ros2 topic hz /scan`

### A2. Εκκίνηση SLAM

```bash
ros2 launch home_robot bringup.launch.py use_slam:=true
```

### A3. Οδήγηση

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -p speed:=0.10 -p turn:=0.10
```

‼️ **Πάντα 0.10/0.10.** Οι default ταχύτητες είναι πολύ γρήγορες και θολώνουν
τον χάρτη.

- [ ] Πέρνα από κάθε δωμάτιο **αργά**.
- [ ] **Στην περιοχή της βάσης: 3-4 περάσματα από διαφορετικές γωνίες.** Ένα
      μόνο πέρασμα αφήνει θόρυβο που γίνεται μόνιμο εμπόδιο — ακριβώς αυτό
      έσπασε τον προηγούμενο χάρτη.
- [ ] Κλείσε βρόχους: γύρνα σε σημεία που έχεις ήδη περάσει, βοηθά το
      loop closure.

### A4. Αποθήκευση — ΧΡΕΙΑΖΟΝΤΑΙ ΚΑΙ ΤΑ ΔΥΟ

```bash
cd ~/robot_ws/src/home_robot                             # το -f είναι σχετικό path
./scripts/save_map.sh malou                              # .posegraph + .data
ros2 run nav2_map_server map_saver_cli -f maps/malou     # .pgm + .yaml
```

‼️ Το `save_map.sh` **μόνο του δεν φτάνει**: σώζει το pose-graph του
slam_toolbox, όχι το `.pgm/.yaml` που φορτώνει το AMCL. Αν παραλείψεις το
δεύτερο, ο νέος χάρτης δεν θα χρησιμοποιηθεί ποτέ και θα νομίζεις ότι το remap
δεν άλλαξε τίποτα.

- [ ] Επιβεβαίωσε ότι και τα 4 αρχεία έχουν σημερινή ημερομηνία:
      `ls -la maps/malou.*`

---

## ΜΕΡΟΣ Β — Έλεγχος του νέου χάρτη ΠΡΙΝ ξαναδιδάξεις

Μη σπαταλήσεις χρόνο διδάσκοντας σημεία πάνω σε κακό χάρτη.

- [ ] Ζήτα από τον Claude να ξανατρέξει την ανάλυση πλοήγησης (τα scripts είναι
      στο scratchpad της συνεδρίας 2026-07-27: `nav_audit.py`, `dock_area.py`).
      Το ζητούμενο: **η γραμμή προσέγγισης της βάσης να φτάνει clearance
      ≥ 0.30 m** κάπου μέσα στο πρώτο μισό μέτρο. Τώρα δεν το φτάνει πουθενά.
- [ ] Οπτικά: `ros2 launch home_robot view_map.launch.py` (default = malou)
      — κοίτα αν η περιοχή της βάσης άνοιξε και αν έφυγαν οι διάσπαρτες
      κουκκίδες. ‼️ Εδώ το `map:=` θέλει **πλήρες path** σε `.yaml`, όχι σκέτο
      όνομα όπως στο `localize.launch.py`.

Αν δεν βελτιώθηκε, επανάλαβε το ΜΕΡΟΣ Α — είναι φθηνότερο από το να ανακαλύψεις
το πρόβλημα μετά από όλα τα παρακάτω.

---

## ΜΕΡΟΣ Γ — Ξαναδίδαξε ό,τι ακύρωσε το remap

### Γ1. Σημεία (`locations.yaml`)

Δύο τρόποι — ο δεύτερος είναι καλύτερος για το `dock`:

```bash
# με κλικ στον χάρτη σε RViz ("Publish Point"), χωρίς ρομπότ
ros2 launch home_robot view_map.launch.py
ros2 run home_robot record_location.py --click

# ή οδηγώντας το ρομπότ στο σημείο (θέλει localize.launch.py να τρέχει)
ros2 run home_robot record_location.py kouzina
```

- [ ] Και τα 6 δωμάτια.
- [ ] **`dock` — κάν' το με το ρομπότ, όχι με κλικ.** Οδήγησέ το στη θέση από
      την οποία **μπαίνει καλά στη βάση** και κατέγραψε εκεί. Μην υπολογίσεις
      «0.4 m μπροστά της» — αυτό ήταν το παλιό λάθος. Το σημείο πρέπει να έχει
      clearance ≥ 0.30 m ΚΑΙ οπτική επαφή με τη βάση για το IR.
- [ ] Έλεγχος: `ros2 run home_robot record_location.py --list`

### Γ2. Room mask

```bash
python3 scripts/make_room_mask.py malou
```

- [ ] Κοίτα το `maps/room_mask_malou_preview.display.png` πριν το μετονομάσεις
      σε `maps/room_mask.png`.
- [ ] Πρόσεξε το `free unreached: N px` στην έξοδο — μεγάλος αριθμός σημαίνει
      ότι κάποιο δωμάτιο δεν καλύφθηκε.

### Γ3. AprilTag — ‼️ ΜΗΝ ΤΟ ΞΕΧΑΣΕΙΣ

Η βαθμονόμηση είναι **map-specific**. Ένα stale calibration είναι **χειρότερο
από καθόλου**: ο relocalizer θα σπρώχνει ενεργά το AMCL σε λάθος θέση κάθε ~8
δευτερόλεπτα που βλέπει το tag.

```bash
# ρομπότ σωστά εντοπισμένο, tag 0.4-1.0 m μακριά, και οι 4 γωνίες στο κάδρο
ros2 launch home_robot localize.launch.py
ros2 service call /apriltag_relocalizer/calibrate_tag std_srvs/srv/Empty
```

- [ ] Επαλήθευση: απομάκρυνε το ρομπότ, ξαναφέρ' το μπροστά στο tag, και δες
      στο log ότι το `/localize_globally` αναφέρει **"shifted 0.00 m"**.
      Γι' αυτό κρίνε — ΟΧΙ για την covariance του AMCL.
- [ ] Νέο αρχείο: `ls -la ~/.ros/saloni_tag_map_pose.yaml` (σημερινή ημερομηνία)

### Γ4. Commit

```bash
git add config/locations.yaml maps/malou.pgm maps/malou.yaml \
        maps/malou.data maps/malou.posegraph maps/room_mask.png
git commit -m "map: remap malou — open up the dock approach"
git push origin master
```

---

## ΜΕΡΟΣ Δ — Επαλήθευση

- [ ] `ros2 launch home_robot localize.launch.py` — το ρομπότ εντοπίζεται σωστά
- [ ] Ένα `goto` σε 2-3 δωμάτια
- [ ] **Το τεστ που μετράει:** `/dock` — φτάνει τώρα ο planner κοντά στη βάση;
      Αν ναι, αναλαμβάνει το IR homing (`use_ir_homing`, default true).
      Αν το ρομπότ ξεφεύγει σταθερά προς μία πλευρά στην τελική προσέγγιση:
      `ros2 param set /roomba_driver ir_swap_buoys true`

---

## Σχετικά

- Ανάλυση που οδήγησε σε αυτό: μνήμη `project_robot_dock_map_blocked`
- IR homing: `home_robot/ir_homing.py` (commit b92ac92)
- Προηγούμενο remap (kela3→malou, 2026-07-22): μνήμη `project_robot_map_malou`
