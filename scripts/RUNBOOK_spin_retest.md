# Runbook — Επόμενη ζωντανή συνεδρία στο kela3 (μία καθισιά)

Ενοποιημένο recipe για όλα τα open items που θέλουν το robot ζωντανό. Κάν' τα με τη
σειρά — το nav πρέπει να οδηγεί σωστά **πριν** τα υπόλοιπα.

## Master checklist (7 items)

**PART 1 — Nav chain (κοινό launch, terminals A–D παρακάτω):**
- [ ] **#1 Spin fix HW re-test** — RViz goal με στροφή ~130° → ομαλό τόξο (commit 18f6632, `ax_max` 0.25→2.0)
- [ ] **#2 goto_room** σε 2-3 δωμάτια (προσοχή domatio_mbamba 0.25m / toualeta 0.30m — goal κοντά σε τοίχο)
- [ ] **#3 Residual final-approach stall** — αν κολλάει ~0.23m πριν το goal: μεγαλύτερο `xy_goal_tolerance` (ΟΧΙ footprint/inflation)
- [ ] **#4 MPPI live validation** — obstacle avoidance με πραγματικό εμπόδιο στη διαδρομή
- [ ] **#5 Orchestration end-to-end** — μία σύνθετη φωνητική εντολή, plan→navigate→act→report μόνο του

**PART 2 — Voice HW tests (ξεχωριστό launch, δες κάτω):**
- [ ] **#6 Wake-word "Έι ρομπότ"** — HW-heard στο XVF3800 (false-trigger check)
- [ ] **#7 TTS barge-in gate** — το μικρόφωνο κόβεται όσο μιλάει το robot

**PART 3 — Νέα features 2026-07-06 (code-complete, HW-untested, δες κάτω):**
- [ ] **#8 Semantic object memory** — θυμάται πού είναι αντικείμενα (map frame) + RAG recall
- [ ] **#9 Dynamic obstacle layer** — άνθρωπος/κατοικίδιο στη διαδρομή → Nav2 παρακάμπτει
- [ ] **#10 Pick visual servoing** — closed-loop XY πριν το grasp (θέλει use_arm + tf_base_arm calib)
- [ ] **#11 Voice barge-in** — "Έι ρομπότ" ενώ μιλάει → σταματά & ακούει

**PART 4 — Fetch mission «φέρε μου το X» (code-complete, HW-untested, δες κάτω):**
- [ ] **#12 Fetch** — βρες→πιάσε(hold)→κουβάλα→παράδωσε (συνθέτει #8+#10+nav); delivery start_pose ή follow

---

## PART 1 — Nav chain

Στόχος: επιβεβαίωση στο πραγματικό robot ότι το fix του spin bug (commit 18f6632,
MPPI `ax_max` 0.25→2.0) δουλεύει — RViz goal με στροφή → ομαλό τόξο αντί για
στριφογύρισμα επιτόπου. Verified σε sim, μένει το hardware.

Δες και: `TODO.md`, memory `project_robot_nav_path_wall_bug`.

---

## 1) Build + Launch  (terminal A)

```bash
cd ~/robot_ws
colcon build --packages-select home_robot --symlink-install
source install/setup.bash
ros2 launch home_robot localize.launch.py map:=kela3 use_obstacle_safety:=true
```

- `use_obstacle_safety:=true` ΥΠΟΧΡΕΩΤΙΚΟ — αλλιώς τίποτα δεν κάνει relay
  `cmd_vel → cmd_vel_safe` και το roomba δεν κινείται (default false στο localize).
- Χάρτης default ήδη `kela3`, AprilTag reloc default on.
- Περίμενε: RViz ανοίγει, AMCL κλειδώνει, scan κάθεται πάνω στους τοίχους.
  Αν η pose είναι λάθος, πλησίασε το AprilTag (σαλόνι) για absolute reloc.

## 2) Preflight + ενεργοποίηση nav chain  (terminal B)

```bash
~/robot_ws/src/home_robot/scripts/preflight.sh
~/robot_ws/src/home_robot/scripts/ensure_nav_active.sh
```

Θέλεις 0 FAIL και όλους τους nav κόμβους `active`.

## 3) Recorder — ΞΕΚΙΝΑ ΠΡΙΝ στείλεις goal  (terminal C)

```bash
~/robot_ws/src/home_robot/scripts/record_nav_test.sh spinfix_retest
```

Καταγράφει tf/scan/odom/imu/amcl + όλη την αλυσίδα cmd_vel. Ctrl-C στο τέλος.
Bag στο `~/nav_test_bags/`.

## 4) Live watch  (terminal D)

```bash
ros2 topic echo /cmd_vel_safe
```

---

## Το τεστ

RViz → **Nav2 Goal** σε σημείο που απαιτεί **στροφή ~130°**, λίγα μέτρα, στον
διάδρομο (ή goto στην κουζίνα). Αυτό ήταν το σενάριο που κόλλαγε.

## PASS / FAIL

| Σημάδι | PASS (fix ok) | FAIL |
|--------|---------------|------|
| `vx` (cmd_vel_safe linear.x) | ανεβαίνει ομαλά ~0.20 m/s | κολλάει ~0.011 m/s → ax_max δεν χτίστηκε / regression |
| `wz` (angular.z) | peak ~0.4–0.5, ομαλό | pinned ±0.6 αλλάζοντας πρόσημο = ταλάντωση |
| κατεύθυνση | μπροστά, τόξο | όπισθεν / backup recovery |
| scan στο RViz | μένει κολλημένο στους τοίχους | φεύγει/χοροπηδάει AMCL → localization, όχι MPPI |

**Διάγνωση αν αποτύχει:**
- vx κολλημένο + wz pinned → MPPI/build: ξανα-build, βεβαιώσου `ax_max: 2.0`
  πέρασε στο install (`grep ax_max install/home_robot/share/home_robot/config/nav2_params.yaml`).
- scan φεύγει από τοίχους + AMCL χοροπηδάει → localization branch: ξανά-reloc
  με AprilTag, έλεγξε EKF yaw.

**Αναμενόμενο residual (ΟΚ, όχι αποτυχία):** μπορεί να κολλήσει ~0.23 m πριν το
goal κοντά σε τοίχο (circumscribed 0.344 > inflation 0.30). Fix αργότερα με
μεγαλύτερο `xy_goal_tolerance` — ΟΧΙ αλλαγή footprint/inflation (footprint & πόρτες
επιβεβαιωμένα σωστά, δες `TODO.md`).

---

## Αν περάσει → συνεχίζουμε στην ίδια συνεδρία

Με το nav να οδηγεί σωστά, δοκιμάζουμε αλυσιδωτά (όλα χρειάζονται το robot ζωντανά):
1. **goto_room** (#2) σε 2-3 δωμάτια (προσοχή domatio_mbamba / toualeta — goal κοντά σε τοίχο).
2. **MPPI** (#4) live validation (obstacle avoidance με πραγματικό εμπόδιο).
3. **Orchestration end-to-end** (#5) — μία σύνθετη φωνητική εντολή, plan→navigate→act→report μόνο του.

---

## PART 2 — Voice HW tests (#6-7)

Στόχος: επιβεβαίωση ζωντανά στο XVF3800 mic array ότι (α) το wake word "Έι ρομπότ"
δεν κάνει false-triggers και πιάνει σωστά, (β) το barge-in gate κόβει το μικρόφωνο
όσο μιλάει το robot (δεν αυτο-τριγκάρεται από το TTS του).

Fixes: wake word 6adbc9c, barge-in gate fb048d4. Δες memory `project_robot_voice_tools`,
`project_robot_wakeword_recordings_todo`, `voice_gate.py`, `tests/test_voice_gate.py`.

### Launch

⚠️ Το `localize.launch.py` ΔΕΝ προωθεί τα voice flags — τα voice nodes ξεκινούν
ξεχωριστά. Δύο επιλογές:

**A) Voice σε ΞΕΧΩΡΙΣΤΟ terminal, πάνω στο running nav** (προτιμότερο για isolated test):
```bash
cd ~/robot_ws && source install/setup.bash
ros2 launch home_robot bringup.launch.py \
  use_wake_word:=true use_stt:=true use_tts:=true use_llm:=true use_memory:=true \
  use_slam:=false use_localization:=false use_roomba:=false \
  use_camera:=false use_rviz:=false use_obstacle_safety:=false use_joy:=false
```
⚠️ ΕΠΑΛΗΘΕΥΣΕ ζωντανά ότι δεν κάνει conflict με το ήδη-τρέχον nav (διπλά nodes).
Αν διαμαρτυρηθεί, κατέβασε το nav και τρέξε ΜΟΝΟ αυτό (τα voice tests δεν θέλουν κίνηση).

**B) Αν τα voice tests γίνουν ΜΟΝΑ τους** (χωρίς nav, ό,τι πιο απλό):
ίδια εντολή — είναι ήδη γραμμένη με όλα τα nav-flags false.

Προϋποθέσεις: Lemonade/NPU up (`lemond.service`), XVF3800 συνδεμένο,
`config/models/max.onnx` deployed (threshold 0.5).

### Το τεστ

**#6 Wake word:**
- Πες καθαρά **"Έι ρομπότ"** → πρέπει να ξυπνήσει (LED/log `wake detected`).
- Πες σκέτο **"Μαξ"** και σκέτο **"Ρομπότ"** πολλές φορές + κανονική κουβέντα →
  πρέπει να ΜΗΝ ξυπνάει (hard negatives). Αν ξυπνάει, ηχογράφησε πραγματικά
  positives με `training/wake_word_hey_robot/record_real.py` και ξανα-train.

**#7 Barge-in gate:**
- Δώσε εντολή που προκαλεί μεγάλη TTS απάντηση.
- Όσο μιλάει το robot, παρακολούθησε `ros2 topic echo /tts/speaking` (πρέπει `true`)
  και βεβαιώσου ότι η ίδια του η φωνή ΔΕΝ ξανα-τριγκάρει wake/STT recording.

### PASS / FAIL

| Σημάδι | PASS | FAIL |
|--------|------|------|
| "Έι ρομπότ" | ξυπνάει κάθε φορά | δεν πιάνει → record real positives + retrain |
| σκέτο "Μαξ"/"Ρομπότ"/κουβέντα | ΔΕΝ ξυπνάει | false-trigger → hard-negative retrain |
| `/tts/speaking` στο playback | `true` (+0.3s tail) | δεν δημοσιεύεται → tts_node param |
| αυτο-echo όσο μιλάει | mic muted, κανένα self-wake | αυτο-τριγκάρεται → SpeakingGate wiring |

---

## PART 3 — Νέα perception/interaction features (#8-11)

Χτίστηκαν 2026-07-06 (commits 020d61e, e788e32, 257798b, c24dd1c). Code-complete +
unit-tested (35 tests), **κανένα HW-tested**. Δες memory
`project_robot_perception_features`, `TODO.md` (section «✨ New features»).

### Launch — τα #8/#9 (+#10 με arm) μαζί, πάνω στο nav

Το `use_perception:=true` ανεβάζει camera + YOLO detector + tracker + τα δύο
dynamic-obstacle layers + object memory. Αντικαθιστά το lean depth stream με την
πλήρη κάμερα (color+depth+aligned+pointcloud).

```bash
cd ~/robot_ws && colcon build --packages-select home_robot --symlink-install && source install/setup.bash
ros2 launch home_robot localize.launch.py map:=kela3 \
  use_obstacle_safety:=true use_perception:=true use_arm:=true   # use_arm μόνο για #10
```
⚠️ Κοστίζει iGPU/CPU (ο detector). Αν το nav ήδη τρέχει χωρίς perception, κατέβασέ το
και ξαναανέβασέ το μ' αυτό — μη τρέχεις δύο realsense (USB conflict).

### #8 Semantic object memory

```bash
ros2 topic echo /object_memory              # JSON: label/x/y/z(map)/room/last_seen
# RViz: πρόσθεσε MarkerArray /object_memory_markers
```
- Γύρνα το robot σε 2-3 δωμάτια με αντικείμενα (cup, bottle, chair, tv…).
- Μετά ρώτα φωνητικά: «Έι ρομπότ, πού είναι το ποτήρι;» (μέσω RAG recall).
- Persist: `~/.robot_object_memory.json` — επιβιώνει restart.

| Σημάδι | PASS | FAIL |
|--------|------|------|
| markers | κάθονται πάνω στα πραγματικά αντικείμενα, σταθερά | πλανώνται/διπλασιάζονται → μείωσε `merge_distance` ή ανέβασε `min_conf` |
| room label | σωστό δωμάτιο ανά αντικείμενο | λάθος → έλεγξε locations.yaml / map→base TF |
| φωνητικό recall | απαντά με το δωμάτιο | κενό → `rag_push` on? Lemonade up? |

### #9 Dynamic obstacle layer

- Με το robot να πλοηγείται προς goal, μπες με τα πόδια στη διαδρομή (ή βάλε κάποιον).
- Watch: `ros2 topic echo /semantic_obstacles` (person cylinder) και το local costmap
  στο RViz να «φουσκώνει» γύρω σου.

| Σημάδι | PASS | FAIL |
|--------|------|------|
| costmap γύρω από άνθρωπο | κόκκινη inflation, Nav2 παρακάμπτει | τίποτα → tracker/detector down, ή camera_info λείπει |
| ταχύ κινούμενο εμπόδιο | `/predicted_obstacles` μπροστά από την πορεία του | καθόλου πρόβλεψη → κατέβασε `min_speed` |
| tuning | `person_radius` (0.6) πολύ σφιχτό/χαλαρό → ρύθμισε | — |

### #10 Pick visual servoing (θέλει use_arm)

⚠️ **ΠΡΩΤΑ βαθμονόμησε το `tf_base_arm`** — το servo καθαρίζει jitter, ΟΧΙ systematic
calibration bias (η κάμερα δεν βλέπει το gripper). Δοκίμασε αργά / με το χέρι κοντά
στο E-stop.

```bash
ros2 topic pub --once pick_command std_msgs/String '{data: "{\"label\": \"cup\"}"}'
# παρακολούθησε τα logs του pick_place_node: "Servo nudge N: Δ=..mm" → "Servo converged"
```

| Σημάδι | PASS | FAIL |
|--------|------|------|
| servo logs | Δ μικραίνει, «converged» πριν το descend | Δ μεγαλώνει/ταλαντώνεται → μείωσε βήμα, έλεγξε TF sign |
| grasp | πιάνει το αντικείμενο | αστοχεί σταθερά κατά offset → tf_base_arm calibration |
| relock | μένει στο ίδιο αντικείμενο | πηδά σε άλλο → μείωσε `servo_relock_radius` |

### #11 Voice barge-in

- Δώσε εντολή με μεγάλη TTS απάντηση· **όσο μιλάει**, πες «Έι ρομπότ».
- Πρέπει να κόψει αμέσως την ομιλία (`sd.stop()`) και να ξεκινήσει νέο γύρο.

| Σημάδι | PASS | FAIL |
|--------|------|------|
| διακοπή στο «Έι ρομπότ» | σταματά μέσα σε <0.5s, ακούει | δεν σταματά → `allow_barge_in` off / threshold ψηλά |
| self-interrupt | ΔΕΝ διακόπτει τον εαυτό του | αυτο-κόβεται → AEC beam όχι καθαρό → `allow_barge_in:=false` ή ↑threshold |

---

## PART 4 — Fetch mission «φέρε μου το X» (#12)

Υλοποιήθηκε 2026-07-06 (f161d50/2635f31/2db066c/40717ec/d4e12c6/fb65fe5). Συνθέτει
object memory → nav → pick(hold) → carry → place. Code-complete + 51 unit tests,
**HW-untested**. Design: `docs/PLAN_fetch_mission.md`, memory
`project_robot_orchestration`.

### Προαπαιτούμενα (ΜΗΝ τα παραλείψεις)

1. **Calibrate `tf_base_arm`** (κοινό με #10 — αλλιώς το grasp αστοχεί).
2. **Το object memory (#8) να έχει ΔΕΙ το αντικείμενο** — γύρνα το robot να το
   καταγράψει πρώτα, αλλιώς το fetch πέφτει σε αργό room-by-room search.
3. Όλα μαζί ζωντανά: perception + arm + mission + nav.

### Launch

```bash
cd ~/robot_ws && colcon build --packages-select home_robot --symlink-install && source install/setup.bash
ros2 launch home_robot localize.launch.py map:=kela3 \
  use_obstacle_safety:=true use_perception:=true use_arm:=true
# ξεχωριστά (ή πρόσθεσέ τα): use_mission:=true + voice (use_llm/stt/tts/wake_word)
# delivery: πρόσθεσε delivery_mode:=follow για homing στον χρήστη (default start_pose)
```

### Το τεστ

```bash
# απευθείας (χωρίς φωνή):
ros2 topic pub --once mission/start std_msgs/String '{data: "fetch:cup"}'
# ή φωνητικά: «Έι ρομπότ, φέρε μου το ποτήρι»
ros2 topic echo /mission/status      # navigating→inspecting→...→done
```
Στάδια που βλέπεις: `RESOLVE` (μιλάει «πάω να φέρω») → οδηγεί σε approach pose →
`VERIFY` → `PICK` (κρατά) → κουβαλάει πίσω → `PLACE` («ορίστε»).

### PASS / FAIL

| Στάδιο | PASS | FAIL |
|--------|------|------|
| RESOLVE | βρίσκει θέση από μνήμη (ή search) | «δεν ξέρω πού είναι» → object memory άδειο (δες #8) |
| approach | σταματά ~0.4m πριν, κοιτάζοντας το αντικείμενο | πάνω στο αντικείμενο / λάθος γωνία → `fetch_approach_dist` |
| VERIFY | το βλέπει σε grasp range, προχωράει | δεν το βλέπει → μνήμη stale· κάνει retry/search |
| PICK hold | πιάνει & **κρατά** (δεν το αφήνει) | το αφήνει αμέσως → `hold` δεν πέρασε |
| carry | κουβαλάει σταθερά ενώ οδηγεί | βραχίονας ασταθής/πέφτει → χρειάζεται "carry pose" (RISK) |
| deliver start_pose | γυρίζει εκεί που έδωσες εντολή & αφήνει | λάθος σημείο → TF start-pose δεν πιάστηκε |
| deliver follow | κεντράρει πάνω σου, σταματά ~0.8m, αφήνει | κυνηγά phantom / δεν σταματά → `deliver_distance`, person detections |
| cancel | «ακύρωσε αποστολή» → σταματά & **αφήνει** το αντικείμενο | μένει να κρατά → bug |
