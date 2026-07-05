# Runbook — Spin/oscillation fix HW re-test (kela3)

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
1. **goto_room** σε 2-3 δωμάτια (προσοχή domatio_mbamba / toualeta — goal κοντά σε τοίχο).
2. **MPPI** live validation (obstacle avoidance με πραγματικό εμπόδιο).
3. **Orchestration end-to-end** — μία σύνθετη φωνητική εντολή, plan→navigate→act→report μόνο του.
