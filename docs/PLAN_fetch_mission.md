# Plan — "Φέρε μου το X" (fetch mission)

Σχέδιο υλοποίησης. **Δεν έχει γραφτεί κώδικας ακόμα** — αυτό είναι το design για
review. Συνθέτει τρία κομμάτια που ήδη υπάρχουν αλλά είναι ασύνδετα:
semantic object memory (πού είναι το X) → Nav2 (πήγαινε) → pick-place με visual
servoing (πιάσ' το) → επιστροφή/παράδοση.

## Στόχος

Φωνητικά: «Ρομπότ Μαξ, φέρε μου το ποτήρι.» → το robot βρίσκει πού είναι το
ποτήρι, πάει, το πιάνει, γυρίζει σε μένα, το αφήνει, αναφέρει.

## Πού μπαίνει

Νέο mission `fetch:<label>` στο **`mission_executor_node.py`** — δίπλα στο υπάρχον
`find:<label>`, ξαναχρησιμοποιεί `_navigate_to`, `_await_future`, cancel handling,
`_speak`, τη state machine. **Όχι** νέο node (consistency + reuse).

Νέο LLM tool `fetch` στο `llm_bridge_node.py` που δημοσιεύει `mission/start` =
`fetch:<label>` (terse description — TOOLS είναι prefill-capped, δες
[[project_robot_voice_tools]]).

## Ροή (state machine)

```
RESOLVE → NAVIGATE_TO_OBJECT → VERIFY → PICK → NAVIGATE_TO_USER → DELIVER → REPORT
             (fallback: find-search αν άγνωστο)      (retry x1)
```

1. **RESOLVE** — πού είναι το X;
   - Publish στο `object_memory/query` (String label), περίμενε `object_memory/answer`
     (JSON instance) με timeout ~1s. Αν επιστρέψει instance με πρόσφατο `last_seen`
     (π.χ. < `memory_max_age` = 300s) → κράτα (x,y) map + room.
   - **Fallback**: αν άγνωστο/παλιό → τρέξε τη λογική του `find:<label>` (γύρος
     δωματίων μέχρι να το δει live), και πάρε τη θέση από το live detection.
   - Αν πουθενά → REPORT "δεν ξέρω πού είναι το X" + FAILED.

2. **NAVIGATE_TO_OBJECT** — πήγαινε σε **approach pose**, όχι πάνω στο αντικείμενο.
   - Νέος helper `_navigate_to_xy(x, y, yaw)` (το υπάρχον `_navigate_to` δέχεται
     μόνο named locations).
   - Approach pose: σημείο σε απόσταση `approach_dist` (~0.4m, arm reach + margin)
     από το αντικείμενο, πάνω στη γραμμή robot→object, με yaw να κοιτάει το
     αντικείμενο (ώστε να πέσει στο workspace κάμερας/βραχίονα). Καθαρή γεωμετρία
     → **pure helper** `approach_pose(robot_xy, obj_xy, dist)` (unit-testable).

3. **VERIFY** — το X είναι όντως εκεί; (η μνήμη μπορεί να είναι stale — μετακινήθηκε)
   - Περίμενε ~2s να σταθεροποιηθεί ο detector, ψάξε το label στο `detected_objects`
     σε grasp range (z < `grasp_range` ~1.0m).
   - Αν όχι εκεί → μικρή τοπική περιστροφή/re-approach, μετά fallback στο find-search.
   - Αν ναι → PICK.

4. **PICK** — δημοσίευσε `pick_command {"label": X}`, περίμενε `pick_result`
   ({status: ok|error}) με timeout (~30s). Το pick_place ήδη κάνει το visual
   servoing (257798b). Σε error → **retry x1** (re-approach), μετά FAILED.

5. **NAVIGATE_TO_USER** — παράδοση. Επιλογές (param `delivery_mode`):
   - **`start_pose`** (default, v1) — στην αρχή του mission κράτα το τρέχον pose του
     robot (TF map→base_link) ως delivery pose· γύρνα εκεί. Απλό & robust.
   - `follow` — βρες τον χρήστη με DOA/person_follower. Καλύτερο UX, πιο εύθραυστο.
   - `location:<room>` — σταθερό σημείο "εμένα".

6. **DELIVER** — άφησε το αντικείμενο (place: `pick_command` place-branch ή arm drop
   σε χαμηλό pose μπροστά) + `_speak("Ορίστε το X.")`.

7. **REPORT** — DONE + επιστροφή σε IDLE (όπως τα άλλα missions).

Cancel σε **κάθε** στάδιο μέσω `_cancel_flag` (ήδη υπάρχει: "ακύρωσε αποστολή" ή
κενό `mission/start`). Αν ακυρωθεί ενώ κρατά αντικείμενο → άφησέ το με ασφάλεια
πρώτα.

## Αλλαγές αρχείων

| Αρχείο | Αλλαγή |
|---|---|
| `home_robot/fetch_planner.py` (νέο) | pure helpers: `approach_pose()`, `resolve_target()` (memory-vs-live επιλογή), unit-testable χωρίς ROS |
| `home_robot/nodes/mission_executor_node.py` | `fetch:` dispatch + `_mission_fetch()` state machine + `_navigate_to_xy()` + subscribe `object_memory/answer`, `pick_result` + publish `object_memory/query`, `pick_command` + capture start pose (TF) |
| `home_robot/nodes/llm_bridge_node.py` | νέο `fetch` tool (terse) + dispatch → `mission/start` `fetch:<label>` |
| `launch/bringup.launch.py` | βεβαίωση ότι `use_mission` σηκώνει τον executor· fetch θέλει use_perception(camera+object_memory)+use_arm+nav μαζί |
| `tests/test_fetch_planner.py` (νέο) | approach-pose γεωμετρία, target resolution (memory hit/stale/miss), static wiring |
| `README.md` / `TODO.md` / runbook | καταγραφή + HW-test βήμα |

## Νέα params (mission_executor)

`fetch_approach_dist` (0.4m) · `fetch_grasp_range` (1.0m) · `memory_max_age` (300s) ·
`delivery_mode` (start_pose) · `pick_timeout` (30s) · `fetch_max_retries` (1).

## Ρίσκα / ανοιχτά (θέλουν HW)

- **Ο βραχίονας κρατά αντικείμενο ενώ κινείται η βάση** — ασφαλές για τον RoArm-M3;
  ίσως χρειαστεί "carry pose" (μαζεμένος, χαμηλό κέντρο βάρους) πριν το NAVIGATE_TO_USER.
- **tf_base_arm calibration** — προϋπόθεση για αξιόπιστο grasp (δες
  [[project_robot_perception_features]]).
- **Stale memory** — το VERIFY step το καλύπτει, αλλά αν το αντικείμενο μετακινείται
  συχνά, το fetch θα πέφτει σε find-search συχνά (αργό).
- **Delivery = start_pose** μπορεί να μην είναι εκεί που στέκεσαι· `follow` mode είναι
  το σωστό αλλά αργότερα.

## Πλάνο δουλειάς (σειρά)

1. `fetch_planner.py` + tests (pure, χωρίς hardware) — γεωμετρία & resolution.
2. `_mission_fetch()` state machine στον executor (RESOLVE→...→REPORT) + helpers.
3. `fetch` LLM tool + dispatch.
4. Wiring/docs. Build + unit tests.
5. **HW test** (χρειάζεται robot): use_perception+use_arm+nav, «φέρε το ποτήρι».

## Launch (όταν υλοποιηθεί)

```bash
ros2 launch home_robot localize.launch.py map:=kela3 \
  use_obstacle_safety:=true use_perception:=true use_arm:=true
# + use_mission:=true, use_llm/use_stt/use_tts/use_wake_word για φωνητικό trigger
```

Δες [[project_robot_orchestration]] (async-action bugs & `_await_future` pattern),
[[project_robot_perception_features]], [[project_robot_arm]].
