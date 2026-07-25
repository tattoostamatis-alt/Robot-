# home_robot — TODO

Status as of 2026-07-06. Items grouped by whether they need the robot powered on.

## ✨ New features (built 2026-07-06, code-complete + unit-tested, all HW-untested)

Enable together with `localize.launch.py ... use_perception:=true` (brings up
camera + detector + tracker + dynamic-obstacle layers + object memory).

- [ ] **Semantic object memory** (020d61e). `use_object_memory` node remembers where
      objects are in the map frame; feeds RAG recall ("πού είναι το X;"). Verify: drive
      around, `ros2 topic echo /object_memory`, check RViz `object_memory_markers` land on
      real objects, then ask Max via voice. Tune `merge_distance`/`min_conf` if noisy.
      **2026-07-09 hardened (desk, SORT-style):** added `confirm_count` (default 3) —
      an instance is *tentative* until seen 3× and is hidden from RAG push + "where is X?"
      answers, so a single false-positive detection no longer pollutes memory forever;
      EMA is now confidence-weighted so a marginal sighting pulls position less. RViz
      shows tentative instances faint with a "?". Unit-tested (14/14). HW-verify the
      confirm threshold isn't so high it drops real objects.
- [ ] **Dynamic obstacle layer** (was already built; e788e32 made it reachable in nav).
      Verify: with `use_perception:=true`, a person walking into the path inflates the local
      costmap (`/predicted_obstacles` + `/semantic_obstacles`) and Nav2 detours. Tune
      `person_radius` / prediction `horizon`.
      **2026-07-09 fixed (desk):** the prediction had a dimensional bug — tracker velocity
      is px/*cycle* but the code projected it as if 1 cycle = 1 s (and gated with a magic
      `min_speed*0.1`), so horizon/min_speed meant nothing and the forecast scaled with the
      detector frame rate. Now measures the real message dt, converts px/cycle→m/s, and
      requires `min_track_hits` (3) before trusting a SORT velocity. Extracted to
      `home_robot/obstacle_prediction.py`, unit-tested (9/9). NOTE still open: velocity is
      in camera frame so robot ego-motion isn't compensated — verify predictions while the
      base is moving, may need to gate on robot stationary or subtract odom twist.
- [ ] **Pick-place visual servoing** (257798b). Closed-loop XY refine before grasp. Needs
      `use_arm:=true`. FIRST calibrate `tf_base_arm` (servo can't fix a calibration bias).
      Verify slowly/by hand: watch the "Servo nudge/converged" logs, confirm it settles over
      the object before descending. Tune `servo_tolerance` / `servo_max_iters`.
      **2026-07-09 hardened (desk):** the eye-to-hand "servo" is really multi-frame target
      settling; it used to grasp at the *last* relock frame, so one noisy RealSense depth
      → descend into the table / grasp air. Now collects the relock estimates and grasps at
      their component-wise **median** (rejects a lone outlier frame), with convergence judged
      by the spread of recent frames. Extracted to `home_robot/servo_filter.py`, unit-tested
      (7/7). Still needs the tf_base_arm calibration + HW grasp test.
- [ ] **Voice barge-in** (c24dd1c). Say "Ρομπότ Μαξ" while Max is talking → he stops and
      listens. `allow_barge_in` defaults true (XVF3800 ch4 AEC). Verify he doesn't interrupt
      *himself* (self-echo); if he does, the AEC beam isn't clean → set false or raise threshold.
      **2026-07-09 hardened (desk):** added a stricter `barge_in_threshold` (0.70 vs wake 0.50)
      — a barge-in is only accepted while actively speaking if the score clears the higher bar,
      since self-echo false positives are brief marginal spikes while a real "Ρομπότ Μαξ" scores
      high/sustained. The whole decision is now a pure `wake_decision()` in voice_gate.py,
      unit-tested (reverb tail never barges, disabled→suppress, etc.). If it still self-echoes on
      HW, raise `barge_in_threshold` toward 0.8–0.9 before disabling barge-in.
- [ ] **Fetch mission ("φέρε μου το X").** `fetch:<label>` in mission_executor composes object
      memory → nav → pick(hold) → carry → place; LLM `fetch` tool. Needs `use_perception` +
      `use_arm` + `use_mission` + nav. **Prereqs: calibrate tf_base_arm; object memory must have
      seen the object.** Verify each stage (resolve/approach/verify/pick/deliver) via mission/status
      + pick_result/place_result. RISK: arm holding an object while the base drives — check RoArm-M3
      stability, may need a "carry pose". Design: `docs/PLAN_fetch_mission.md`.
      Delivery: default returns to where you stood (`delivery_mode:=start_pose`); try
      `delivery_mode:=follow` to have it home in on you (person detections) — verify it
      centres + stops at ~0.8m and doesn't chase phantoms.
      **2026-07-09 hardened (desk):** follow-delivery `homing_twist` is now a proportional
      follower — forward speed scales with remaining distance (capped) so it eases to a gentle
      stop at the user instead of driving full-speed then slamming to zero (safer approach to a
      person). Unit-tested (17/17). Still HW-untested end-to-end; carry-pose arm stability + the
      full resolve→approach→pick→carry→place chain remain the live-test items.

## 🧪 Needs the robot live (do these together in one session)

- [x] **Spin/oscillation fix — HW re-test on kela3.** CONFIRMED 2026-07-06: `/cmd_vel_safe`
      ramps smoothly to 0.20 m/s, wz stays small/smooth, no pin/oscillation, no spin-in-place.
      ax_max 0.25→2.0 fix holds. See `project_robot_nav_path_wall_bug`.
- [ ] **`goto_room` HW test — PARTIAL, fragile at doorways.** 2026-07-06: kouzina and saloni both
      reachable, but planner/controller repeatedly give up right at doorway/pinch points (corridor
      mid-point, kouzina↔saloni doorway) with "Failed to create plan"/"Failed to make progress",
      sometimes spin-recovery itself aborting on collision risk. Workaround that reliably works:
      a tiny manual `/cmd_vel` nudge (turn + creep, ~1s each) past the pinch point, then resend the
      same goal — succeeds every time tried. NOT an inflation/footprint problem (doorways are only
      ~0.78m, raising inflation would seal them — see nav2_params.yaml comments). Also watch for:
      (a) AMCL can diverge from the in-place spin-recovery attempts at these pinch points — recover
      via `ros2 service call /apriltag_relocalizer/relocalize std_srvs/srv/Empty` (one-shot per
      session, won't auto-refire just by showing the tag again); (b) RViz's own "2D Nav Goal" can
      silently preempt a CLI `goto_room.py` goal if both are sent around the same time.
      **DONE 2026-07-07**: `recovery_manager_node` now tries a direct creep+turn nudge on
      `cmd_vel_safe` (bypassing Nav2's Spin/BackUp, which abort in tight doorways) BEFORE
      falling back to Nav2's BackUp+Spin actions. Params `nudge_linear_speed`/`nudge_angular_speed`
      (0.10/0.10)/`nudge_duration` (1.2s). Code-complete, builds + unit tests pass — **not yet
      HW-tested at an actual doorway pinch-point**.
- [ ] **Residual final-approach stall (~0.23 m short near walls).** Separate from the spin bug;
      caused by circumscribed 0.344 > inflation 0.30 making CostCritic over-conservative near
      walls. Likely fix: slightly larger `xy_goal_tolerance` — do NOT tighten footprint (collision
      risk) or raise inflation (starts sealing the ~0.78 m doorways). Confirm on HW.
      **2026-07-09: bumped `xy_goal_tolerance` 0.25→0.30** (0.25 gave only ~2 cm margin over the
      ~0.23 m stall). Code change only — **still needs HW confirm** that goals near walls now
      complete without the endless final-align spin.
- [ ] **MPPI controller — live validation.** RPP→MPPI swap (commit chain, `project_robot_mppi_controller`)
      untested on hardware beyond the spin fix.
- [ ] **Orchestration end-to-end.** task_planner → mission_executor → recovery async bugs fixed
      2026-07-03 (8000fd4); never run as a full mission end-to-end live. Give one compound spoken
      goal and watch plan → navigate → act → report unattended.
- [ ] If a goto to **domatio_mbamba** (centre clearance 0.25 m) or **toualeta** (0.30 m) stalls,
      nudge the goal point toward the room centre — the taught points sit close to a wall.

## ⚙️ Hardware / reliability

- [ ] **Mini PC abrupt shutdowns.** 3 crashes 2026-06-25, all on wall power, ~15-17 min uptime.
      AC brick / cable suspected — confirm and replace. See `project_npu_minipc_power`.
- [ ] **MT7902 internal WiFi.** No mainline driver yet; running on a USB dongle. Wait for the
      official driver, don't install out-of-tree.

## 🎙️ Voice UX (software-only, no robot motion)

- [x] **TTS self-echo / barge-in gate.** DONE — `tts_node` publishes `tts/speaking`
      (Bool) around playback (+0.3 s reverb tail); `wake_word_node` and `stt_node`
      run a shared `SpeakingGate` (`home_robot/voice_gate.py`) and drop mic input
      while the robot speaks, so its own voice can't trip the wake word or start a
      recording. Unit + wiring tests in `tests/test_voice_gate.py`. Params:
      `suppress_on_tts`, `tts_release_tail`, `speaking_tail`. Not yet HW-heard.
- [x] **Wake-word false-triggers.** DONE 2026-07-05 — root cause was the wake word
      being a single syllable ("Μαξ"), too few phonemes to separate from everyday
      speech. Lengthened to the compound phrase **"Ρομπότ Μαξ" / "Robot Max"**; plain
      "Μαξ" and plain "Ρομπότ" trained in as heavily-weighted hard negatives so only
      the two together fire. Retrained (`generate_data.py` phrase lists updated),
      held-out eval: pos mean 0.91, neg mean 0.002 (worst "Μαξ"-alone 0.35 < 0.5
      threshold), noise 0.000. Deployed to `config/models/max.onnx`, threshold kept at
      0.5. NOT yet HW-heard on the XVF3800 — confirm live, and optionally record real
      "Ρομπότ Μαξ" positives via `record_real.py` if real speech scores low.

## ✅ Settled / parked (no action)

- **Spin bug root cause** — found & fixed in sim (see above; HW re-test is the only remainder).
- **Footprint vs kela3 doorways** — verified offline, robot fits through every door at inscribed
      and inflation radius; no change needed.
- **IMU translation x/y/z** — left as-is by decision (exact IMU chassis position not needed).
- **Wake word** — "Μαξ" → "Ρομπότ Μαξ" 2026-07-05 to kill false-triggers (see
  Voice UX above), then → **"Έι ρομπότ" / "Hey robot"** 2026-07-25 by user
  request (new model, `training/wake_word_hey_robot/`). Retraining also fixed a
  data bug that had been hurting every model: clips longer than the 2s window
  were left-truncated but kept their label, teaching "ρομπότ" alone as a
  positive. Held-out eval pos min 0.609 / mean 0.991, neg max 0.382.
  HW-heard confirmation still pending.
- **YOLO NPU quantization** — `yolo11n_int8.onnx` left as-is, not integrating; YOLO stays on
      iGPU/ROCm (~63 fps).
- **Vision Q&A** — stays on Gemini cloud; NPU/Qwen3-VL path not being revisited.
