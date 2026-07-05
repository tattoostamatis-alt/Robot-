# home_robot — TODO

Status as of 2026-07-05. Items grouped by whether they need the robot powered on.

## 🧪 Needs the robot live (do these together in one session)

- [ ] **Spin/oscillation fix — HW re-test on kela3.** Fixed in sim (commit 18f6632: MPPI
      `ax_max` 0.25→2.0 was capping the first cmd to 0.0125 m/s → recovery loops). Verified in
      Gazebo; confirm on real hardware that an RViz goal now drives a smooth arc instead of
      spinning in place. See `project_robot_nav_path_wall_bug`.
- [ ] **`goto_room` HW test.** kela3 remap done (locations re-taught + room_mask). Never driven
      live. Doorway passability already verified offline (all 6 rooms connected at inscribed 0.20
      and inflation 0.30 — footprint/inflation are correct, no change needed).
- [ ] **Residual final-approach stall (~0.23 m short near walls).** Separate from the spin bug;
      caused by circumscribed 0.344 > inflation 0.30 making CostCritic over-conservative near
      walls. Likely fix: slightly larger `xy_goal_tolerance` — do NOT tighten footprint (collision
      risk) or raise inflation (starts sealing the ~0.78 m doorways). Confirm on HW.
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
- **Wake word** — changed "Μαξ" → compound "Ρομπότ Μαξ" 2026-07-05 to kill
  false-triggers (see Voice UX above); HW-heard confirmation pending.
- **YOLO NPU quantization** — `yolo11n_int8.onnx` left as-is, not integrating; YOLO stays on
      iGPU/ROCm (~63 fps).
- **Vision Q&A** — stays on Gemini cloud; NPU/Qwen3-VL path not being revisited.
