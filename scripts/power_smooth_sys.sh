#!/usr/bin/env bash
# power-smooth — hold the mini PC's power draw steady, machine-wide.
#
# Installed to /usr/local/sbin/power-smooth by scripts/install_power_smooth.sh
# and applied at every boot by power-smooth.service. This is the system-level
# tool: it owns the whole box, not just the robot stack, and it survives a
# reboot. scripts/power_smooth.sh (the robot-side, dashboard-facing script)
# forwards to it.
#
# WHY
# ---
# The machine cuts out by itself on the portable power station and never on
# wall power. The station is an Anker SOLIX C300 DC — 288 Wh, and DC-only:
# there is NO inverter in it. The mini PC is fed over USB-C Power Delivery from
# one of its 140 W ports. So the old "the inverter trips" theory is wrong in
# its mechanism, though right in its conclusion: what a PD source dislikes is a
# load that lurches. A fast step in current makes the port's own protection
# trip or forces a PD renegotiation, and a renegotiation drops the rail for
# long enough to kill an unbuffered mini PC instantly. That is the one story
# that also explains the boots that died in 0 s (POST inrush) and the "blue
# screen" that only unplugging fixes (the sink stuck in a bad PD contract).
#
# Software cannot add a capacitor. What it can do is stop handing the port
# sudden steps. Every knob below exists to cut dP/dt, not average watts.
#
# WHAT IT ACTUALLY TOUCHES
# ------------------------
#   cpu band   scaling_min_freq/scaling_max_freq per policy. The lever is the
#              BAND, not the ceiling: a narrow band cannot lurch. Measured on
#              this machine the default 0.62-2.9 GHz band let the summed clock
#              jump 10.78 GHz in 100 ms on an IDLE robot.
#   boost      off outside `off`. Boost is by construction a sudden step.
#   epp        amd-pstate's energy_performance_preference -> `power`, which
#              makes the governor lazier about ramping. Without this the band
#              still gets traversed at full speed.
#   igpu       `flat` only: pins the iGPU below its 3000 MHz DPM level. That
#              level is the single biggest step this SoC can take, but capping
#              it costs real GPU throughput (YOLO on ROCm, RViz), so `eco`
#              leaves the GPU alone.
#
# The APU reports its own package power as PPT in the amdgpu hwmon
# (power1_average). `power-smooth measure` reads it, so every claim here is
# checkable in watts rather than guessed from frequencies.
#
#   power-smooth status
#   power-smooth off | eco | flat
#   power-smooth measure [seconds]
#
# ‼️ Not free: a narrow band means the CPU never drops to its low-power floor,
# so average draw goes UP while the swing goes down. That is the trade being
# made on purpose — a station that stays on and eats more is better than one
# that cuts power mid-write.

set -uo pipefail

DEFAULTS=/etc/default/power-smooth
CPU=/sys/devices/system/cpu
GPU_DEV=$(echo /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | head -1)
GPU_DEV=${GPU_DEV%/pp_dpm_sclk}

# ‼️ `off` restores 2900000, which is what this machine was found running, NOT
# the 3900000 in disable-cpu-boost.service. Something lowered the cap to
# 2.9 GHz after that unit was written and restoring 3.9 here would quietly
# raise the ceiling on a machine whose owner deliberately capped it.
#
# ‼️‼️ Turning boost off does not merely stop opportunistic clocking — on
# amd-pstate it lowers `cpuinfo_max_freq` ITSELF, measured here from 5.09/3.49
# GHz (P/E) down to 2.00/1.86 GHz. So with boost off, NO profile can have a
# ceiling above 2.0 GHz: ask for 2.2 and the kernel silently clamps to 1.86-2.0.
# An earlier version asked for 2.2 in `flat` and 2.0 in `eco` and got the exact
# same machine both times, which is why they measured identically (11.68 vs
# 11.72 W mean, same 12.97 W swing) — the difference was never applied.
# The two profiles are therefore separated by their FLOOR and by the iGPU,
# not by their ceiling.
#
#            min_khz  max_khz  boost  epp                  gpu_cap
declare -A PROFILES=(
  [off]="623377  2900000  1  balance_performance  no"
  [eco]="1200000 2000000  0  power                no"
  [flat]="1600000 2000000  0  power                yes"
)

die() { echo "power-smooth: $*" >&2; exit 1; }

need_root() { [ "$(id -u)" -eq 0 ] || die "needs root (sudo power-smooth $*)"; }

# --- cpu ---------------------------------------------------------------

# Per-policy hardware limits differ: this is a heterogeneous P/E part. P-cores
# top out at ~5.09 GHz, E-cores at only ~3.49 GHz, so a band that is legal on
# one is refused by the other. Clamp per policy instead of assuming.
apply_cpu() {
  local want_min=$1 want_max=$2 n=0
  for pol in "$CPU"/cpufreq/policy*; do
    [ -w "$pol/scaling_max_freq" ] || continue
    local hw_min hw_max lo hi
    hw_min=$(cat "$pol/cpuinfo_min_freq") || continue
    hw_max=$(cat "$pol/cpuinfo_max_freq") || continue
    lo=$want_min; hi=$want_max
    [ "$lo" -lt "$hw_min" ] && lo=$hw_min
    [ "$hi" -gt "$hw_max" ] && hi=$hw_max
    [ "$lo" -gt "$hi" ] && lo=$hi
    # Order matters: a min above the current max is silently refused. Drop the
    # floor first, set the ceiling, then raise the floor.
    echo "$hw_min" > "$pol/scaling_min_freq" 2>/dev/null
    echo "$hi"     > "$pol/scaling_max_freq" 2>/dev/null
    echo "$lo"     > "$pol/scaling_min_freq" 2>/dev/null
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] || die "no writable cpufreq policies"
  echo "  cpu    : $n policies -> $(bc <<< "scale=2;$want_min/1000000")-$(bc <<< "scale=2;$want_max/1000000") GHz"
}

apply_boost() {
  local want=$1
  if [ -w "$CPU/cpufreq/boost" ]; then
    echo "$want" > "$CPU/cpufreq/boost" 2>/dev/null \
      && echo "  boost  : $want" || echo "  boost  : refused"
  else
    echo "  boost  : n/a"
  fi
}

apply_epp() {
  local want=$1 n=0
  for pol in "$CPU"/cpufreq/policy*; do
    [ -w "$pol/energy_performance_preference" ] || continue
    echo "$want" > "$pol/energy_performance_preference" 2>/dev/null && n=$((n + 1))
  done
  [ "$n" -gt 0 ] && echo "  epp    : $want ($n policies)" || echo "  epp    : n/a"
}

# --- igpu --------------------------------------------------------------

# pp_dpm_sclk on this APU lists three states (600 / ~780 / 3000 MHz). In manual
# mode you are supposed to write the indices you want to allow, and dropping
# the top one would remove the biggest single step the chip can make.
#
# ‼️ Measured: this APU REFUSES that write ("cap refused" — many APUs return
# EINVAL for per-level sclk masking, it is a discrete-GPU feature). So try it,
# and when it is refused fall back to `low`, which the same interface does
# honour and which pins the GPU to its bottom state. `low` is blunter — it
# costs real GPU throughput — but a refused cap that reports success would be
# worse than either.
apply_gpu() {
  local cap=$1 lvl err
  [ -n "$GPU_DEV" ] && [ -w "$GPU_DEV/power_dpm_force_performance_level" ] \
    || { echo "  igpu   : n/a"; return; }
  if [ "$cap" != yes ]; then
    echo auto > "$GPU_DEV/power_dpm_force_performance_level" 2>/dev/null \
      && echo "  igpu   : auto"
    return
  fi
  lvl=$(awk -F: '{print $1}' "$GPU_DEV/pp_dpm_sclk" 2>/dev/null | head -n -1 | tr '\n' ' ')
  if [ -n "${lvl// /}" ]; then
    echo manual > "$GPU_DEV/power_dpm_force_performance_level" 2>/dev/null
    if err=$( { echo "$lvl" > "$GPU_DEV/pp_dpm_sclk"; } 2>&1 ); then
      echo "  igpu   : capped to DPM levels [$lvl] (top state withheld)"
      return
    fi
  fi
  # Fall back to the level the interface does accept.
  if echo low > "$GPU_DEV/power_dpm_force_performance_level" 2>/dev/null; then
    echo "  igpu   : per-level cap refused${err:+ ($err)}; forced to 'low' instead"
  else
    echo auto > "$GPU_DEV/power_dpm_force_performance_level" 2>/dev/null
    echo "  igpu   : cannot be capped on this APU, left auto"
  fi
}

# --- commands ----------------------------------------------------------

cmd_apply() {
  local name=$1
  [ -n "${PROFILES[$name]+x}" ] || die "unknown profile '$name' (off|eco|flat)"
  need_root "$name"
  read -r pmin pmax pboost pepp pgpu <<< "${PROFILES[$name]}"
  echo "power-smooth: applying '$name'"
  # ‼️ boost FIRST. It moves cpuinfo_max_freq itself (5.09 -> 2.00 GHz when
  # turned off), so the band has to be clamped against the limits that will
  # actually be in force, not the ones from before the switch. Doing this the
  # other way round is what made `flat` and `eco` come out identical.
  apply_boost "$pboost"
  apply_cpu "$pmin" "$pmax"
  apply_epp "$pepp"
  apply_gpu "$pgpu"
  # Remember it so the boot unit reapplies the same thing. Best-effort: a
  # read-only /etc must not make the profile itself fail.
  if [ -w "$(dirname "$DEFAULTS")" ] || [ -w "$DEFAULTS" ]; then
    printf '# Written by power-smooth. Profile applied at boot.\nPROFILE=%s\n' \
      "$name" > "$DEFAULTS" 2>/dev/null && echo "  saved  : $DEFAULTS (survives reboot)"
  fi
}

cmd_status() {
  local tmin=0 tmax=0 n=0 mn mx
  for pol in "$CPU"/cpufreq/policy*; do
    [ -r "$pol/scaling_min_freq" ] || continue
    tmin=$((tmin + $(cat "$pol/scaling_min_freq")))
    tmax=$((tmax + $(cat "$pol/scaling_max_freq")))
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] || die "no cpufreq"
  mn=$((tmin / n)); mx=$((tmax / n))
  printf 'cpu band : %.2f - %.2f GHz  (%d policies)\n' \
    "$(bc -l <<< "$mn/1000000")" "$(bc -l <<< "$mx/1000000")" "$n"
  echo "boost    : $(cat "$CPU/cpufreq/boost" 2>/dev/null || echo n/a)"
  echo "epp      : $(cat "$CPU/cpufreq/policy0/energy_performance_preference" 2>/dev/null || echo n/a)"
  if [ -n "$GPU_DEV" ] && [ -r "$GPU_DEV/power_dpm_force_performance_level" ]; then
    echo "igpu     : $(cat "$GPU_DEV/power_dpm_force_performance_level") | $(tr '\n' ' ' < "$GPU_DEV/pp_dpm_sclk")"
  fi
  local ppt
  ppt=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_average 2>/dev/null | head -1)
  [ -n "$ppt" ] && printf 'ppt now  : %.2f W\n' "$(bc -l <<< "$ppt/1000000")"

  # Name the profile if the band matches one, so "what is it set to" has an
  # answer that is not four decimal places.
  #
  # ‼️ Compare against the CLAMPED band, not the nominal one. With boost off
  # the kernel caps every policy at its own cpuinfo_max (2.00 GHz on a P-core,
  # 1.86 on an E-core), so a profile asking for 2.00 reads back as 1.93 average
  # and a naive comparison calls a correctly applied profile "custom".
  local saved=''
  [ -r "$DEFAULTS" ] && saved=$(sed -n 's/^PROFILE=//p' "$DEFAULTS" 2>/dev/null)
  for name in "${!PROFILES[@]}"; do
    read -r pmin pmax _ _ _ <<< "${PROFILES[$name]}"
    local emin=0 emax=0 k=0
    for pol in "$CPU"/cpufreq/policy*; do
      [ -r "$pol/cpuinfo_max_freq" ] || continue
      local hmin hmax lo hi
      hmin=$(cat "$pol/cpuinfo_min_freq"); hmax=$(cat "$pol/cpuinfo_max_freq")
      lo=$pmin; hi=$pmax
      [ "$lo" -lt "$hmin" ] && lo=$hmin
      [ "$hi" -gt "$hmax" ] && hi=$hmax
      [ "$lo" -gt "$hi" ] && lo=$hi
      emin=$((emin + lo)); emax=$((emax + hi)); k=$((k + 1))
    done
    [ "$k" -gt 0 ] || continue
    if [ "$mn" = "$((emin / k))" ] && [ "$mx" = "$((emax / k))" ]; then
      echo "profile  : $name"
      [ -n "$saved" ] && [ "$saved" != "$name" ] \
        && echo "at boot  : $saved  ‼️ differs from what is running now"
      # ‼️ Explicit 0. A bare `return` hands back the status of the last
      # command, and that last command is the `!=` test — which is FALSE, and
      # so returns 1, exactly when the saved profile matches the running one.
      # That is the normal case, so `status` was reporting failure whenever
      # everything was fine, and the dashboard would have rendered an error.
      return 0
    fi
  done
  echo "profile  : custom"
  [ -n "$saved" ] && echo "at boot  : $saved"
  return 0
}

# Sample PPT and report the swing and the worst step. Needs no root: the hwmon
# node is world-readable, so the user can check the machine any time.
cmd_measure() {
  local dur=${1:-30} hw
  hw=$(echo /sys/class/drm/card*/device/hwmon/hwmon*/power1_average | head -1)
  [ -r "$hw" ] || die "no PPT sensor"
  python3 - "$hw" "$dur" <<'PY'
import statistics, sys, time
path, dur = sys.argv[1], float(sys.argv[2])
s, end = [], time.monotonic() + dur
with open(path) as f:
    while time.monotonic() < end:
        f.seek(0); s.append(int(f.read()) / 1e6); time.sleep(0.1)
j = [abs(b - a) for a, b in zip(s, s[1:])] or [0.0]
print(f"PPT over {dur:.0f}s: mean {statistics.fmean(s):.2f} W  "
      f"min {min(s):.2f}  max {max(s):.2f}  swing {max(s)-min(s):.2f}  "
      f"worst step {max(j):.2f} W/100ms")
PY
}

case "${1:-status}" in
  status)  cmd_status ;;
  measure) cmd_measure "${2:-30}" ;;
  off|eco|flat) cmd_apply "$1" ;;
  boot)
    # Used by power-smooth.service: apply whatever /etc/default says, quietly
    # doing nothing if the file says off or is missing.
    p=off
    [ -r "$DEFAULTS" ] && . "$DEFAULTS" 2>/dev/null && p=${PROFILE:-off}
    cmd_apply "$p"
    ;;
  *) echo "usage: power-smooth {status|off|eco|flat|measure [sec]|boot}" >&2; exit 2 ;;
esac
