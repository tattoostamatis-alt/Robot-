#!/bin/bash
# Flatten the mini PC's power draw, for running off the portable power station.
#
# The machine cuts out by itself on the power station and never on wall power
# (user, 2026-08-04 — see project_npu_minipc_power). That is the inverter, not
# the PC: either an idle-shutoff when the draw falls too low, or a trip on a
# current spike. Software cannot fix an inverter, but it can stop handing it
# either extreme.
#
# The lever is the frequency BAND, not the ceiling. Default is 0.62-2.9 GHz per
# core, i.e. an 18 GHz spread across eight cores, and the summed clock was
# measured swinging 4.97 -> 17.52 GHz with jumps of 10.78 GHz per 100 ms on an
# IDLE robot. Narrowing the band cuts both ends at once: the floor stops the
# machine falling into the draw an idle-shutoff watches for, and the ceiling
# stops the spikes that trip an overcurrent cutoff.
#
# Measured on this machine, summed clock over eight cores, 45 s each:
#
#   profile          band          mean      swing   worst jump   sd
#   off (default)  0.62-2.9 GHz   9.58 GHz  12.54     10.78      2.51
#   eco            1.2-2.0 GHz   11.75 GHz   5.59      4.60      1.19
#   flat           1.6-2.2 GHz   13.94 GHz   3.68      3.54      0.70
#
# `flat` is the steadiest and draws the most on average; `eco` gives up half
# the smoothing for a much smaller idle cost. Neither is free — a narrow band
# means the CPU never drops to its low-power floor.
#
#   power_smooth.sh status
#   power_smooth.sh eco | flat | off
#
# Uses the existing NOPASSWD helper /usr/local/bin/cpufreq-set, so it needs no
# install of its own. Nothing here survives a reboot: disable-cpu-boost.service
# re-applies its own cap at every boot by design, and this script is meant to
# be switched on for a power-station session and off again afterwards.

set -uo pipefail

HELPER=/usr/local/bin/cpufreq-set
# Threads for reading (16 with SMT), PHYSICAL cores for writing (8). The helper
# takes a physical core id and sets both its siblings — passing it a thread
# number silently addresses the wrong half of the machine, and asking it for
# core 8..15 fails outright.
THREADS=$(ls -d /sys/devices/system/cpu/cpu[0-9]* 2>/dev/null | wc -l)
CORES=$(cat /sys/devices/system/cpu/cpu[0-9]*/topology/core_id 2>/dev/null \
        | sort -un | tr '\n' ' ')

# ‼️ `off` restores 623377-2900000, which is what this machine was found
# running, NOT the 3900000 in disable-cpu-boost.service. Something lowered the
# cap to 2.9 GHz after that unit was written, and restoring 3.9 here would
# quietly raise the ceiling on a machine whose owner deliberately capped it.
declare -A PROFILES=(
  [off]="623377 2900000"
  [eco]="1200000 2000000"
  [flat]="1600000 2200000"
)

usage() { echo "usage: power_smooth.sh {status|off|eco|flat}" >&2; exit 2; }

cmd_status() {
  local total_min=0 total_max=0 n=0
  for c in $(seq 0 $((THREADS - 1))); do
    local d=/sys/devices/system/cpu/cpu$c/cpufreq
    [ -r "$d/scaling_min_freq" ] || continue
    total_min=$((total_min + $(cat "$d/scaling_min_freq")))
    total_max=$((total_max + $(cat "$d/scaling_max_freq")))
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] || { echo "no cpufreq available" >&2; exit 1; }
  local mn=$((total_min / n)) mx=$((total_max / n))
  printf 'band per core: %.2f - %.2f GHz  (%d threads)\n' \
         "$(echo "$mn/1000000" | bc -l)" "$(echo "$mx/1000000" | bc -l)" "$n"
  # Name the profile if it matches one, so "what is it set to" has an answer
  # that is not four decimal places.
  for name in "${!PROFILES[@]}"; do
    read -r pmin pmax <<< "${PROFILES[$name]}"
    if [ "$mn" = "$pmin" ] && [ "$mx" = "$pmax" ]; then
      echo "profile: $name"; return
    fi
  done
  echo "profile: custom"
}

cmd_apply() {
  local name="$1"
  read -r min max <<< "${PROFILES[$name]}"
  local failed=0 tried=0
  for c in $CORES; do
    tried=$((tried + 1))
    # The helper takes a physical core id and sets both SMT siblings; it also
    # validates against each core's own hardware limits, which matters here
    # because the E-cores top out at 3.49 GHz against the P-cores' 5.09.
    sudo -n "$HELPER" "$c" "$min" "$max" >/dev/null 2>&1 || failed=$((failed + 1))
  done
  if [ "$failed" -gt 0 ]; then
    echo "‼️ $failed of $tried cores refused — is $HELPER installed with NOPASSWD?" >&2
    exit 1
  fi
  echo "applied '$name'"
  cmd_status
}

case "${1:-status}" in
  status) cmd_status ;;
  off|eco|flat) cmd_apply "$1" ;;
  *) usage ;;
esac
