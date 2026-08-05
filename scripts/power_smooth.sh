#!/bin/bash
# Robot-side front end for the machine-wide power smoother.
#
#   power_smooth.sh status
#   power_smooth.sh eco | flat | off
#
# This used to narrow the CPU frequency band by itself, through the
# cpufreq-set NOPASSWD helper. That covered one knob out of four and, being
# user-level, forgot everything at reboot — which is no good for a machine
# that is being cut off mid-write. The real tool is now
# /usr/local/sbin/power-smooth (source: power_smooth_sys.sh): it also drops
# boost, sets the amd-pstate EPP, can cap the iGPU's top DPM state, and saves
# the choice so power-smooth.service reapplies it at every boot.
#
# Everything here is a thin forward, so the web dashboard's Σύστημα tab and the
# command line drive exactly the same thing.
#
# Why any of this exists: the mini PC cuts out on the Anker SOLIX C300 DC and
# never on wall power. That station is DC-only — the PC hangs off a 140 W
# USB-C PD port, with no inverter anywhere. A PD source is not upset by average
# draw, it is upset by sudden steps, and a renegotiation drops the rail long
# enough to kill an unbuffered mini PC outright. Every profile below exists to
# cut dP/dt. See project_npu_minipc_power.

set -uo pipefail

SYSTOOL=/usr/local/sbin/power-smooth

if [ ! -x "$SYSTOOL" ]; then
  cat >&2 <<EOF
power_smooth: $SYSTOOL is not installed.

Run once, on the mini PC:

    sudo ~/robot_ws/src/home_robot/scripts/install_power_smooth.sh

Until then no profile can be applied. (Deliberately not falling back to the
old CPU-only path: a partial fix that looks like the whole one is how you end
up trusting a profile that never capped the iGPU.)
EOF
  exit 1
fi

case "${1:-status}" in
  status|measure)
    # Read-only, no privilege needed.
    exec "$SYSTOOL" "$@"
    ;;
  off|eco|flat)
    if ! sudo -n "$SYSTOOL" "$1" 2>/dev/null; then
      echo "‼️ sudo -n $SYSTOOL refused — re-run install_power_smooth.sh (it writes the sudoers entry)" >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: power_smooth.sh {status|off|eco|flat|measure [sec]}" >&2
    exit 2
    ;;
esac
