#!/bin/bash
# One-time root install for the system-wide power smoother.
#
#   sudo scripts/install_power_smooth.sh
#
# Installs:
#   /usr/local/sbin/power-smooth        the tool (from power_smooth_sys.sh)
#   power-smooth.service                applies the saved profile at every boot
#   /etc/sudoers.d/power-smooth         so the robot/dashboard can switch it
#   /usr/local/sbin/power-blackbox      always-on PPT recorder
#   power-blackbox.service              so the next cut leaves evidence
#
# ‼️ The script is COPIED to a root-owned location rather than granted sudo
# where it lives. A NOPASSWD sudoers entry pointing at a file the invoking user
# can edit is a root shell with extra steps — anyone who can write
# ~/robot_ws/src/... could put anything in it and run it as root. /usr/local/sbin
# is root-owned, so the copy is what gets the privilege.
#
# Re-run after changing power_smooth_sys.sh; the copy does not follow the
# original.
#
# Installing does NOT change how the machine runs: the boot unit applies
# whatever /etc/default/power-smooth says, and that file starts out absent,
# which means `off`. Pick a profile afterwards with `sudo -n power-smooth eco`.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "run me with sudo" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
DST=/usr/local/sbin/power-smooth
UNIT=/etc/systemd/system/power-smooth.service
OWNER="${SUDO_USER:-$(logname 2>/dev/null || echo dimi)}"

install -o root -g root -m 0755 "$HERE/power_smooth_sys.sh" "$DST"
echo "installed $DST"

install -o root -g root -m 0644 "$HERE/power-smooth.service" "$UNIT"
echo "installed $UNIT"

SUDOERS=/etc/sudoers.d/power-smooth
cat > "$SUDOERS" <<EOF
# Lets the robot stack and the web dashboard switch the power profile without
# a password. Installed by home_robot/scripts/install_power_smooth.sh.
$OWNER ALL=(root) NOPASSWD: $DST
EOF
chmod 0440 "$SUDOERS"

# An invalid sudoers file locks sudo for everyone, so validate before leaving
# it in place.
if visudo -c -f "$SUDOERS" >/dev/null; then
  echo "installed $SUDOERS for $OWNER"
else
  rm -f "$SUDOERS"
  echo "sudoers snippet was invalid — removed, nothing changed" >&2
  exit 1
fi

# --- black box ---------------------------------------------------------
# Separate from the smoother on purpose: the smoother is a guess that needs
# proving, and this is what proves it. It records whether draw actually stepped
# before a cut, which no post-mortem so far has been able to say.
BB=/usr/local/sbin/power-blackbox
BB_UNIT=/etc/systemd/system/power-blackbox.service
install -o root -g root -m 0755 "$HERE/power_blackbox.py" "$BB"
install -o root -g root -m 0644 "$HERE/power-blackbox.service" "$BB_UNIT"
install -o root -g root -m 0644 /dev/null /var/log/power-blackbox.csv 2>/dev/null || true
echo "installed $BB and $BB_UNIT"

systemctl daemon-reload
systemctl enable power-smooth.service >/dev/null
echo "enabled power-smooth.service (applies the saved profile at boot)"
systemctl enable --now power-blackbox.service >/dev/null
echo "started power-blackbox.service -> /var/log/power-blackbox.csv"

echo
echo "state now:"
"$DST" status | sed 's/^/  /'
echo
echo "choose a profile:  sudo -n power-smooth eco     # or flat, or off"
echo "check the watts :  power-smooth measure 30"
echo "after a cut     :  power-blackbox --last 30"
