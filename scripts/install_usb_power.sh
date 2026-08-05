#!/bin/bash
# One-time root install for scripts/usb_power.sh.
#
#   sudo scripts/install_usb_power.sh
#
# ‼️ The script is COPIED to a root-owned location rather than granted sudo
# where it lives. A NOPASSWD sudoers entry pointing at a file the invoking user
# can edit is a root shell with extra steps — anyone who can write
# ~/robot_ws/src/... could put anything in it and run it as root. /usr/local/sbin
# is root-owned, so the copy is what gets the privilege.
#
# Re-run after changing usb_power.sh; the copy does not follow the original.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "run me with sudo" >&2
  exit 1
fi

SRC="$(cd "$(dirname "$0")" && pwd)/usb_power.sh"
DST=/usr/local/sbin/robot-usb-power
OWNER="${SUDO_USER:-$(logname 2>/dev/null || echo dimi)}"

install -o root -g root -m 0755 "$SRC" "$DST"
echo "installed $DST"

SUDOERS=/etc/sudoers.d/robot-usb-power
cat > "$SUDOERS" <<EOF
# Lets the robot stack power-cycle a wedged USB sensor without a password.
# Installed by home_robot/scripts/install_usb_power.sh.
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

echo
echo "check it:   sudo -n $DST list"
echo "try it:     sudo -n $DST cycle mic"
