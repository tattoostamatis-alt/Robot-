#!/bin/bash
# Power-cycle one USB device by name, so a wedged sensor is recoverable from
# the other side of the house instead of by walking over and replugging it.
#
# Why not uhubctl (which is installed): it only drives hubs that report
# per-port power switching, and on this machine the RealSense is plugged
# straight into a root hub — the one device with a documented habit of going
# blind mid-session (see project_robot_realsense_startup_segfault) is the one
# uhubctl cannot touch. The kernel's own per-port `disable` attribute covers
# every port on every hub, root hubs included, so that is what this uses.
#
# ‼️ Ports are resolved from the device, never hardcoded. USB enumeration on
# this machine is NOT stable across reboots or replugs — the IMU's path alone
# has moved three times (5-1.2 -> 3-4 -> 1-1.1, see 99-robot-devices.rules),
# and a hardcoded port number would eventually power-cycle the wrong device.
#
#   usb_power.sh list                 what is switchable, and where
#   usb_power.sh cycle lidar          off, wait, on
#   usb_power.sh cycle camera 3       off, wait 3 s, on
#
# Needs root to write the sysfs attribute. Install as root-owned and grant it
# NOPASSWD in sudoers — see scripts/install_usb_power.sh.

set -u

SYS=/sys/bus/usb/devices

# name -> how to find its USB device directory. Kept as matchers rather than
# paths for the reason in the header.
find_dev() {
  case "$1" in
    lidar)  by_tty /dev/lidar ;;
    arm)    by_tty /dev/arm ;;
    imu)    by_tty /dev/imu ;;
    roomba) by_tty /dev/roomba ;;
    camera) by_id 8086 0b07 ;;      # Intel RealSense D435
    mic)    by_id 2886 001a ;;      # reSpeaker XVF3800 4-Mic Array
    *)      echo "unknown device: $1" >&2; return 1 ;;
  esac
}

by_tty() {
  local node="$1" real
  real=$(readlink -f "$node" 2>/dev/null) || return 1
  [ -n "$real" ] || return 1
  local d="/sys/class/tty/$(basename "$real")/device"
  d=$(readlink -f "$d" 2>/dev/null) || return 1
  # Walk up until the directory looks like a USB device (has idVendor).
  while [ "$d" != "/" ]; do
    [ -f "$d/idVendor" ] && { echo "$d"; return 0; }
    d=$(dirname "$d")
  done
  return 1
}

by_id() {
  local vid="$1" pid="$2" d
  for d in "$SYS"/[0-9]*-[0-9]*; do
    [ -f "$d/idVendor" ] || continue
    if [ "$(cat "$d/idVendor")" = "$vid" ] && [ "$(cat "$d/idProduct")" = "$pid" ]; then
      echo "$d"; return 0
    fi
  done
  return 1
}

# /sys/.../5-1.1 -> the `disable` attribute of the port it hangs off.
#   5-1.1 -> hub 5-1,  port 1
#   4-2   -> hub usb4, port 2   (straight into a root hub)
port_attr() {
  local devdir="$1" name hub port
  name=$(basename "$devdir")
  if [[ "$name" == *.* ]]; then
    hub="${name%.*}"
    port="${name##*.}"
  else
    hub="usb${name%%-*}"
    port="${name##*-}"
  fi
  local hubdir
  if [ "$hub" = "usb${name%%-*}" ]; then
    hubdir="$SYS/${name%%-*}-0:1.0"
  else
    hubdir="$SYS/${hub}:1.0"
  fi
  echo "$hubdir/${hub}-port${port}/disable"
}

cmd_list() {
  printf '%-8s %-22s %-10s %s\n' NAME DEVICE PORT SWITCHABLE
  for n in lidar arm imu roomba camera mic; do
    local d attr state
    if ! d=$(find_dev "$n" 2>/dev/null); then
      printf '%-8s %-22s %-10s %s\n' "$n" '-' '-' 'not present'
      continue
    fi
    attr=$(port_attr "$d")
    if [ -w "$attr" ]; then state='yes (writable)'
    elif [ -e "$attr" ]; then state='yes (needs root)'
    else state='NO'; fi
    printf '%-8s %-22s %-10s %s\n' "$n" "$(basename "$d")" \
           "$(basename "$(dirname "$attr")")" "$state"
  done
}

cmd_cycle() {
  local name="$1" wait="${2:-2}" d attr
  d=$(find_dev "$name") || { echo "cannot find $name" >&2; exit 1; }
  attr=$(port_attr "$d")
  [ -e "$attr" ] || { echo "no port control for $name ($attr)" >&2; exit 1; }
  if [ ! -w "$attr" ]; then
    echo "need root to write $attr" >&2; exit 1
  fi
  echo "cycling $name ($(basename "$d")) via $attr, ${wait}s off"
  echo 1 > "$attr" || exit 1
  sleep "$wait"
  echo 0 > "$attr" || exit 1
  # The device has to re-enumerate before anything can open it again; without
  # the wait a caller that immediately restarts a driver races the kernel.
  sleep 3
  echo "done — $name should be re-enumerating"
}

case "${1:-list}" in
  list)  cmd_list ;;
  cycle) shift; [ $# -ge 1 ] || { echo "usage: usb_power.sh cycle <name> [seconds]" >&2; exit 2; }
         cmd_cycle "$@" ;;
  *)     echo "usage: usb_power.sh {list|cycle <name> [seconds]}" >&2; exit 2 ;;
esac
