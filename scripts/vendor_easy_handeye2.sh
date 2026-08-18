#!/usr/bin/env bash
# Vendors easy_handeye2 (marcoesposito1988/easy_handeye2, master branch — ROS2 port,
# 0.5.0, ament_python, no distro pin) into this workspace's src/, for eye_on_base
# hand-eye calibration between the body-mounted D435 and the RoArm-M3's arm_base.
#
# Tracker-agnostic: reads tf directly, so it reuses the existing apriltag_node +
# gripper_tag (config/apriltag.yaml) instead of needing aruco_ros. See
# home_robot/launch/handeye_calibrate.launch.py for the wired-up frame names.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo ">>> cloning marcoesposito1988/easy_handeye2 (master) …"
git clone --depth 1 --branch master https://github.com/marcoesposito1988/easy_handeye2.git "$TMP/easy_handeye2"

for pkg in easy_handeye2 easy_handeye2_msgs; do
  echo ">>> vendoring $pkg -> $SRC_DIR/$pkg"
  rm -rf "${SRC_DIR:?}/$pkg"
  cp -r "$TMP/easy_handeye2/$pkg" "$SRC_DIR/$pkg"
  rm -rf "$SRC_DIR/$pkg/.git"
done

echo ">>> done. Now build:"
echo "    cd $(dirname "$SRC_DIR") && rosdep install -iyr --from-paths src/easy_handeye2 src/easy_handeye2_msgs && colcon build --packages-select easy_handeye2_msgs easy_handeye2 --symlink-install"
