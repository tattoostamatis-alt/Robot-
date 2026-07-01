#!/usr/bin/env bash
# Vendors the RoArm-M3 description + MoveIt config from Waveshare's roarm_ws into
# this workspace's src/, for the MoveIt 3D control stack (launch/arm_moveit.launch.py
# + home_robot/nodes/arm_moveit_bridge.py).
#
# Waveshare only ships a ROS2 *Humble* branch; the URDF/meshes are version-agnostic
# and reused as-is, and the MoveIt config runs on Jazzy after one tweak: the IKFast
# C++ plugin (Humble-built) is swapped for the stock KDL solver, applied below.
#
# We deliberately do NOT vendor roarm_driver — arm_driver.py owns /dev/arm and
# MoveIt executes through arm_moveit_bridge.py instead.
set -euo pipefail

# Workspace src/ is two levels up from this repo (…/src/home_robot/scripts).
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo ">>> cloning waveshareteam/roarm_ws (ros2-humble) …"
git clone --depth 1 --branch ros2-humble https://github.com/waveshareteam/roarm_ws.git "$TMP/roarm_ws"

MAIN="$TMP/roarm_ws/src/roarm_main"
for pkg in roarm_description roarm_moveit; do
  echo ">>> vendoring $pkg -> $SRC_DIR/$pkg"
  rm -rf "${SRC_DIR:?}/$pkg"
  cp -r "$MAIN/$pkg" "$SRC_DIR/$pkg"
  rm -rf "$SRC_DIR/$pkg/.git"
done

echo ">>> patching kinematics.yaml IKFast -> KDL (Jazzy)"
cat > "$SRC_DIR/roarm_moveit/config/roarm_m3/kinematics.yaml" <<'YAML'
hand:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.05
  kinematics_solver_attempts: 3
YAML

echo ">>> done. Now build:"
echo "    cd $(dirname "$SRC_DIR") && colcon build --packages-select roarm_description roarm_moveit home_robot --symlink-install"
