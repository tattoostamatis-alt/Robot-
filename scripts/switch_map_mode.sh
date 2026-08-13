#!/usr/bin/env bash
# Hot-swap between AMCL localization and slam_toolbox mapping WITHOUT
# restarting the rest of the stack (voice, camera, arm, dashboard, VNC,
# Foxglove, Nav2 planner/controller/costmaps). Used by map_session.sh's
# `new` and `switch`, which used to shell out to `robot stop` + `robot
# max`/`robot map` — a full teardown that took ~90s and dropped the voice
# pipeline on the way back up (see project_robot_new_map_half_stack memory).
#
#   switch_map_mode.sh slam                 # start mapping a NEW area
#   switch_map_mode.sh localize <map_name>  # localize against a saved map
#
# How: kill only the localization/SLAM node PROCESSES, by their exact ROS
# node name (matched on the `-r __node:=<name>` remap every Node action
# gets — precise, unlike matching on executable path: nav2_map_server is
# shared between /map_server and the keepout /filter_mask_server, which
# must NOT be touched here). The `ros2 launch` tree that originally started
# these does not try to respawn them — nav2's localization_launch.py
# defaults use_respawn to False, and bringup's slam_toolbox LifecycleNode
# never sets respawn=True either — so killing the process is enough, nothing
# races to bring it back. A standalone replacement (map_mode_localize.launch.py
# or map_mode_slam.launch.py) is then started detached, same pattern `robot
# max`/`robot map` use for the whole stack.
#
# Nav2's costmaps pick up the new /map + map->odom TF on their own — the
# static_layer subscribes with transient_local durability, so a fresh
# publisher's latched message reaches it with no restart needed on that side.
#
# ‼️ NOT `set -u`: same reason as map_session.sh — ROS's setup.bash reads
# AMENT_TRACE_SETUP_FILES without defining it first.
set -o pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAPS="$SRC_DIR/maps"
LOG=/tmp/map_session.log

log() { echo "[$(date +%H:%M:%S)] switch_map_mode: $*" >> "$LOG"; }

source /opt/ros/jazzy/setup.bash
source "$HOME/robot_ws/install/setup.bash"
export ROS_DOMAIN_ID=0

# Every node this script is allowed to touch — the localization/SLAM slice
# only. Order doesn't matter: killing a name with no matching process is a
# silent no-op, so the same superset is used going into either mode (a plain
# map switch while already localizing has no slam_toolbox to kill; mapping
# a second time in a row has no AMCL to kill).
LOCAL_NODES="map_server amcl lifecycle_manager_localization room_markers_node pose_saver_node global_localizer apriltag_node apriltag_relocalizer"
SLAM_NODES="slam_toolbox"

kill_nodes() {
  local name pids
  for name in "$@"; do
    pids=$(ps -eo pid,cmd --no-headers \
      | grep -E -- "--ros-args.*-r __node:=${name}( |\$)" \
      | grep -v grep | awk '{print $1}')
    [ -z "$pids" ] && continue
    log "  → σταματώ $name ($pids)"
    kill -TERM $pids 2>/dev/null || true
  done
}

wait_gone() {
  local name pids i
  for i in $(seq 1 10); do
    pids=""
    for name in "$@"; do
      pids="$pids $(ps -eo pid,cmd --no-headers \
        | grep -E -- "--ros-args.*-r __node:=${name}( |\$)" \
        | grep -v grep | awk '{print $1}')"
    done
    [ -z "${pids// /}" ] && return 0
    sleep 1
  done
  # Stragglers after 10s: SIGKILL, same escalation `robot stop` uses.
  log "  → $(echo $pids | wc -w) δεν τερμάτισαν, SIGKILL"
  kill -9 $pids 2>/dev/null || true
  sleep 1
}

case "${1:-}" in

  slam)
    log "→ ΧΑΡΤΟΓΡΑΦΗΣΗ (hot swap, το υπόλοιπο stack μένει πάνω)"
    kill_nodes $LOCAL_NODES $SLAM_NODES
    wait_gone $LOCAL_NODES $SLAM_NODES
    setsid nohup ros2 launch home_robot map_mode_slam.launch.py \
      >> "$LOG" 2>&1 < /dev/null &
    echo "mapping started (hot swap)"
    ;;

  localize)
    NAME="${2:-}"
    [ -z "$NAME" ] && { echo "usage: switch_map_mode.sh localize <name>"; exit 2; }
    [ -f "$MAPS/$NAME.yaml" ] || { echo "no such map: $NAME"; exit 2; }
    log "→ ΕΝΤΟΠΙΣΜΟΣ σε $NAME (hot swap, το υπόλοιπο stack μένει πάνω)"
    kill_nodes $LOCAL_NODES $SLAM_NODES
    wait_gone $LOCAL_NODES $SLAM_NODES
    setsid nohup ros2 launch home_robot map_mode_localize.launch.py "map:=$NAME" \
      >> "$LOG" 2>&1 < /dev/null &
    echo "switching to $NAME (hot swap)"
    ;;

  *)
    echo "usage: switch_map_mode.sh {slam|localize <name>}"
    exit 2
    ;;
esac
