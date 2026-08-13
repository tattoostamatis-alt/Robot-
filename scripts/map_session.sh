#!/usr/bin/env bash
# Map management for the web dashboard: switch the active map, start a new
# mapping run, or save the map being built.
#
#   map_session.sh switch <name> [extra launch args...]
#   map_session.sh new    [extra launch args...]
#   map_session.sh save   <name>
#
# `switch` and `new` hot-swap via switch_map_mode.sh — only the
# localization/SLAM nodes (map_server, amcl, slam_toolbox, pose_saver,
# global_localizer, room_markers, the AprilTag relocalizer) are killed and
# replaced; voice, camera, arm, dashboard, VNC and Foxglove stay up the whole
# time, so the browser never disconnects. Before 2026-08-10 both did
# `robot stop` + a full relaunch (~90s, dashboard/voice dropped and
# reconnected on their own) — see project_robot_new_map_half_stack memory for
# why that existed in the first place and project_robot_map_hotswap for why it
# no longer needs to be that heavy. `[extra launch args...]` (use_perception,
# llm_backend) is accepted here for backward compatibility but ignored: those
# nodes are no longer touched by a hot swap, so there is nothing left to carry
# forward.
#
# ‼️ This script MUST live under robot_ws/src, not the install tree: `robot stop`
# enumerates victims by matching `/opt/ros/jazzy` or `robot_ws/install` in the
# command line, so a copy under install/ would kill the very script that is
# calling it, halfway through.
# ‼️ NOT `set -u`: ROS's own setup.bash reads AMENT_TRACE_SETUP_FILES without
# defining it first, so nounset makes every command here die on line 8 of a
# file we do not own. Cost an entire save that reported "unbound variable".
set -o pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAPS="$SRC_DIR/maps"
SWITCH_MODE_SH="$SRC_DIR/scripts/switch_map_mode.sh"
LOG=/tmp/map_session.log

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

source /opt/ros/jazzy/setup.bash
source "$HOME/robot_ws/install/setup.bash"
export ROS_DOMAIN_ID=0

case "${1:-}" in

  switch)
    NAME="${2:-}"
    [ -z "$NAME" ] && { echo "usage: map_session.sh switch <name>"; exit 2; }
    [ -f "$MAPS/$NAME.yaml" ] || { echo "no such map: $NAME"; exit 2; }
    log "switching to $NAME (hot swap)"
    bash "$SWITCH_MODE_SH" localize "$NAME" >> "$LOG" 2>&1
    echo "switching to $NAME"
    ;;

  new)
    log "starting a new mapping run (hot swap)"
    bash "$SWITCH_MODE_SH" slam >> "$LOG" 2>&1
    echo "mapping started"
    ;;

  save)
    NAME="${2:-}"
    [ -z "$NAME" ] && { echo "usage: map_session.sh save <name>"; exit 2; }
    case "$NAME" in *[!A-Za-z0-9_-]*) echo "bad name: $NAME"; exit 2;; esac
    mkdir -p "$MAPS"
    log "saving map $NAME"
    # Two halves, and BOTH are needed. map_saver_cli writes the .pgm/.yaml that
    # map_server loads next time; serialize_map writes the .posegraph/.data that
    # let slam_toolbox resume and EXTEND this map instead of starting empty.
    timeout 90 ros2 run nav2_map_server map_saver_cli -f "$MAPS/$NAME" >> "$LOG" 2>&1
    RC=$?
    # ‼️ Only when slam_toolbox is actually up. Outside a mapping run the
    # service does not exist and `ros2 service call` BLOCKS FOREVER waiting for
    # it — the save succeeded on disk while the browser waited for a reply that
    # never came. Saving a map from localize mode (no posegraph) is legitimate,
    # so this is a skip, not an error.
    # ‼️ The probe used to be `timeout 10 ros2 service list | grep serialize_map`
    # and it FALSE-NEGATIVED on 2026-08-08: the branch went to the else and the
    # save logged "slam_toolbox not running" while slam_toolbox had been mapping
    # for 34 minutes. The .pgm was written, the .posegraph/.data were NOT — an
    # unresumable map that looked saved. Measured on the mapping stack (71 nodes,
    # load ~40): the FIRST `ros2 service list` after a cold daemon took 9.5 s,
    # later ones 1-3 s. So 10 s was not a generous margin, it was a coin flip,
    # and it lands on the wrong side exactly when the stack is busy — which is
    # always, during a mapping run. pgrep on the executable answers instantly and
    # cannot time out; the `timeout 60` on the call below is what still guards
    # the original blocking case.
    if pgrep -f async_slam_toolbox_node >/dev/null 2>&1; then
      timeout 60 ros2 service call /slam_toolbox/serialize_map \
        slam_toolbox/srv/SerializePoseGraph "{filename: '$MAPS/$NAME'}" \
        >> "$LOG" 2>&1 || log "serialize_map failed"
    else
      log "slam_toolbox not running — saved image only, map will not be resumable"
    fi
    if [ $RC -eq 0 ] && [ -f "$MAPS/$NAME.yaml" ]; then
      echo "saved $NAME"
    else
      echo "save failed — see $LOG"
      exit 1
    fi
    ;;

  *)
    echo "usage: map_session.sh {switch <name>|new|save <name>}"
    exit 2
    ;;
esac
