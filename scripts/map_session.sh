#!/usr/bin/env bash
# Map management for the web dashboard: switch the active map, start a new
# mapping run, or save the map being built.
#
#   map_session.sh switch <name> [extra launch args...]
#   map_session.sh new    [extra launch args...]
#   map_session.sh save   <name>
#
# ‼️ `switch` and `new` TEAR THE WHOLE STACK DOWN and bring it back up — the map
# is baked into map_server at launch and there is no runtime way to swap it. The
# dashboard therefore asks for confirmation before calling either, and the
# browser reconnects on its own once the new dashboard is listening (~90 s).
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
ROBOT="$HOME/bin/robot"
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
    shift 2
    log "switching to $NAME ($*)"
    "$ROBOT" stop >> "$LOG" 2>&1
    # setsid: this script is about to be replaced by the launch, and the launch
    # must not die with the HTTP request that started it.
    setsid nohup "$ROBOT" max "map:=$NAME" "$@" >> "$LOG" 2>&1 < /dev/null &
    echo "switching to $NAME"
    ;;

  new)
    shift
    log "starting a new mapping run ($*)"
    "$ROBOT" stop >> "$LOG" 2>&1
    # SLAM, not localization: bringup's own default. use_dashboard is false in
    # bringup (localize is what normally turns it on), so ask for it explicitly
    # or the user loses the UI they started the mapping from.
    #
    # ‼️ use_joy is spelled out for the same reason, and it is the whole point
    # of a mapping run: you drive the house by hand with the PS5 pad. It used
    # to be left at bringup's default, which was FALSE — no joy_node started,
    # so the pad did nothing for the entire run while the dashboard's
    # arrow buttons still drove the base, which is exactly how it was reported.
    # bringup now defaults it true as well; kept explicit so a future change to
    # that default cannot silently take the controller away again.
    sudo systemctl start ros-sllidar-c1.service 2>/dev/null
    setsid nohup ros2 launch home_robot bringup.launch.py \
      use_slam:=true use_dashboard:=true use_arm:=true use_joy:=true "$@" \
      >> "$LOG" 2>&1 < /dev/null &
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
    if timeout 10 ros2 service list 2>/dev/null | grep -q '/slam_toolbox/serialize_map'; then
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
