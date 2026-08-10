#!/usr/bin/env bash
# Restart the stack after the dashboard has already patched nav2_params.yaml's
# two keepout_filter `enabled:` lines (home_robot/keepout_toggle.py). This
# script's only job is the restart — it does not touch the YAML itself, so a
# failed patch (caught in Python, before this is ever spawned) never gets
# this far.
#
# Any arguments (e.g. use_keepout:=true, use_perception:=true) are carried
# straight through to `robot max`, same as scripts/apply_skirt_margin.sh and
# scripts/map_session.sh's `switch` branch — see either for why: without this
# the relaunch silently reverts perception/LLM backend to defaults, and
# without use_keepout:=true specifically, the newly-enabled costmap layer
# would sit WARNing "Filter mask was not received" forever (filter_mask_server
# only starts when that flag is passed at launch).
#
# ‼️ Nav2 reads costmap plugin config once in on_configure, so nothing short
# of a full relaunch picks up the enabled: change. Mirrors
# scripts/apply_skirt_margin.sh exactly (see it for setsid/nohup and why this
# must live under src, not install).
set -o pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT="$HOME/bin/robot"
LOG=/tmp/apply_keepout_toggle.log

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

source /opt/ros/jazzy/setup.bash
source "$HOME/robot_ws/install/setup.bash"
export ROS_DOMAIN_ID=0

log "restarting with new keepout state ($*)"
"$ROBOT" stop >> "$LOG" 2>&1
setsid nohup "$ROBOT" max "$@" >> "$LOG" 2>&1 < /dev/null &
echo "restarting"
