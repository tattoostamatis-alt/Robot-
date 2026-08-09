#!/usr/bin/env bash
# Restart the stack after the dashboard has already patched nav2_params.yaml's
# collision_monitor Stop polygon (home_robot/collision_skirt.py). This script's
# only job is the restart — it does not touch the YAML itself, so a failed
# patch (caught in Python, before this is ever spawned) never gets this far.
# Any arguments (e.g. use_perception:=true) are carried straight through to
# `robot max`, the same way map_session.sh forwards them, so applying a margin
# does not silently revert the perception/LLM backend choice.
#
# ‼️ Nav2 reads the Stop polygon once in on_configure, so nothing short of a
# full relaunch picks up a new margin — same limitation documented in
# home_robot/safety_settings.py for inflation_radius vs. this one. Mirrors
# scripts/map_session.sh's `switch` branch exactly: same restart shape, same
# reasons (see that file for the full explanation of setsid/nohup and why this
# script must live under src, not install).
set -o pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT="$HOME/bin/robot"
LOG=/tmp/apply_skirt_margin.log

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

source /opt/ros/jazzy/setup.bash
source "$HOME/robot_ws/install/setup.bash"
export ROS_DOMAIN_ID=0

log "restarting with new collision skirt margin ($*)"
"$ROBOT" stop >> "$LOG" 2>&1
# setsid: this script is about to be replaced by the launch, and the launch
# must not die with the HTTP request that started it.
setsid nohup "$ROBOT" max "$@" >> "$LOG" 2>&1 < /dev/null &
echo "restarting"
