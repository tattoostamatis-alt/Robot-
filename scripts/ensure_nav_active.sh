#!/usr/bin/env bash
# Self-heal the Nav2 lifecycle chain: activate any managed node that is not
# `active`. Normally navigation_launch autostarts everything and this is a
# no-op, but on 2026-07-03 part of the chain (behavior_server,
# velocity_smoother, collision_monitor) came up inactive under full-stack
# boot load and goals silently went nowhere. Run by a TimerAction in
# bringup.launch.py; safe to run by hand anytime.
#
# 2026-07-08: the 60s TimerAction fired but planner_server activation kept
# timing out under boot load — 3 attempts exhausted before the node settled,
# so the script gave up (exit 1) and every goal failed with "Costmap timed
# out waiting for update" / planning from (0,0). Running it by hand minutes
# later (idle system) activated instantly. Fix: be far more persistent —
# ATTEMPTS spans several minutes so it keeps retrying until the stack calms
# down, and each lifecycle transition gets a longer timeout.
set -uo pipefail

NODES="controller_server smoother_server planner_server route_server
       behavior_server velocity_smoother collision_monitor bt_navigator
       waypoint_follower docking_server"

ATTEMPTS=12       # was 3 — boot load can keep planner busy well past 60s
SET_TIMEOUT=30    # was 20 — activate can be slow while costmaps load
SLEEP_BETWEEN=12  # was 10 — let the stack settle between retries

for attempt in $(seq 1 $ATTEMPTS); do
  all_active=true
  for n in $NODES; do
    state=$(timeout 5 ros2 lifecycle get /$n 2>/dev/null | cut -d' ' -f1)
    case "$state" in
      active) ;;
      unconfigured)
        all_active=false
        echo "[ensure_nav_active] /$n unconfigured -> configure+activate (attempt $attempt/$ATTEMPTS)"
        timeout $SET_TIMEOUT ros2 lifecycle set /$n configure >/dev/null 2>&1
        timeout $SET_TIMEOUT ros2 lifecycle set /$n activate  >/dev/null 2>&1
        ;;
      inactive)
        all_active=false
        echo "[ensure_nav_active] /$n inactive -> activate (attempt $attempt/$ATTEMPTS)"
        timeout $SET_TIMEOUT ros2 lifecycle set /$n activate >/dev/null 2>&1
        ;;
      *)
        # not discovered (yet) or mid-transition — recheck next attempt
        all_active=false
        echo "[ensure_nav_active] /$n state '$state' — will recheck (attempt $attempt/$ATTEMPTS)"
        ;;
    esac
  done
  $all_active && { echo "[ensure_nav_active] nav chain fully active"; exit 0; }
  sleep $SLEEP_BETWEEN
done

echo "[ensure_nav_active] WARNING: nav chain still not fully active after $ATTEMPTS attempts"
exit 1
